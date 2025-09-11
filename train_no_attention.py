#!/usr/bin/env python3
"""
纯TreeFFN Seq2Seq实验 - Encoder-Decoder架构
完全消融attention和自回归，只用TreeFFN组件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Union, Tuple, Optional

from src.TreeGPT import *
from src.arc_treegpt import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeFFNSeq2SeqBlock(nn.Module):
    """
    简化的TreeFFN Seq2Seq Block - 只有Encoder + Decoder TreeFFN
    """
    
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 tree_iterations: int = 3):
        super().__init__()
        
        # 编码器TreeFFN - 处理输入序列
        self.encoder_tree_ffn = TreeFFN(
            d_in=d_model,
            d_h=d_model,
            num_node_classes=None,
            num_tree_classes=None,
            use_edge_proj=True,
            use_gating=True,
            residual=True,
            dropout=dropout,
            tree_iterations=tree_iterations
        )
        
        # 解码器TreeFFN - 生成输出序列
        self.decoder_tree_ffn = TreeFFN(
            d_in=d_model,
            d_h=d_model,
            num_node_classes=None,
            num_tree_classes=None,
            use_edge_proj=True,
            use_gating=True,
            residual=True,
            dropout=dropout,
            tree_iterations=tree_iterations
        )
        
        # LayerNorms
        self.ln_encoder = nn.LayerNorm(d_model)
        self.ln_decoder = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的并行seq2seq前向传播 - 只有编码器和解码器
        x: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # 1. 编码器：从左到右处理输入序列
        encoder_edges = self._create_encoder_edges(seq_len, device)
        encoder_outputs = []
        
        h_enc = self.ln_encoder(x)
        for b in range(batch_size):
            h_b = h_enc[b]  # [seq_len, d_model]
            enc_out = self.encoder_tree_ffn(h_b, encoder_edges, root_idx=0)
            
            if 'hidden' in enc_out:
                encoder_outputs.append(enc_out['hidden'])
            else:
                encoder_outputs.append(h_b)
        
        encoder_h = torch.stack(encoder_outputs, dim=0)
        h = x + encoder_h
        
        # 2. 解码器：从右到左生成输出序列
        decoder_edges = self._create_decoder_edges(seq_len, device)
        decoder_outputs = []
        
        h_dec = self.ln_decoder(h)
        for b in range(batch_size):
            h_b = h_dec[b]  # [seq_len, d_model]
            dec_out = self.decoder_tree_ffn(h_b, decoder_edges, root_idx=seq_len-1)  # 从最后一个节点开始
            
            if 'hidden' in dec_out:
                decoder_outputs.append(dec_out['hidden'])
            else:
                decoder_outputs.append(h_b)
        
        decoder_h = torch.stack(decoder_outputs, dim=0)
        h = h + decoder_h
        
        return h
    
    def _create_encoder_edges(self, seq_len: int, device: torch.device) -> torch.LongTensor:
        """编码器边：从左到右的相邻连接"""
        edges = []
        for i in range(seq_len - 1):
            edges.append([i, i + 1])
        
        if not edges:
            edges.append([0, 0])
        
        return torch.tensor(edges, dtype=torch.long, device=device).t()
    
    def _create_decoder_edges(self, seq_len: int, device: torch.device) -> torch.LongTensor:
        """解码器边：从右到左的相邻连接"""
        edges = []
        for i in range(seq_len - 1, 0, -1):
            edges.append([i, i - 1])
        
        if not edges:
            edges.append([0, 0])
        
        return torch.tensor(edges, dtype=torch.long, device=device).t()


class TreeFFNSeq2Seq(nn.Module):
    """
    纯TreeFFN的Seq2Seq模型 - 完全并行，无自回归
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 n_layers: int = 3,
                 max_seq_len: int = 8192,
                 dropout: float = 0.1,
                 tree_iterations: int = 3):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # TreeFFN Seq2Seq blocks
        self.blocks = nn.ModuleList([
            TreeFFNSeq2SeqBlock(
                d_model=d_model,
                dropout=dropout,
                tree_iterations=tree_iterations
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        纯并行seq2seq前向传播 - 一次处理整个序列
        input_ids: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 位置编码
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        
        # Token + Position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        h = token_emb + pos_emb
        
        # 通过TreeFFN Seq2Seq blocks
        for block in self.blocks:
            h = block(h)
        
        # 最终层归一化和输出投影
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits


class TreeFFNSeq2SeqARC(TreeFFNSeq2Seq):
    """ARC任务的TreeFFN Seq2Seq"""
    
    def __init__(self, vocab_size: int = 17, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)


def calculate_full_accuracy(logits, targets, ignore_index=16):
    """计算完整序列准确率"""
    predictions = torch.argmax(logits, dim=-1)
    mask = (targets != ignore_index)
    
    batch_size = targets.size(0)
    full_correct = 0
    
    for i in range(batch_size):
        sample_mask = mask[i]
        if sample_mask.sum() == 0:
            continue
        sample_predictions = predictions[i][sample_mask]
        sample_targets = targets[i][sample_mask]
        if torch.all(sample_predictions == sample_targets):
            full_correct += 1
    
    return full_correct / batch_size if batch_size > 0 else 0.0


def train_treeffn_seq2seq_model():
    """训练纯TreeFFN Seq2Seq模型 - 完全并行，无自回归"""
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🚀 Training TreeFFN Seq2Seq Model on device: {device}")
    
    # 模型配置 - 纯TreeFFN Seq2Seq
    model_config = {
        'vocab_size': 17,
        'd_model': 256,
        'n_layers': 2,  # 每层有3个TreeFFN，所以层数可以少一些
        'max_seq_len': 8192,
        'dropout': 0.1,
        'tree_iterations': 2,  # 稍微少一些加快训练
    }
    
    # 创建TreeFFN Seq2Seq模型
    model = TreeFFNSeq2SeqARC(**model_config).to(device)
    
    # 优化器配置 - 为所有TreeFFN的T参数设置特殊学习率
    tree_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'tree_ffn.T' in name:
            tree_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': 1e-4},
        {'params': tree_params, 'lr': 1e-3}   # T参数更高学习率
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载数据
    train_dataset = ARCDataset('arc-prize-2025/arc-agi_training_challenges.json', max_length=8192)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # 可以稍大一些，因为是并行处理
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    eval_dataset = ARCDataset('arc-prize-2025/arc-agi_evaluation_challenges.json', max_length=8192)
    tokenizer = ARCGridTokenizer()
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # 训练循环
    model.train()
    epochs = 5
    best_val_acc = 0.0
    global_step = 0
    validation_interval = 300  # 每300步验证一次
    
    training_history = {
        'epochs': [],
        'train_losses': [],
        'token_accs': [],
        'full_accs': [],
        'val_token_accs': [],
        'val_full_accs': [],
        'lr': [],
        'steps': []
    }
    
    logger.info("🎯 Starting TreeFFN Encoder-Decoder Training:")
    logger.info(f"  Model: {model_config}")
    logger.info(f"  Architecture: Encoder TreeFFN + Decoder TreeFFN only")
    logger.info(f"  Pure parallel processing - no attention, no autoregression")
    logger.info(f"  Each block has 2 TreeFFN components with {model_config['tree_iterations']} iterations")
    logger.info(f"  Encoder: left-to-right, Decoder: right-to-left")
    logger.info(f"  Total TreeFFN components: {model_config['n_layers'] * 2}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_token_acc = 0
        epoch_full_acc = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            global_step += 1
            
            input_ids = batch['input_ids'].to(device)
            seq_lengths = batch['seq_lengths']
            
            # 创建目标
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            # 创建mask - GPU向量化操作
            batch_size, seq_len = targets.shape
            mask = torch.zeros_like(targets, dtype=torch.bool)
            
            seq_lengths_tensor = torch.tensor(seq_lengths, device=device, dtype=torch.long)
            pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            actual_lengths = torch.clamp(seq_lengths_tensor.unsqueeze(1) - 1, 0, seq_len)
            mask = pos_indices < actual_lengths
            
            optimizer.zero_grad()
            
            # 纯并行前向传播 - 一次处理整个序列，无自回归
            logits = model(inputs)
            
            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=16,
                reduction='none'
            )
            loss = loss.view_as(targets)
            loss = (loss * mask.float()).sum() / mask.float().sum()
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == targets) & mask
            total_correct = correct.sum()
            total_valid = mask.sum()
            token_acc = (total_correct.float() / total_valid.float()).item() if total_valid > 0 else 0.0
            
            full_acc = calculate_full_accuracy(logits, targets, ignore_index=16)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_token_acc += token_acc
            epoch_full_acc += full_acc
            num_batches += 1
            
            # 计算所有TreeFFN的平均T值 - 现在每层只有2个TreeFFN
            all_T_values = []
            for block in model.blocks:
                all_T_values.append(block.encoder_tree_ffn.T.item())
                all_T_values.append(block.decoder_tree_ffn.T.item())
            avg_T = sum(all_T_values) / len(all_T_values)
            
            progress_bar.set_postfix({
                'step': global_step,
                'loss': f'{loss.item():.4f}',
                'token_acc': f'{token_acc:.4f}',
                'full_acc': f'{full_acc:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'T_avg': f'{avg_T:.2f}'
            })
            
            # 简单内存管理
            if global_step % 50 == 0:
                if device == 'mps':
                    torch.mps.empty_cache()
                elif device == 'cuda':
                    torch.cuda.empty_cache()
            
            # 验证
            if global_step % validation_interval == 0:
                logger.info(f"\\n🎯 Step {global_step}: Running validation...")
                val_token_acc, val_full_acc, val_samples = evaluate_treeffn_seq2seq_model(
                    model, eval_dataset, tokenizer, device, max_samples=100
                )
                
                training_history['steps'].append(global_step)
                training_history['epochs'].append(epoch + 1)
                training_history['train_losses'].append(loss.item())
                training_history['token_accs'].append(token_acc)
                training_history['full_accs'].append(full_acc)
                training_history['val_token_accs'].append(val_token_acc)
                training_history['val_full_accs'].append(val_full_acc)
                training_history['lr'].append(optimizer.param_groups[0]['lr'])
                
                logger.info(f"Step {global_step} (Epoch {epoch+1}):")
                logger.info(f"  Current Loss: {loss.item():.4f}")
                logger.info(f"  Current Token Acc: {token_acc:.4f}")
                logger.info(f"  Current Full Acc: {full_acc:.4f}")
                logger.info(f"  Val Token Acc: {val_token_acc:.4f} ({val_samples} samples)")
                logger.info(f"  Val Full Acc: {val_full_acc:.4f} ({val_samples} samples)")
                
                # 保存最佳模型
                if val_full_acc > best_val_acc:
                    best_val_acc = val_full_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': global_step,
                        'train_loss': loss.item(),
                        'token_acc': token_acc,
                        'full_acc': full_acc,
                        'val_token_acc': val_token_acc,
                        'val_full_acc': val_full_acc,
                        'config': model_config
                    }, 'best_treeffn_seq2seq.pth')
                    logger.info(f"💾 Saved best model with val full acc {val_full_acc:.4f}")
                
                logger.info("-" * 60)
        
        scheduler.step()
        
        # Epoch统计
        avg_loss = epoch_loss / num_batches
        avg_token_acc = epoch_token_acc / num_batches
        avg_full_acc = epoch_full_acc / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        # 统计所有TreeFFN的T值 - 现在每层只有2个TreeFFN
        all_T_values = []
        for block in model.blocks:
            all_T_values.append(block.encoder_tree_ffn.T.item())
            all_T_values.append(block.decoder_tree_ffn.T.item())
        
        avg_T = sum(all_T_values) / len(all_T_values)
        min_T = min(all_T_values)
        max_T = max(all_T_values)
        
        logger.info(f"\\n📊 Epoch {epoch+1}/{epochs} Summary:")
        logger.info(f"  Average Train Loss: {avg_loss:.4f}")
        logger.info(f"  Average Train Token Acc: {avg_token_acc:.4f}")
        logger.info(f"  Average Train Full Acc: {avg_full_acc:.4f}")
        logger.info(f"  LR: {current_lr:.6f}")
        logger.info(f"  TreeFFN Iterations: avg={avg_T:.2f}, min={min_T:.2f}, max={max_T:.2f}")
        logger.info(f"  Total TreeFFN components: {len(all_T_values)} (2 per layer)")
        logger.info(f"  Total Steps: {global_step}")
        
        logger.info("=" * 60)
    
    # 保存训练历史
    with open('training_history_treeffn_seq2seq.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("✅ TreeFFN Seq2Seq training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    return model


def evaluate_treeffn_seq2seq_model(model, eval_dataset, tokenizer, device, max_samples=100):
    """评估TreeFFN Seq2Seq模型"""
    model.eval()
    total_token_acc = 0
    total_full_acc = 0
    num_batches = 0
    
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            seq_lengths = batch['seq_lengths']
            
            try:
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                
                # 创建mask
                batch_size, seq_len = targets.shape
                mask = torch.zeros_like(targets, dtype=torch.bool)
                seq_lengths_tensor = torch.tensor(seq_lengths, device=device, dtype=torch.long)
                pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                actual_lengths = torch.clamp(seq_lengths_tensor.unsqueeze(1) - 1, 0, seq_len)
                mask = pos_indices < actual_lengths
                
                # 纯并行推理
                logits = model(inputs)
                
                # 计算token准确率
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == targets) & mask
                total_correct = correct.sum()
                total_valid = mask.sum()
                token_acc = (total_correct.float() / total_valid.float()).item() if total_valid > 0 else 0.0
                
                # 计算完整序列准确率
                full_acc = calculate_full_accuracy(logits, targets, ignore_index=16)
                
                total_token_acc += token_acc
                total_full_acc += full_acc
                num_batches += 1
                
            except Exception:
                continue
    
    model.train()
    
    avg_token_acc = total_token_acc / num_batches if num_batches > 0 else 0.0
    avg_full_acc = total_full_acc / num_batches if num_batches > 0 else 0.0
    
    return avg_token_acc, avg_full_acc, num_batches


def main():
    """主函数"""
    logger.info("🚀 Starting TreeFFN Seq2Seq Experiment")
    start_time = time.time()
    
    # 训练模型
    model = train_treeffn_seq2seq_model()
    
    total_time = time.time() - start_time
    logger.info(f"\\n🎉 Experiment completed in {total_time:.2f} seconds")
    
    logger.info("\\n" + "="*60)
    logger.info("TREEFFN ENCODER-DECODER EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info("✅ Pure TreeFFN architecture - no attention, no autoregression")
    logger.info("✅ Encoder TreeFFN (left-to-right) + Decoder TreeFFN (right-to-left)")
    logger.info("✅ Completely parallel processing - maximum speed")
    logger.info("✅ Simple architecture - only 2 TreeFFN per layer")
    logger.info("✅ Model trained and saved")
    logger.info("📊 Check 'training_history_treeffn_seq2seq.json' for results")
    logger.info("📝 Compare with complex architectures")
    logger.info("🚀 Simplest and fastest TreeFFN setup!")
    logger.info("="*60)


if __name__ == "__main__":
    main()