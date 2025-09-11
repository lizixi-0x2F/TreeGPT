#!/usr/bin/env python3
"""
手动评估检查 - 加载最佳检查点并在eval集合中抽几道题测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader

# Import our modules
from src.TreeGPT import *
from src.arc_treegpt import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_best_model():
    """加载最佳检查点"""
    device = 'cpu'  # 使用CPU避免MPS内存问题
    
    # 加载检查点
    checkpoint = torch.load('best_treeffn_seq2seq.pth', map_location=device)
    config = checkpoint['config']
    
    logger.info(f"📊 Loaded checkpoint from step {checkpoint['step']} (epoch {checkpoint['epoch']})")
    logger.info(f"  Train loss: {checkpoint['train_loss']:.4f}")
    logger.info(f"  Token acc: {checkpoint['token_acc']:.4f}")
    logger.info(f"  Full acc: {checkpoint['full_acc']:.4f}")
    logger.info(f"  Val token acc: {checkpoint['val_token_acc']:.4f}")
    logger.info(f"  Val full acc: {checkpoint['val_full_acc']:.4f}")
    
    # 创建模型
    from train_no_attention import TreeFFNSeq2SeqARC
    model = TreeFFNSeq2SeqARC(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"🚀 Model loaded on {device}")
    logger.info(f"  Config: {config}")
    
    return model, device, checkpoint

def manual_evaluation(model, device, num_samples=5):
    """手工评估几个样本"""
    logger.info(f"🔍 Manual evaluation on {num_samples} samples")
    
    # 加载eval数据
    eval_dataset = ARCDataset('arc-prize-2025/arc-agi_evaluation_challenges.json', max_length=8192)
    tokenizer = ARCGridTokenizer()
    
    model.eval()
    
    # 随机选择几个样本
    indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Sample {i+1}/{num_samples} (Index: {idx})")
            
            # 获取样本
            sample = eval_dataset[idx]
            sample_id = sample['sample_id']
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            
            logger.info(f"  Sample ID: {sample_id}")
            logger.info(f"  Input length: {sample['input_length']}")
            logger.info(f"  Sequence length: {sample['seq_length']}")
            
            # 创建输入输出
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            # 前向传播
            logits = model(inputs)
            
            # 预测
            predictions = torch.argmax(logits, dim=-1)
            
            # 计算准确率
            mask = (targets != 16)  # 非padding token
            correct = (predictions == targets) & mask
            token_acc = (correct.sum().float() / mask.sum().float()).item() if mask.sum() > 0 else 0.0
            
            # 检查完整序列准确率
            batch_correct = True
            if mask.sum() > 0:
                sample_predictions = predictions[0][mask[0]]
                sample_targets = targets[0][mask[0]]
                batch_correct = torch.all(sample_predictions == sample_targets).item()
            full_acc = 1.0 if batch_correct else 0.0
            
            logger.info(f"  Token accuracy: {token_acc:.4f}")
            logger.info(f"  Full accuracy: {full_acc:.4f}")
            
            # 显示一些具体的预测vs目标对比
            if mask.sum() > 10:  # 只有足够长的序列才显示
                valid_preds = predictions[0][mask[0]][:20]  # 显示前20个有效token
                valid_targets = targets[0][mask[0]][:20]
                
                logger.info(f"  Predictions (first 20): {valid_preds.cpu().tolist()}")
                logger.info(f"  Targets    (first 20): {valid_targets.cpu().tolist()}")
                
                # 找出不匹配的位置
                mismatches = []
                for j, (pred, target) in enumerate(zip(valid_preds, valid_targets)):
                    if pred != target:
                        mismatches.append(f"pos_{j}: pred={pred.item()}, target={target.item()}")
                
                if mismatches:
                    logger.info(f"  Mismatches: {', '.join(mismatches)}")
                else:
                    logger.info(f"  ✅ Perfect match in first 20 tokens!")
            
            # 尝试解码输出网格（如果这是完整的ARC任务）
            try:
                # 获取原始数据以了解任务结构
                with open('arc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
                    original_data = json.load(f)
                
                if sample_id in original_data:
                    task_data = original_data[sample_id]
                    logger.info(f"  Task has {len(task_data['train'])} training examples")
                    
                    # 显示输入网格大小
                    test_input = task_data['test'][0]['input']
                    logger.info(f"  Test input grid size: {len(test_input)}x{len(test_input[0])}")
                    
                    # 如果有ground truth输出，显示期望的输出网格大小
                    if 'output' in task_data['test'][0]:
                        test_output = task_data['test'][0]['output']
                        logger.info(f"  Test output grid size: {len(test_output)}x{len(test_output[0])}")
            except Exception as e:
                logger.info(f"  Could not decode task structure: {e}")

def main():
    """主函数"""
    logger.info("🔍 Starting manual evaluation check")
    
    # 加载模型
    model, device, checkpoint = load_best_model()
    
    # 手动评估
    manual_evaluation(model, device, num_samples=5)
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ Manual evaluation completed!")
    logger.info(f"📊 Model was saved at step {checkpoint['step']} with:")
    logger.info(f"  - Validation token accuracy: {checkpoint['val_token_acc']:.4f}")
    logger.info(f"  - Validation full accuracy: {checkpoint['val_full_acc']:.4f}")
    logger.info("🔍 Check the detailed outputs above for any factual errors")

if __name__ == "__main__":
    main()