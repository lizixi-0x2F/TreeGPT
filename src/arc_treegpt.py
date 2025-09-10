import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging

try:
    from .TreeGPT import TreeGPT
except ImportError:
    from TreeGPT import TreeGPT

logger = logging.getLogger(__name__)


class ARCGridTokenizer:
    """
    ARC网格序列化tokenizer
    将网格扁平化为序列，添加特殊token用于分隔
    """
    
    def __init__(self):
        # ARC颜色值 0-9 + 特殊token
        self.color_tokens = list(range(10))  # 0-9 颜色
        self.GRID_START = 10  # 网格开始
        self.GRID_END = 11    # 网格结束
        self.ROW_SEP = 12     # 行分隔符
        self.EXAMPLE_SEP = 13 # 例题分隔符
        self.INPUT_OUTPUT_SEP = 14  # 输入输出分隔符
        self.TEST_START = 15  # 测试开始
        self.PAD = 16         # 填充
        
        self.vocab_size = 17
        
    def grid_to_sequence(self, grid: List[List[int]]) -> List[int]:
        """将单个网格转换为序列"""
        sequence = [self.GRID_START]
        for i, row in enumerate(grid):
            if i > 0:
                sequence.append(self.ROW_SEP)
            sequence.extend(row)
        sequence.append(self.GRID_END)
        return sequence
    
    def encode_arc_sample(self, sample: Dict) -> List[int]:
        """
        编码ARC样本: exam_in<sep>exam_out<sep>exam_in<sep>exam_out<sep>...test_in
        """
        sequence = []
        
        # 编码训练例题
        for i, example in enumerate(sample['train']):
            if i > 0:
                sequence.append(self.EXAMPLE_SEP)
            
            # 输入网格
            sequence.extend(self.grid_to_sequence(example['input']))
            sequence.append(self.INPUT_OUTPUT_SEP)
            # 输出网格
            sequence.extend(self.grid_to_sequence(example['output']))
        
        # 测试部分
        sequence.append(self.TEST_START)
        sequence.extend(self.grid_to_sequence(sample['test'][0]['input']))
        
        return sequence
    
    def decode_grid(self, sequence: List[int]) -> List[List[int]]:
        """从序列解码回网格"""
        if not sequence:
            return [[0]]
            
        # 找到网格边界
        try:
            start_idx = sequence.index(self.GRID_START) + 1
            end_idx = sequence.index(self.GRID_END)
            grid_seq = sequence[start_idx:end_idx]
        except ValueError:
            # 如果没有找到边界token，直接处理整个序列
            grid_seq = [t for t in sequence if t in self.color_tokens]
            if not grid_seq:
                return [[0]]
        
        # 按行分隔符分割
        rows = []
        current_row = []
        
        for token in grid_seq:
            if token == self.ROW_SEP:
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif token in self.color_tokens:
                current_row.append(token)
        
        if current_row:
            rows.append(current_row)
        
        # 确保至少有一行
        if not rows:
            return [[0]]
        
        return rows
    
    def decode_sequence(self, sequence: np.ndarray) -> List[List[int]]:
        """
        Decode a token sequence back to grid format
        This method is used by the prediction generation pipeline
        """
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        
        # Try to find the last complete grid in the sequence
        # Look for TEST_START marker and decode what follows
        try:
            if self.TEST_START in sequence:
                test_start_idx = sequence.index(self.TEST_START)
                # Everything after TEST_START should be the prediction
                prediction_seq = sequence[test_start_idx + 1:]
            else:
                # If no TEST_START, try to decode the entire sequence
                prediction_seq = sequence
            
            # Use existing decode_grid method
            grid = self.decode_grid(prediction_seq)
            return grid
            
        except Exception as e:
            # Fallback to a simple 2x2 grid if decoding fails
            logger.warning(f"Failed to decode sequence, using fallback: {e}")
            return [[0, 0], [0, 0]]


class ARCDataset(Dataset):
    """ARC数据集 - 动态序列长度，无padding"""
    
    def __init__(self, data_path: str, solutions_path: str = None, max_length: int = 2048):
        self.tokenizer = ARCGridTokenizer()
        self.max_length = max_length
        self.samples = []
        
        # 加载数据
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.challenges = data
        
        # 加载解决方案（如果提供）
        self.solutions = None
        if solutions_path:
            with open(solutions_path, 'r') as f:
                self.solutions = json.load(f)
        
        self.sample_ids = list(data.keys())
        
        logger.info(f"Loaded {len(self.sample_ids)} ARC samples")
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample = self.challenges[sample_id]
        
        # 编码输入序列
        input_sequence = self.tokenizer.encode_arc_sample(sample)
        
        # 检查是否有解决方案
        has_solution = False
        if self.solutions and sample_id in self.solutions:
            # 使用解决方案文件中的答案
            target_grid = self.solutions[sample_id][0]  # 取第一个解决方案
            target_sequence = self.tokenizer.grid_to_sequence(target_grid)
            full_sequence = input_sequence + target_sequence
            has_solution = True
        elif 'output' in sample['test'][0]:
            # 使用测试数据中的答案（评估模式）
            target_grid = sample['test'][0]['output']
            target_sequence = self.tokenizer.grid_to_sequence(target_grid)
            full_sequence = input_sequence + target_sequence
            has_solution = True
        else:
            # 没有答案，只有输入（纯推理模式）
            full_sequence = input_sequence
        
        # 动态长度：只截断，不填充
        if len(full_sequence) > self.max_length:
            full_sequence = full_sequence[:self.max_length]
        
        return {
            'input_ids': torch.tensor(full_sequence, dtype=torch.long),
            'sample_id': sample_id,
            'input_length': len(input_sequence),
            'seq_length': len(full_sequence),  # 实际序列长度
            'has_solution': has_solution
        }


class ARCTreeGPT(TreeGPT):
    """为ARC任务特化的TreeGPT"""
    
    def __init__(self, vocab_size: int = 17, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)
        
    def generate_arc_solution(self, 
                             input_sequence: torch.Tensor,
                             tokenizer: ARCGridTokenizer,
                             max_grid_tokens: int = 2048,
                             temperature: float = 0.1) -> List[List[int]]:
        """
        为ARC任务生成解决方案 - 更确定性的生成
        """
        self.eval()
        device = input_sequence.device
        
        # 开始生成输出网格
        current_seq = input_sequence.clone()
        
        with torch.no_grad():
            for _ in range(max_grid_tokens):
                # 截断到最大长度
                if current_seq.size(1) > self.max_seq_len:
                    current_seq = current_seq[:, -self.max_seq_len:]
                
                logits = self(current_seq)
                next_token_logits = logits[:, -1, :]
                
                # 使用低温度采样或贪心解码
                if temperature <= 0.01:
                    # 贪心解码
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    # 低温度采样
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # 如果生成了网格结束符，停止生成
                if next_token.item() == tokenizer.GRID_END:
                    break
        
        # 解码生成的序列
        generated_seq = current_seq[0].cpu().tolist()
        
        # 找到输出网格部分
        try:
            # 找到最后一个网格开始的位置
            grid_starts = [i for i, token in enumerate(generated_seq) if token == tokenizer.GRID_START]
            if grid_starts:
                last_grid_start = grid_starts[-1]
                output_tokens = generated_seq[last_grid_start:]
                grid = tokenizer.decode_grid(output_tokens)
                return grid
        except:
            pass
        
        # 回退：返回简单的1x1网格
        return [[0]]


def collate_fn(batch):
    """动态长度collate函数 - 按批次内最大长度padding"""
    # 找到批次内最大长度
    max_length = max(item['seq_length'] for item in batch)
    
    input_ids = []
    sample_ids = []
    input_lengths = []
    seq_lengths = []
    has_solutions = []
    
    for item in batch:
        seq = item['input_ids']
        current_length = len(seq)
        
        # 只在批次内padding到最大长度
        if current_length < max_length:
            # 使用PAD token (16) 填充
            padding = torch.full((max_length - current_length,), 16, dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
            
        input_ids.append(padded_seq)
        sample_ids.append(item['sample_id'])
        input_lengths.append(item['input_length'])
        seq_lengths.append(item['seq_length'])
        has_solutions.append(item['has_solution'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'sample_ids': sample_ids,
        'input_lengths': input_lengths,
        'seq_lengths': seq_lengths,  # 实际长度，用于mask
        'has_solutions': has_solutions
    }


def compute_streaming_autoregressive_loss(model, batch, tokenizer, device):
    """
    高效的流式自回归损失计算 - 批量并行处理
    """
    input_ids = batch['input_ids']
    input_lengths = batch['input_lengths']
    has_solutions = batch['has_solutions']
    
    batch_size = input_ids.size(0)
    
    # 过滤出有效样本
    valid_samples = []
    for b in range(batch_size):
        if not has_solutions[b]:
            continue
            
        full_seq = input_ids[b]
        input_length = input_lengths[b]
        seq_length = (full_seq != 16).sum().item()
        
        if seq_length > input_length:
            valid_samples.append({
                'input_seq': full_seq[:input_length],
                'target_seq': full_seq[input_length:seq_length],
                'batch_idx': b
            })
    
    if not valid_samples:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 批量处理所有有效样本
    total_loss = 0.0
    
    for sample in valid_samples:
        input_seq = sample['input_seq']
        target_seq = sample['target_seq']
        
        # 创建一个包含输入和部分目标的序列进行单次前向传播
        # 使用标准的language modeling方式：预测下一个token
        full_context = torch.cat([input_seq, target_seq])
        
        if len(full_context) <= 1:
            continue
            
        # 截断到模型最大长度
        if len(full_context) > model.max_seq_len:
            full_context = full_context[-model.max_seq_len:]
            # 重新计算输入和目标的边界
            effective_input_len = min(len(input_seq), model.max_seq_len - 1)
            inputs_part = full_context[:-1]
            targets_part = full_context[1:]
        else:
            inputs_part = full_context[:-1]
            targets_part = full_context[1:]
        
        # 单次前向传播计算整个序列的损失
        if len(inputs_part) > 0:
            logits = model(inputs_part.unsqueeze(0))  # [1, seq_len, vocab_size]
            
            # 只计算目标部分的损失（跳过输入部分）
            target_start = len(input_seq) - 1 if len(input_seq) > 0 else 0
            target_start = max(0, min(target_start, len(logits[0]) - 1))
            
            if target_start < len(logits[0]) and target_start < len(targets_part):
                target_logits = logits[0, target_start:]  # [target_len, vocab_size]
                target_tokens = targets_part[target_start:]  # [target_len]
                
                if len(target_logits) > 0 and len(target_tokens) > 0:
                    # 确保长度匹配
                    min_len = min(len(target_logits), len(target_tokens))
                    target_logits = target_logits[:min_len]
                    target_tokens = target_tokens[:min_len]
                    
                    if min_len > 0:
                        loss = F.cross_entropy(target_logits, target_tokens, reduction='mean')
                        total_loss += loss
    
    return total_loss / len(valid_samples) if valid_samples else torch.tensor(0.0, device=device, requires_grad=True)


def compute_streaming_autoregressive_accuracy(model, batch, tokenizer, device):
    """
    高效的流式自回归准确率计算
    """
    input_ids = batch['input_ids']
    input_lengths = batch['input_lengths']  
    has_solutions = batch['has_solutions']
    
    batch_size = input_ids.size(0)
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    valid_samples = 0
    
    for b in range(batch_size):
        if not has_solutions[b]:
            continue
            
        full_seq = input_ids[b]
        input_length = input_lengths[b]
        seq_length = (full_seq != 16).sum().item()
        
        if seq_length <= input_length:
            continue
            
        input_seq = full_seq[:input_length]
        target_seq = full_seq[input_length:seq_length]
        
        if len(target_seq) == 0:
            continue
        
        # 快速批量生成 - 一次性预测多个token
        current_seq = input_seq.clone()
        predictions = []
        target_tokens = target_seq.cpu().tolist()
        
        with torch.no_grad():
            # 限制生成步数以提高速度
            max_steps = min(len(target_tokens), 100)  # 限制最大步数以提高评估速度
            
            for step in range(max_steps):
                if current_seq.size(0) > model.max_seq_len:
                    current_seq = current_seq[-model.max_seq_len:]
                
                logits = model(current_seq.unsqueeze(0))
                next_token_logits = logits[0, -1, :]
                predicted_token = torch.argmax(next_token_logits, dim=-1)
                predictions.append(predicted_token.item())
                
                # 使用预测的token继续生成
                current_seq = torch.cat([current_seq, predicted_token.unsqueeze(0)])
                
                # 早期停止条件
                if predicted_token.item() == tokenizer.GRID_END:
                    break
        
        # 快速准确率计算
        compare_length = min(len(predictions), len(target_tokens))
        if compare_length > 0:
            correct_in_seq = sum(1 for i in range(compare_length) 
                               if predictions[i] == target_tokens[i])
            
            total_tokens += compare_length
            correct_tokens += correct_in_seq
            
            # 序列级准确率（简化版：只检查前N个token是否匹配）
            if compare_length >= min(10, len(target_tokens)) and correct_in_seq == compare_length:
                correct_sequences += 1
        
        valid_samples += 1
    
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    seq_acc = correct_sequences / valid_samples if valid_samples > 0 else 0.0
    
    return token_acc, seq_acc, valid_samples


# 新的高效流式接口 
def compute_pure_autoregressive_loss(model, batch, tokenizer, device, max_gen_length=None):
    """流式自回归损失 - 使用高效版本"""
    return compute_streaming_autoregressive_loss(model, batch, tokenizer, device)


def compute_pure_autoregressive_accuracy(model, batch, tokenizer, device, max_gen_length=None):
    """流式自回归准确率 - 使用高效版本"""
    return compute_streaming_autoregressive_accuracy(model, batch, tokenizer, device)


# 保持向后兼容的函数名
def compute_autoregressive_loss(model, batch, tokenizer, device, max_gen_length=50):
    """兼容性函数 - 调用流式版本"""
    return compute_streaming_autoregressive_loss(model, batch, tokenizer, device)


def compute_autoregressive_accuracy(model, batch, tokenizer, device, max_gen_length=50):
    """兼容性函数 - 调用流式版本"""
    return compute_streaming_autoregressive_accuracy(model, batch, tokenizer, device)


def train_arc_model():
    """训练ARC模型 - 完全自回归模式"""
    # 必须使用MPS设备
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device not available!")
    
    device = torch.device('mps')
    logger.info(f"Using device: {device}")
    
    # 模型参数 - 大幅减小以适应MPS内存限制
    model_config = {
        'vocab_size': 17,
        'd_model': 64,   # 进一步减小到64
        'n_heads': 2,    # 减到2个头
        'n_layers': 2,   # 只用2层
        'max_seq_len': 256, # 减到256
        'dropout': 0.1,
        'tree_iterations': 1  # 只用1次迭代
    }
    
    # 创建模型
    model = ARCTreeGPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    tokenizer = ARCGridTokenizer()
    
    # 加载训练数据 - 限制最大序列长度
    train_dataset = ARCDataset(
        'arc-prize-2025/arc-agi_training_challenges.json',
        'arc-prize-2025/arc-agi_training_solutions.json',
        max_length=256  # 匹配模型的max_seq_len
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # 进一步减小batch size到1
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    model.train()
    epochs = 5  # 减少训练轮次以节省时间和内存
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            # 自回归损失计算
            loss = compute_autoregressive_loss(
                model, batch, tokenizer, device
            )
            
            if loss.item() > 0:  # 只有当有有效样本时才更新
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}: Average Autoregressive Loss = {avg_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch+1}/{epochs}: No valid batches")
    
    # 保存模型
    torch.save(model.state_dict(), 'arc_treegpt_model.pth')
    logger.info("Model saved to arc_treegpt_model.pth")
    
    return model


def evaluate_arc_model(model: ARCTreeGPT):
    """评估ARC模型"""
    device = next(model.parameters()).device
    tokenizer = ARCGridTokenizer()
    
    # 加载评估数据
    eval_dataset = ARCDataset('arc-prize-2025/arc-agi_evaluation_challenges.json', max_length=8192)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    results = {}
    
    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        sample_id = batch['sample_ids'][0]
        input_length = batch['input_lengths'][0]
        
        # 只用输入部分生成
        input_only = input_ids[:, :input_length]
        
        # 生成解决方案
        predicted_grid = model.generate_arc_solution(input_only, tokenizer)
        
        # 获取真实答案
        with open('arc-prize-2025/arc-agi_evaluation_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        true_grid = solutions[sample_id][0]  # 取第一个解决方案
        
        # 检查预测是否正确
        is_correct = (predicted_grid == true_grid)
        if is_correct:
            correct_predictions += 1
        
        total_samples += 1
        
        results[sample_id] = {
            'predicted': predicted_grid,
            'actual': true_grid,
            'correct': is_correct
        }
        
        if total_samples <= 5:  # 打印前几个例子
            logger.info(f"Sample {sample_id}:")
            logger.info(f"  Predicted: {predicted_grid}")
            logger.info(f"  Actual: {true_grid}")
            logger.info(f"  Correct: {is_correct}")
    
    accuracy = correct_predictions / total_samples
    logger.info(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 训练模型
    model = train_arc_model()
    
    # 评估模型
    results = evaluate_arc_model(model)