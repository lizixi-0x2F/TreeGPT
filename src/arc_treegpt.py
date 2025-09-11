import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging
from .TreeGPT import TreeGPT

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
    
    def __init__(self, data_path: str, max_length: int = 2048):
        self.tokenizer = ARCGridTokenizer()
        self.max_length = max_length
        self.samples = []
        
        # 加载数据
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.challenges = data
        self.sample_ids = list(data.keys())
        
        logger.info(f"Loaded {len(self.sample_ids)} ARC samples")
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample = self.challenges[sample_id]
        
        # 编码输入序列
        input_sequence = self.tokenizer.encode_arc_sample(sample)
        
        # 检查是否有答案（训练数据有答案，测试数据没有）
        if 'output' in sample['test'][0]:
            # 训练模式：创建目标序列
            target_grid = sample['test'][0]['output']
            target_sequence = self.tokenizer.grid_to_sequence(target_grid)
            full_sequence = input_sequence + target_sequence
        else:
            # 测试模式：只有输入
            full_sequence = input_sequence
        
        # 动态长度：只截断，不填充
        if len(full_sequence) > self.max_length:
            full_sequence = full_sequence[:self.max_length]
        
        return {
            'input_ids': torch.tensor(full_sequence, dtype=torch.long),
            'sample_id': sample_id,
            'input_length': len(input_sequence),
            'seq_length': len(full_sequence)  # 实际序列长度
        }


class ARCTreeGPT(TreeGPT):
    """为ARC任务特化的TreeGPT - 基于新的TreeFFN Encoder-Decoder架构"""
    
    def __init__(self, vocab_size: int = 17, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)
        
    def generate_arc_solution(self, 
                             input_sequence: torch.Tensor,
                             tokenizer: ARCGridTokenizer,
                             max_grid_tokens: int = 2048) -> List[List[int]]:
        """
        为ARC任务生成解决方案 - 使用纯并行推理
        """
        self.eval()
        device = input_sequence.device
        
        with torch.no_grad():
            # 直接并行推理 - 不需要自回归生成
            logits = self(input_sequence)
            
            # 获取预测的token序列
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            # 解码生成的序列 - 使用最后部分作为输出网格
            generated_seq = predicted_tokens[0].cpu().tolist()
            
            # 尝试解码输出网格
            try:
                # 简化的解码逻辑 - 直接使用后半部分
                if len(generated_seq) > len(input_sequence[0]):
                    output_tokens = generated_seq[len(input_sequence[0]):]
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
    
    return {
        'input_ids': torch.stack(input_ids),
        'sample_ids': sample_ids,
        'input_lengths': input_lengths,
        'seq_lengths': seq_lengths  # 实际长度，用于mask
    }