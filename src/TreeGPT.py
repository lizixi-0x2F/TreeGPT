import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .TreeFFN import TreeFFN


class TreeFFNSeq2SeqBlock(nn.Module):
    """
    TreeFFN Encoder-Decoder Block - 纯并行处理，无attention和自回归
    """
    
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 tree_iterations: int = 3):
        super().__init__()
        
        # 编码器TreeFFN - 从左到右处理输入序列
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
        
        # 解码器TreeFFN - 从右到左生成输出序列
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
        并行encoder-decoder前向传播
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
            dec_out = self.decoder_tree_ffn(h_b, decoder_edges, root_idx=seq_len-1)
            
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


class TreeGPT(nn.Module):
    """
    TreeGPT模型 - 基于TreeFFN Encoder-Decoder架构
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 n_layers: int = 2,
                 max_seq_len: int = 8192,
                 dropout: float = 0.1,
                 tree_iterations: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # TreeFFN Encoder-Decoder blocks
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
        纯并行前向传播 - 一次处理整个序列
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
        
        # 通过TreeFFN Encoder-Decoder blocks
        for block in self.blocks:
            h = block(h)
        
        # 最终层归一化和输出投影
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits