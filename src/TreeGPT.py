import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from .Attn import MultiHeadSelfAttention
    from .TreeFFN import TreeFFN
except ImportError:
    from Attn import MultiHeadSelfAttention
    from TreeFFN import TreeFFN


class TreeGPTBlock(nn.Module):
    """
    GPT Block with Attention + TreeFFN (simplified for AST mode)
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 tree_iterations: int = 2):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # 使用原始TreeFFN，但不需要分类输出
        self.tree_ffn = TreeFFN(
            d_in=d_model,
            d_h=d_model,  # 保持相同维度
            num_node_classes=None,  # 不需要分类
            num_tree_classes=None,  # 不需要分类
            use_edge_proj=True,
            use_gating=True,
            residual=True,
            dropout=dropout,
            tree_iterations=tree_iterations
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        标准序列处理: [attention -> TreeFFN (with sequential edges)]
        x: [batch_size, seq_len, d_model]
        mask: attention mask
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # 1. Self-Attention + 残差连接
        h = x + self.attention(self.ln1(x), mask)
        
        # 2. TreeFFN + 残差连接 (为序列创建边)
        tree_outputs = []
        edges = self._create_sequential_edges(seq_len, device)
        
        for b in range(batch_size):
            h_b = self.ln2(h[b])  # [seq_len, d_model]
            
            # TreeFFN forward
            tree_out = self.tree_ffn(h_b, edges, root_idx=0)
            
            # 获取hidden状态
            if 'hidden' in tree_out:
                tree_outputs.append(tree_out['hidden'])
            else:
                tree_outputs.append(h_b)  # 回退到输入
        
        tree_h = torch.stack(tree_outputs, dim=0)  # [batch_size, seq_len, d_model]
        h = h + tree_h
        
        return h
    
    def _create_sequential_edges(self, seq_len: int, device: torch.device) -> torch.LongTensor:
        """为序列创建边，用于TreeFFN"""
        edges = []
        
        # 相邻连接 (i -> i+1)
        for i in range(seq_len - 1):
            edges.append([i, i + 1])
        
        
        if not edges:
            # 如果序列太短，至少创建一个自环
            edges.append([0, 0])
        
        return torch.tensor(edges, dtype=torch.long, device=device).t()


class TreeGPT(nn.Module):
    """
    GPT model with TreeFFN blocks
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1,
                 tree_iterations: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # TreeGPT blocks
        self.blocks = nn.ModuleList([
            TreeGPTBlock(
                d_model=d_model,
                n_heads=n_heads,
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
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 位置编码
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        
        # Token + Position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        h = token_emb + pos_emb
        
        # 创建因果掩码 (如果没有提供mask)
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 通过TreeGPT blocks
        for block in self.blocks:
            h = block(h, attention_mask)
        
        # 最终层归一化和输出投影
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits
    
    def generate(self, 
                 input_ids: torch.Tensor, 
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        简单的生成函数
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            with torch.no_grad():
                logits = self(input_ids_cond)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids