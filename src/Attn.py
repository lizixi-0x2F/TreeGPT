import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力模块 (Scaled Dot‑Product Attention)

    Parameters
    ----------
    d_model : int
        输入/输出特征维度（即每个节点的隐藏向量维度）。
    n_heads : int
        注意力头数。要求 `d_model % n_heads == 0`。
    dropout : float, optional (default=0.1)
        Dropout 概率，应用在注意力权重上。
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads     # 每个 head 的维度

        # Q/K/V 三个线性投影（共享输出维度）
        self.q_proj    = nn.Linear(d_model, d_model)
        self.k_proj    = nn.Linear(d_model, d_model)
        self.v_proj    = nn.Linear(d_model, d_model)

        # 输出投影
        self.out_proj  = nn.Linear(d_model, d_model)

        self.dropout   = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : Tensor of shape (B, L, d_model)
            输入序列（树节点特征）。这里假设所有节点都在同一批中。
        mask : Tensor or None, optional
            形状为 (B, L) 或 (B, L, L)。  
            - 若是 `(B, L)`，表示对每个位置的 *有效*/`1` 与 *无效* / `0` 做掩码。  
              在注意力计算中将对应列设为 `-inf`（不参与 softmax）。  
            - 若是 `(B, L, L)`，可做更细粒度的遮蔽（如树结构只允许父子连接）。

        Returns
        -------
        out : Tensor of shape (B, L, d_model)
            经过多头注意力后的表示。
        """
        B, L, _ = x.size()

        # ---- 1. Q/K/V 投影并 reshape 为 (B, n_heads, L, d_k) ----
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_k)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_k)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_k)

        # 维度交换为 (B, n_heads, L, d_k) 方便后面矩阵乘
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ---- 2. 计算缩放点积注意力 ----
        #   scores : (B, n_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ---- 3. 应用 mask（若有）----
        if mask is not None:
            # 对于二元掩码，先广播到 (B, n_heads, L, L)
            if mask.dim() == 2:          # (B, L)
                mask = mask.unsqueeze(1).unsqueeze(2)   # -> (B, 1, 1, L)
            elif mask.dim() == 3:        # (B, L, L)
                mask = mask.unsqueeze(1)                 # -> (B, 1, L, L)
            else:
                raise ValueError(f"mask must be 2D or 3D tensor, got {mask.dim()}D")

            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)          # (B, n_heads, L, L)
        attn_weights = self.dropout(attn_weights)

        # ---- 4. 加权求和得到 context 向量 ----
        #   context : (B, n_heads, L, d_k)
        context = torch.matmul(attn_weights, v)

        # ---- 5. 合并头，投影回原维度 ----
        context = context.permute(0, 2, 1, 3).contiguous()    # (B, L, n_heads, d_k)
        context = context.view(B, L, self.d_model)            # (B, L, d_model)

        out = self.out_proj(context)                         # (B, L, d_model)

        return out
