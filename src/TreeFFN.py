import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add  # pip install torch-scatter
from typing import Optional, Dict, Tuple

class TreeFFN(nn.Module):
    """
    Global Parent‑Child Aggregation MLP.
    可选门控聚合、残差连接、双向传播。
    优化版本：消息传播计算优化 + 中间结果缓存
    """

    def __init__(self,
                 d_in: int,          # 输入特征维度
                 d_h: int = 128,     # 隐藏层维度
                 num_node_classes: int = None,
                 num_tree_classes: int = None,
                 use_edge_proj: bool = False,
                 use_gating: bool = False,
                 residual: bool = True,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 tree_iterations: int = 2):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(float(tree_iterations)), requires_grad=True)
        self.use_edge_proj = use_edge_proj
        self.use_gating = use_gating
        self.residual = residual
        self.bidirectional = bidirectional

        # 基础投影：节点自身 → 隐藏空间
        self.W_s = nn.Linear(d_in, d_h, bias=False)

        # 统一父子聚合权重
        self.W_pc = nn.Linear(d_h, d_h, bias=False)

        if use_edge_proj:
            # 边信息投影（可选）
            self.W_edge = nn.Linear(2 * d_h, d_h, bias=False)

        if use_gating:
            # 门控权重
            self.W_gate = nn.Linear(2 * d_h, 1)   # scalar gate

        self.dropout = nn.Dropout(dropout)

        # 输出层（节点级）
        self.num_node_classes = num_node_classes
        if num_node_classes is not None:
            self.node_out = nn.Linear(d_h, num_node_classes)

        # 输出层（树级，根节点或全局池化）
        self.num_tree_classes = num_tree_classes
        if num_tree_classes is not None:
            self.tree_out = nn.Linear(d_h, num_tree_classes)

        # 缓存变量
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_valid = False
        self._last_edge_index = None

    def _invalidate_cache(self):
        """清除缓存"""
        self._cache.clear()
        self._cache_valid = False

    def _compute_edge_features(self, h: torch.Tensor, edge_index: torch.LongTensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        优化的边特征计算 - 移除无效缓存，专注计算优化
        """
        # 直接计算，不使用复杂缓存
        p_idx = edge_index[0]   # 父节点索引
        c_idx = edge_index[1]   # 子节点索引
        
        # 批量获取父子节点特征
        h_p = h[p_idx]  # [E, d_h]
        h_c = h[c_idx]  # [E, d_h]
        
        # 计算消息 - 优化版本
        if self.use_edge_proj:
            # 预计算concatenated features避免重复计算
            edge_concat = torch.cat([h_p, h_c], dim=1)  # [E, 2*d_h]
            msg = self.W_edge(edge_concat)  # [E, d_h]
            gate_input = edge_concat if self.use_gating else None
        else:
            # 简单加法更快
            msg = h_p + h_c  # [E, d_h]
            gate_input = torch.cat([h_p, h_c], dim=1) if self.use_gating else None
        
        return msg, gate_input, p_idx, c_idx

    def _apply_gating(self, msg: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """应用门控机制"""
        if self.use_gating and gate_input is not None:
            alpha = torch.sigmoid(self.W_gate(gate_input))  # [E, 1]
            return msg * alpha
        return msg

    def _aggregate_messages(self, msg: torch.Tensor, p_idx: torch.Tensor, c_idx: torch.Tensor, N: int) -> torch.Tensor:
        """
        极简的消息聚合 - 移除复杂优化，专注速度
        """
        # 直接使用scatter_add，最简单的方式
        agg = torch.zeros(N, msg.size(1), dtype=msg.dtype, device=msg.device)
        
        # 分别处理父节点和子节点的消息聚合
        agg.scatter_add_(0, p_idx.unsqueeze(1).expand(-1, msg.size(1)), msg)
        agg.scatter_add_(0, c_idx.unsqueeze(1).expand(-1, msg.size(1)), msg)
        
        return agg

    def _streaming_forward_step(self, h: torch.Tensor, edge_index: torch.LongTensor, N: int) -> torch.Tensor:
        """
        单次迭代的流式前向传播步骤
        """
        # 1. 计算边特征（带缓存）
        msg, gate_input, p_idx, c_idx = self._compute_edge_features(h, edge_index)
        
        # 2. 应用门控
        msg = self._apply_gating(msg, gate_input)
        
        # 3. 聚合消息
        agg = self._aggregate_messages(msg, p_idx, c_idx, N)
        
        # 4. 更新隐藏状态
        if self.bidirectional:
            h_up = F.relu(self.W_pc(agg) + h)
            h_down = F.relu(self.W_pc(agg) + h)
            new_h = torch.cat([h_up, h_down], dim=1)
        else:
            new_h = F.relu(self.W_pc(agg) + h)
            if self.residual:
                new_h = new_h + h
        
        return new_h

    # ------------------------------------------------------------------
    def forward(self,
                node_feats: torch.Tensor,     # [N, d_in]
                edge_index: torch.LongTensor,  # [2, E]
                root_idx: int = 0):
        """
        简化的前向传播：保持T可学习，简化其他计算
        """
        N = node_feats.size(0)
        
        # 初始隐藏状态
        h = self.W_s(node_feats)          # [N, d_h]
        
        # 使用可学习的T参数，但简化计算
        max_iterations = max(1, min(int(self.T.item()) + 1, 3))  # 限制最大3次迭代
        
        # 简化迭代处理 - 保持T的梯度
        for step_idx in range(max_iterations):
            # 计算当前步骤的权重（保持可学习）
            step_weight = torch.sigmoid(self.T - step_idx)
            
            # 执行单步前向传播
            step_h = self._streaming_forward_step(h, edge_index, N)
            
            # 加权更新（保持T的学习能力）
            h = h + step_weight * (step_h - h)
            
            # 早期停止：如果权重太小，后续迭代贡献很小
            if step_weight < 0.05:
                break
        
        # 最终隐藏状态
        final_h = self.dropout(h)

        outputs = {}
        
        # 始终返回隐藏状态 (for TreeGPT use)
        outputs['hidden'] = final_h

        if self.num_node_classes is not None:
            node_logits = self.node_out(self.dropout(final_h))   # [N, C]
            outputs['node_logits'] = node_logits

        if self.num_tree_classes is not None:
            root_hidden = final_h[root_idx]                       # 只取根节点
            tree_logit  = self.tree_out(self.dropout(root_hidden))
            outputs['tree_logit'] = tree_logit.unsqueeze(0)       # [1, C]

        return outputs
