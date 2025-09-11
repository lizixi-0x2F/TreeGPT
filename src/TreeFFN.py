import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeFFN(nn.Module):
    """
    Global Parent‑Child Aggregation MLP.
    可选门控聚合、残差连接、双向传播。
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

    # ------------------------------------------------------------------
    def forward(self,
                node_feats: torch.Tensor,     # [N, d_in]
                edge_index: torch.LongTensor,  # [2, E]
                root_idx: int = 0):
        """
        :param node_feats: 节点原始特征
        :param edge_index: 父子关系，第一行父节点索引，第二行为子节点索引
        :param root_idx: 根节点在 node_feats 中的索引
        :return:
            - node_logits [N, num_node_classes]  (若设置了)
            - tree_logit   [1, num_tree_classes] (若设置了)
        """
        N = node_feats.size(0)

        # 初始隐藏状态
        h = self.W_s(node_feats)          # [N, d_h]

        # 使用可微分的循环次数 - 软迭代
        # 为了保持梯度流，我们使用固定的最大迭代次数，但用T来加权每次迭代的贡献
        max_iterations = 10  # 固定最大迭代次数
        
        # 初始化累积的隐藏状态
        accumulated_h = torch.zeros_like(h)
        
        for step_idx in range(max_iterations):
            # 计算当前步骤的权重 (sigmoid-based soft gating)
            step_weight = torch.sigmoid(self.T - step_idx)  # 当T>step_idx时权重较大
            
            # ------------------------------------------------------------------
            # 1️⃣ 计算所有边的消息
            p_idx = edge_index[0]   # 父节点索引
            c_idx = edge_index[1]   # 子节点索引

            h_p = h[p_idx]
            h_c = h[c_idx]

            if self.use_edge_proj:
                msg = self.W_edge(torch.cat([h_p, h_c], dim=1))  # [E, d_h]
            else:   # 简单加和
                msg = h_p + h_c

            if self.use_gating:
                gate_input = torch.cat([h_p, h_c], dim=1)      # [E, 2*d_h]
                alpha = torch.sigmoid(self.W_gate(gate_input))  # [E, 1]
                msg = msg * alpha

            # ------------------------------------------------------------------
            # 2️⃣ 把消息聚合到每个节点（父子两端都收到）
             # 创建输出张量
            agg = torch.zeros(N, msg.size(-1), dtype=msg.dtype, device=msg.device)

              # 第一次聚合
            agg.scatter_add_(0, p_idx.unsqueeze(-1).expand(-1, msg.size(-1)), msg)

              # 第二次聚合（累加）
            agg.scatter_add_(0, c_idx.unsqueeze(-1).expand(-1, msg.size(-1)), msg)

            # ------------------------------------------------------------------
            # 3️⃣ 计算这一步的隐藏状态更新
            if self.bidirectional:
                step_h_up   = F.relu(self.W_pc(agg) + h)
                step_h_down = F.relu(self.W_pc(agg) + h)
                step_h = torch.cat([step_h_up, step_h_down], dim=1)
            else:
                step_h = F.relu(self.W_pc(agg) + h)
                if self.residual:
                    step_h = step_h + h
                    
            # 用权重累积这一步的贡献
            accumulated_h = accumulated_h + step_weight * step_h
            
            # 更新h为这一步的结果，为下一步做准备
            h = step_h

        # ------------------------------------------------------------------
        # 4️⃣ 最终隐藏状态 - 使用累积的结果并应用dropout
        final_h = self.dropout(accumulated_h)

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