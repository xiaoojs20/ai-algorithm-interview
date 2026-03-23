import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) - Gated Layer
    
    LaTeX Formula (Gating):
    $ y = \sum_{i=1}^{n} G(x)_i E_i(x) $
    $ G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g, k)) $

    主要考查点: Router 负载均衡 (Balance Loss) 和 Top-K 路由逻辑。
    """
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        ) for _ in range(num_experts)])

    def forward(self, x):
        # x: [batch, seq, d_model]
        batch, seq, d_model = x.shape
        x_flat = x.view(-1, d_model) # [batch*seq, d_model]
        
        # 1. 计算路由得分 (Router logits)
        router_logits = self.router(x_flat) # [B*S, num_experts]
        
        # 2. 获取 Top-K 专家及权重
        weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1) # [B*S, top_k]
        
        output = torch.zeros_like(x_flat)
        
        # 3. 专家计算 (面试简单版: 手动循环；实际大规模用 einsum 或 dispatch)
        for i in range(self.num_experts):
            # 获取当前专家被选中的索引
            # batch_idx[mask] 就是哪些样本选了这个专家索引 i
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                # 找到被当前专家负责的样本子集
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                
                # 回填结果并乘以对应的权重
                # 权重在 weights 里的位置需对齐
                for k in range(self.top_k):
                    sample_mask = (selected_experts[:, k] == i)
                    if sample_mask.any():
                        output[sample_mask] += weights[sample_mask, k].unsqueeze(-1) * self.experts[i](x_flat[sample_mask])
                        
        return output.view(batch, seq, d_model)
