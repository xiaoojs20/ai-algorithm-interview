import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Expert(nn.Module):
    """
    Individual Expert module (Standard FFN).
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Feed-Forward network logic
        return self.w2(self.act(self.w1(x)))

class SparseMoE(nn.Module):
    r"""
    Sparse Mixture of Experts (MoE) - Modern Implementation.
    
    Architecture Highlights:
    1. Gating/Router: Learns which experts to activate for each token.
    2. Top-K Routing: Activates only a subset of experts to save computation.
    3. Load Balancing: Ensures experts are utilized uniformly.
    
    Formula:
    $ y = \sum_{i \in \text{TopK}} \text{Router}(x)_i \cdot \text{Expert}_i(x) $
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, expert_ff: Optional[int] = None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Expert intermediate dimension (usually 4x as d_model)
        expert_ff = expert_ff or (4 * d_model)
        
        # 1. Router (Gating Network)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 2. Experts Collection
        self.experts = nn.ModuleList([Expert(d_model, expert_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with expert dispatching.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            l_aux: Auxiliary loss for load balancing
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model) # (N, d_model), N = batch * seq
        
        # 3. Calculate Router Score (Logits)
        router_logits = self.router(x_flat) # (N, num_experts)
        
        # 4. Top-K Gating
        # probabilities P(专家i | x)
        # We use softmax over ALL experts for auxiliary loss, but only Top-K for output
        probs = F.softmax(router_logits, dim=-1) # (N, num_experts)
        
        # 获取 Top-K 权重和索引
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 5. Expert Computation (Optimized Dispatching)
        # Initialize output
        final_output = torch.zeros_like(x_flat)
        
        # top_k_weights 形状: (N, top_k), top_k_indices 形状: (N, top_k)
        # N 是 Token 总量 (batch * seq)
        
        # 遍历所有专家，依次处理被分配到该专家的 Token
        for i in range(self.num_experts):
            # 找到所有 Top-K 索引中包含当前专家索引 i 的位置
            # mask 形状: (N, top_k)
            mask = (top_k_indices == i)
            
            # 使用 torch.where 提取 2D 坐标：
            # row_indices: 样本索引，即“哪些 Token”选中了专家 i (0 到 N-1)
            # slot_indices: Top-K 索引，即专家 i 是该 Token 的第几个选中选手 (0 到 top_k-1)
            row_indices, slot_indices = torch.where(mask)
            
            if row_indices.numel() > 0:
                # 1. Dispatch: 提取属于当前专家 i 的输入数据 (N_assigned, d_model)
                expert_input = x_flat[row_indices]
                
                # 2. Expert Forward: 让专家 i 处理这些 Token
                expert_output = self.experts[i](expert_input)
                
                # 3. Combine: 根据对应的路由权重进行加权累加
                # 使用提取的坐标从 top_k_weights (N, top_k) 中取出权重
                # shape 变为 (N_assigned, 1) 以便与 expert_output 相乘
                weights = top_k_weights[row_indices, slot_indices].unsqueeze(-1)
                final_output[row_indices] += weights * expert_output
        
        # 6. Auxiliary Loss (Load Balancing) - For Interview Points!
        # $ L_{aux} = N_{exp} \cdot \sum f_i \cdot P_i $
        # f_i: fraction of tokens dispatched to expert i
        # P_i: average probability assigned to expert i
        
        # Fraction of tokens dispatched (counts)
        # top_k_indices: (N, top_k)
        count_experts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            count_experts[i] = (top_k_indices == i).any(dim=-1).float().mean()
        
        # Mean scores across batch
        mean_probs = probs.mean(dim=0)
        
        # Balancing Loss (Entropy-like)
        l_aux = self.num_experts * torch.sum(count_experts * mean_probs)
        
        return final_output.view(batch_size, seq_len, self.d_model), l_aux

# Test Code
if __name__ == "__main__":
    d_model = 128
    num_experts = 8
    top_k = 2
    
    moe = SparseMoE(d_model, num_experts, top_k)
    x = torch.randn(2, 32, d_model) # [batch, seq, d_model]
    
    out, aux_loss = moe(x)
    print(f"Output shape: {out.shape}")
    print(f"Auxiliary Loss: {aux_loss.item():.4f}")
