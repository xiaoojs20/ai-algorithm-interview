import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    """
    GQA (Grouped Query Attention)
    
    原理：
    MHA (Multi-Head Attention): 每个 Query 头都有对应的 Key/Value 头。存储开销大。
    MQA (Multi-Query Attention): 所有 Query 头共享同一个 Key/Value 头。存储开销小但性能有损。
    GQA (Grouped Query Attention): 将 Query 分组，每一组共享一个 Key/Value 头。是 MHA 和 MQA 的折中方案。
    
    优势：
    - 显著降低模型推理时的 KV Cache 显存占用。
    - 保持类似 MHA 的模型表现，且推理速度比 MHA 更快。
    
    Shapes:
    - n_q_heads: 总 Query 头数
    - n_kv_heads: 总 KV 头数 (n_q_heads 应能被 n_kv_heads 整除)
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_q_heads
        self.num_groups = n_q_heads // n_kv_heads
        
        # 投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        # 注意：K, V 投影到较小的维度 (n_kv_heads * d_k)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        # 1. 线性投影
        q = self.w_q(x).view(B, L, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # 2. 对 K, V 进行重复 (Repeat Interleave) 以匹配 Q 的头数
        # (B, n_kv_heads, L, d_k) -> (B, n_q_heads, L, d_k)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
    
        # 3. 计算缩放点积注意力
        # Q: (B, H_q, L, d_k), K^T: (B, H_q, d_k, L) -> Scores: (B, H_q, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attn = torch.softmax(scores, dim=-1)
        # (B, H_q, L, L) @ (B, H_q, L, d_k) -> (B, H_q, L, d_k)
        out = torch.matmul(attn, v)
        
        # 4. 拼接多头并线性变换
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(out)

if __name__ == "__main__":
    d_model, n_q, n_kv = 128, 8, 2
    model = GQA(d_model, n_q, n_kv)
    x = torch.randn(1, 10, d_model)
    print(f"GQA output shape: {model(x).shape}")
