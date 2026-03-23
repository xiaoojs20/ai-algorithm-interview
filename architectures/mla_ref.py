import torch
import torch.nn as nn
import math

class MLA(nn.Module):
    """
    MLA (Multi-head Latent Attention) - 这里的实现基于 DeepSeek-V2 原理
    
    原理：
    旨在极度压缩推理时的 KV Cache。相比于 MQA 或 GQA 只是简单的头数共享，
    MLA 通过对 KV 进行低秩潜变量投影 (Low-Rank Latent Projection) 来压缩信息。
    
    核心思想：
    1. KV 压缩：Q, K, V 都通过一个低秩矩阵投影到低维潜空间 $d_c$。
    2. 解压缩：在计算时再将潜变量 $c^{KV}$ 升维回多头 $d_h$。
    3. 位置编码：为了节省缓存，RoPE 被应用到 Query 和 Key 的独立子集维度上。
    
    优势：
    - 推理时的 KV Cache 特别小（仅存储低秩潜变量 $c^{KV}$）。
    - 维持极高的多头表达能力。
    
    Shapes:
    - d_model: 模型维度
    - d_c: 隐变量 (Latent) 维度 (d_c << d_model)
    - d_h: 每个注意力头的维度
    - h: 头数
    """
    def __init__(self, d_model, d_c, d_h, h, rope_dim=32):
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c
        self.d_h = d_h
        self.h = h
        self.rope_dim = rope_dim
        
        # 1. KV 压缩投影
        self.kv_a_proj = nn.Linear(d_model, d_c) # 压缩到 latent d_c
        self.kv_b_proj = nn.Linear(d_c, h * d_h) # 升维回到 multi-head
        
        # 2. Q 投影 (通常也进行压缩投影以保持精度)
        self.q_a_proj = nn.Linear(d_model, d_c)
        self.q_b_proj = nn.Linear(d_c, h * d_h)
        
        # 3. 对 Q, K 额外增加 RoPE 所需的维度投影 (解耦)
        self.q_rope_proj = nn.Linear(d_model, h * rope_dim)
        self.k_rope_proj = nn.Linear(d_model, rope_dim)
        
        # 4. 输出投影
        self.o_proj = nn.Linear(h * d_h, d_model)
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # --- KV 计算 ---
        # 1. [关键过程] 压缩到 Latent: (B, L, d_c)
        # 此阶段在推理时只需缓存 c_kv
        c_kv = self.kv_a_proj(x)
        
        # 2. 从 Latent 升维回多头并计算 Content K, V
        # k_content, v_content shape: (B, L, h, d_h)
        kv_up = self.kv_b_proj(c_kv).view(B, L, self.h, self.d_h)
        k_content = kv_up.transpose(1, 2)
        v_content = kv_up.transpose(1, 2)
        
        # --- Q 计算 ---
        # 同样进行多步投影: (B, L, h, d_h)
        q_content = self.q_b_proj(self.q_a_proj(x)).view(B, L, self.h, self.d_h).transpose(1, 2)
        
        # --- RoPE 位置增强 ---
        # 计算独立的 RoPE 分量 (解耦位置编码)
        q_rope = self.q_rope_proj(x).view(B, L, self.h, self.rope_dim).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, L, 1, self.rope_dim).transpose(1, 2) # 共享一份 RoPE K
        
        # --- 拼接 Content 和 RoPE ---
        # q: (B, h, L, d_h + rope_dim)
        # k: (B, h, L, d_h + rope_dim)
        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope.repeat(1, self.h, 1, 1)], dim=-1)
        
        # --- 计算 Attention ---
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
             scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        # out: (B, h, L, d_h) - 注意力只作用于 v_content
        out = torch.matmul(attn, v_content)
        
        # 出路合并
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

if __name__ == "__main__":
    B, L, D = 1, 10, 128
    d_c, d_h, h = 64, 32, 4
    model = MLA(D, d_c, d_h, h)
    x = torch.randn(B, L, D)
    print(f"MLA output shape: {model(x).shape}")
