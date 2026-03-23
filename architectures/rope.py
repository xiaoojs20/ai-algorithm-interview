import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    RoPE (Rotary Positional Embedding) - 旋转位置编码
    
    原理：通过对向量进行两两一组的平面旋转注入位置信息，仅保留相对位置依赖。
    """
    def __init__(self, d_model, base=10000):
        super().__init__()
        # TODO: Initialize precomputed frequencies
        pass

    def forward(self, x, seq_len):
        # x: (Batch, Heads, L, D)
        # TODO: 
        # 1. 计算维度频率 theta_i
        # 2. 构造位置旋转分量 freqs (L, D/2) -> emb (L, D)
        # 3. 采用“半半拆分”法执行旋转：
        #    a. 将 x 拆分为 x1, x2 (各一半维度)
        #    b. 构造旋转后的辅助项 x_rotated = [-x2, x1]
        #    c. 根据公式 x * cos + x_rotated * sin 计算并返回
        pass
