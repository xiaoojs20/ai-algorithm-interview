import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    RoPE (Rotary Positional Embedding) - 旋转位置编码
    
    原理介绍：
    RoPE 是目前 LLM（如 LLaMA, GLM）的主流位置编码方式。它通过将 Query 和 Key 向量分对（每两个维度一组）并在 2D 平面上进行旋转来注入位置信息。
    
    核心优势：
    1. 相对位置依赖：两个向量经过 RoPE 编码后的点积只取决于它们的相对距离 $(m-n)$，这符合直觉（邻近的词相关性更高）。
    2. 远程衰减：随着相对距离增加，其点积呈现衰减趋势（通过选取 base=10000 频率）。
    3. 线性性：位置编码可以直接作用于 Query 和 Key 向量，不需要修改 Attention 矩阵。

    LaTeX 公式:
    对于每一对维度 $[x_1, x_2]$，旋转角度为 $m\theta_i$：
    $ \text{RoPE}(x_1, x_2, m) = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $
    
    其中 $\theta_i = 10000^{-2i/d}, i \in [0, d/2)$
    """
    def __init__(self, d_model, base=10000):
        super().__init__()
        # 1. 预计算频率序列 theta_i
        # 维度是 d_model，但旋转是两两一组进行的，所以频率只有 d_model / 2 个
        # inv_freq: Shape [d_model / 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        # x: (Batch, Heads, L, D)
        t = torch.arange(seq_len).type_as(self.inv_freq)
        
        # 2. 计算位置 m 与频率 theta 的乘积 (m * theta_i)
        # 用 einsum 实现外积：t(L) 和 inv_freq(D/2) -> freqs(L, D/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) 
        
        # 3. 构造完整的旋转角度（通过将两个 freqs 拼接）
        # emb: (L, D)
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        # 计算 cos 和 sin
        cos, sin = emb.cos(), emb.sin() # Shape: (L, D)
        
        # 4. 执行旋转操作 (基于複数运算的快速实现)
        # 原始公式：x_new = [x1*cos - x2*sin, x2*cos + x1*sin]
        # 下面代码拆分前一半和后一半维度
        # x_rotated 是 [-x_half2, x_half1] 这样可以一行代码完成旋转计算
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        # 公式简化为：x * cos + x_rotated * sin
        return (x * cos) + (x_rotated * sin)
