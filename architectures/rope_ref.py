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
        
        # 3. 构造旋转矩阵的向量化分量 (L, D)
        # 这里采用 LLaMA/HuggingFace 风格的半半拆分法：
        # 将 D 维向量分为 [x1, x2, ..., x_{d/2}] 和 [x_{d/2+1}, ..., x_d]
        # 然后将它们视为 (x_i, x_{i+d/2}) 这样的配对进行 2D 旋转。
        emb = torch.cat((freqs, freqs), dim=-1) # Shape: (L, D)
        
        cos, sin = emb.cos(), emb.sin() # Shape: (L, D)
        
        # 4. 执行旋转变换 (Rotary Transformation)
        # 数学原理：对于一对坐标 (a, b)，旋转角度 theta 后的坐标为：
        # [a*cos(theta) - b*sin(theta), b*cos(theta) + a*sin(theta)]
        # 
        # 为了高效实现（向量化），我们将输入 x 拆成两半：x1 和 x2
        # x: [x_1, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
        # x1: [x_1, ..., x_{d/2}]
        # x2: [x_{d/2+1}, ..., x_d]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        
        # 构造旋转变换的辅助向量 x_rotated = [-x2, x1]
        # x_rotated: [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        # 计算：x * cos + x_rotated * sin
        # 对应位置相加后：
        # 前半部分 (1 to d/2): x_i * cos(theta_i) - x_{i+d/2} * sin(theta_i)
        # 后半部分 (d/2 to d): x_{i+d/2} * cos(theta_i) + x_i * sin(theta_i)
        # 这完美匹配了复数旋转/2D 旋转矩阵的逐元素展开逻辑。
        return (x * cos) + (x_rotated * sin)
