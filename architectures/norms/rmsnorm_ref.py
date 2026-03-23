import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    
    原理介绍：
    RMSNorm 是 LayerNorm 的一个简化变体。与 LayerNorm 相比，它去除了“减去均值”的步骤（即 re-centering），仅保留“根据均方根缩放”的步骤（re-scaling）。
    
    1. 计算效率：由于不需要计算均值，减少了约 10%~40% 的计算开销（取决于后端实现）。
    2. 数学性质：它具有“缩放不变性”（scale-invariant），这在深度学习中已被证明是归一化能够稳定训练的关键。
    3. 实验效果：在 Transformer 模型（如 LLaMA）中，移除均值中心化并不会损害性能，反而训练更稳定。

    LaTeX 公式:
    $ \text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum_{j=1}^n x_j^2 + \epsilon}} \cdot \gamma_i $
    
    Shapes:
    - x: (B, L, D) 其中 B=Batch size, L=Sequence length, D=Hidden dimension
    - weight (gamma): (D,) 可学习的缩放参数
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        # 初始化可学习参数 gamma 为全 1
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 1. 计算均方根 (Root Mean Square)
        # x.pow(2): 求每个元素的平方
        # .mean(-1, keepdim=True): 对最后一个维度求均值，保持维度 (B, L, 1) 以便后续广播
        # + self.eps: 防止除以 0 的数值不稳定性
        ms = x.pow(2).mean(-1, keepdim=True)
        
        # 2. 计算倒数平方根 (Inverse Square Root)
        # torch.rsqrt(ms + eps) 等于 1.0 / sqrt(ms + eps)
        # 相比先 sqrt 再除法，rsqrt 在现代加速器上通常更快
        inv_sqrt = torch.rsqrt(ms + self.eps)
        
        # 3. 执行归一化
        # x: (B, L, D) * inv_sqrt: (B, L, 1) -> (B, L, D) 通过广播机制实现
        norm_x = x * inv_sqrt
        
        # 4. 最后乘以可学习参数 gamma (self.weight)
        # self.weight: (D,) 被广播到 (B, L, D)
        return self.weight * norm_x
