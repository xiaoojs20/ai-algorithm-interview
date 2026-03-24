import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization (LN) - Transformer 核心组件
    
    原理：对每个样本的所有特征维度进行归一化。
    公式：y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Shapes:
    - 输入 x: (B, L, D)
    - 计算均值 mean 和方差 var 的维度: dim=-1 (特征维)
    - gamma, beta: (D,)
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 初始化可学习参数 gamma (缩放) 和 beta (平移)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # x: (B, L, D)
        
        # 1. 在最后一维计算均值
        # keepdim=True 方便后续广播计算 (B, L, 1)
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. 在最后一维计算方差
        # var = E[(x-mean)^2]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 3. 归一化并执行线性变换
        # (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. y = x_norm * gamma + beta
        return x_norm * self.gamma + self.beta

if __name__ == "__main__":
    B, L, D = 2, 10, 128
    x = torch.randn(B, L, D)
    
    ln = LayerNorm(D)
    out = ln(x)
    
    # 验证均值接近 0，方差接近 1
    print(f"Mean (should be ~0): {out.mean(dim=-1).mean():.4f}")
    print(f"Var (should be ~1): {out.var(dim=-1).mean():.4f}")
