import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization (LN)
    原理：对每个样本的所有特征维度 (B, L, D) 的 D 维取均值和方差。
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # TODO: Initialize gamma and beta
        pass
        
    def forward(self, x):
        # TODO: Implement normalization over dim=-1
        pass

if __name__ == "__main__":
    B, L, D = 2, 10, 128
    ln = LayerNorm(D)
    x = torch.randn(B, L, D)
    print(f"LayerNorm Output Shape: {ln(x).shape}")
