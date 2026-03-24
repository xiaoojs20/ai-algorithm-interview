import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    """
    Batch Normalization (BN)
    原理：对每个特征维度在 Batch 维上取统计值。
    注意：在评估模式 (eval) 下，使用累积的全局均值和方差。
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # TODO: Initialize gamma, beta, and buffers for running_mean/var
        pass
        
    def forward(self, x):
        # TODO: Implement train-time calculation and inference-time reuse
        pass

if __name__ == "__main__":
    B, C = 2, 16
    bn = BatchNorm1d(C)
    x = torch.randn(B, C)
    print(f"BN Train output shape: {bn(x).shape}")
