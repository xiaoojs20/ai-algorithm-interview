import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """
    FFN with SwiGLU Activation (LLaMA 等主流模型使用)
    
    原理：通过 SiLU (Swish) 激活后的门控信号作用于内容分量。
    公式：(SiLU(xW) * xV) * W_down
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Initialize gate, content, and down projections
        pass
        
    def forward(self, x):
        # TODO: Implement chunking, SiLU gating, and down projection
        pass

if __name__ == "__main__":
    B, L, D = 2, 10, 128
    D_ff = (D * 8 // 3)
    model = SwiGLUFFN(D, D_ff)
    x = torch.randn(B, L, D)
    print(f"Output shape: {model(x).shape}")
