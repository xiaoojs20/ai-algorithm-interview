import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network (FFN)
    结构：x -> Linear -> Activation -> Linear -> Out
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Initialize layers
        pass
        
    def forward(self, x):
        # TODO: Implement forward pass
        pass

if __name__ == "__main__":
    B, L, D = 1, 10, 128
    D_ff = 4 * D
    model = StandardFFN(D, D_ff)
    x = torch.randn(B, L, D)
    print(f"Output shape: {model(x).shape}")
