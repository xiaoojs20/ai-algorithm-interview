import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm
    
    LaTeX Formula:
    $ \bar{a}_i = \frac{a_i}{\sqrt{\frac{1}{n}\sum a_i^2 + \epsilon}} \cdot g_i $
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, L, D)
        # 1. Calculate mean of squares along the last dimension (D)
        # x.pow(2): (B, L, D) -> (B, L, D)
        # .mean(-1, keepdim=True): (B, L, D) -> (B, L, 1)
        # 2. Add epsilon for numerical stability
        # 3. Take reciprocal square root (1 / sqrt(...))
        # torch.rsqrt is equivalent to 1.0 / torch.sqrt(...)
        # 4. Normalize x by the inverse square root
        # x: (B, L, D) * (B, L, 1) -> (B, L, D)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 5. Scale by learnable weight parameter
        # self.weight: (D)
        # (B, L, D) * (D) -> (B, L, D)
        return self.weight * norm
