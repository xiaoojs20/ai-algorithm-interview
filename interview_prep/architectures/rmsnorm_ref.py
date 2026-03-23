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
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm
