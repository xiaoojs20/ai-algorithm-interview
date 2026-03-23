import torch
import torch.nn as nn

class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA)
    
    LaTeX Formula:
    $ \mathbf{W}_{updated} = \mathbf{W} + \Delta \mathbf{W} = \mathbf{W} + \frac{\alpha}{r} \mathbf{BA} $
    $ \text{where } \mathbf{A} \in \mathbb{R}^{d_{in} \times r}, \mathbf{B} \in \mathbb{R}^{r \times d_{out}} $
    """
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # A: (d_in, r) initialized with Gaussian
        self.lora_A = nn.Parameter(torch.empty(d_in, r))
        # B: (r, d_out) initialized with zeros
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x, W_base):
        # W_base: 原矩阵 [d_in, d_out]
        base_out = x @ W_base
        lora_delta = (x @ self.lora_A) @ self.lora_B
        
        return base_out + lora_delta * self.scaling
