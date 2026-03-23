import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    
    LaTeX Formula:
    $ \mathbf{R}_{\Theta, m}^d = \mathbf{x} \cos(m\theta) + \mathbf{x}_{rotate} \sin(m\theta) $
    $ \text{where } \theta_i = 10000^{-2i/d} $
    """
    def __init__(self, d_model, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [seq, d/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [seq, d]
        
        cos, sin = emb.cos(), emb.sin()
        
        # x: [batch, head, seq, d]
        # 应用旋转
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (x_rotated * sin)
