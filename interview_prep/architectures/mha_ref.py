import torch
import torch.nn as nn
from .sdpa_ref import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA)
    
    LaTeX Formula:
    $ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $
    
    Shapes:
    - B: Batch size
    - L: Sequence length
    - D: Embedding dimension (d_model)
    - H: Number of heads
    - d_k: Dimension per head (D // H)
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # Input: (B, L, D)
        batch_size = q.size(0)
        
        # 1. Linear Transformation: (B, L, D) -> (B, L, D)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. Split heads: (B, L, D) -> (B, L, H, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k)
        k = k.view(batch_size, -1, self.n_heads, self.d_k)
        v = v.view(batch_size, -1, self.n_heads, self.d_k)
        
        # 3. Transpose sequence and heads: (B, L, H, d_k) -> (B, H, L, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # 4. Scaled Dot-Product Attention: (B, H, L_q, d_k), (B, H, L_k, d_k), (B, H, L_k, d_k) -> (B, H, L_q, d_k)
        x = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 5. Transpose sequence and heads back: (B, H, L, d_k) -> (B, L, H, d_k)
        # Why contiguous()? transpose() only changes metadata (strides), 
        # but view() in the next step requires memory to be contiguous.
        x = x.transpose(1, 2).contiguous() 
        
        # 6. Concatenate heads: (B, L, H, d_k) -> (B, L, D)
        x = x.view(batch_size, -1, self.d_model)
        
        # 7. Final Output Projection: (B, L, D) -> (B, L, D)
        return self.w_o(x)
