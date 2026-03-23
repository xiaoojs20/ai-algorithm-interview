import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled Dot-Product Attention (SDPA)
    
    LaTeX Formula:
    $ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $
    
    Shapes:
    - query: (B, H, L_q, d_k)
    - key: (B, H, L_k, d_k)
    - value: (B, H, L_k, d_k)
    - mask: (B, 1, L_q, L_k) or (B, H, L_q, L_k)
    - output: (B, H, L_q, d_k)
    """
    d_k = query.size(-1)
    
    # query @ key^T -> (B, H, L_q, d_k) @ (B, H, d_k, L_k) -> (B, H, L_q, L_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        # mask shape: (B, 1, L_q, L_k) or (B, H, L_q, L_k)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # softmax(dim=-1) -> (B, H, L_q, L_k)
    p_attn = F.softmax(scores, dim=-1)
    
    # p_attn @ value -> (B, H, L_q, L_k) @ (B, H, L_k, d_k) -> (B, H, L_q, d_k)
    return torch.matmul(p_attn, value)


# Efficient implementation equivalent to the following:
# 并非真的torch底层代码，是带功能的等效逻辑
def scaled_dot_product_attention_full(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """
    Advanced SDPA with support for:
    1. Causal Masking
    2. Grouped Query Attention (GQA)
    3. Custom Scaling
    4. Dropout
    
    Shapes:
    - B: Batch size
    - H_q: Number of query heads
    - H_kv: Number of KV heads (H_q == H_kv for MHA, H_q > H_kv for GQA)
    - L: Target seq length
    - S: Source seq length
    - D: Head dimension
    """
    # query: (B, H_q, L, D), key: (B, H_kv, S, D), value: (B, H_kv, S, D)
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # Prepare attention bias (masking)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        # Create lower triangular mask for autoregressive generation
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        # Add external mask if provided
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    # 1. GQA Handling: repeat_interleave KV heads to match query heads
    # (B, H_kv, S, D) -> (B, H_q, S, D)
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    # 2. Score Calculation: (B, H_q, L, D) @ (B, H_q, D, S) -> (B, H_q, L, S)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    # 3. Apply Bias/Mask
    # (B, H_q, L, S) + (L, S) broadcast -> (B, H_q, L, S)
    attn_weight += attn_bias
    
    # 4. Softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    # 5. Dropout
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    # 6. Final Aggregation: (B, H_q, L, S) @ (B, H_q, S, D) -> (B, H_q, L, D)
    return attn_weight @ value