import torch
import torch.nn.functional as F

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
# 并非真的torch底层代码
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

    # 1. scale
    # 2. bias & mask (causal + attn_mask)
    # 3. weight = softmax(query @ key^T * scale + bias)
    # 4. dropout(weight)
    # 5. output = weight @ value
    
    # B, H, L | S, D
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value