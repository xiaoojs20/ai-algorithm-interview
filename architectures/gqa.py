import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    """
    GQA (Grouped Query Attention)
    
    原理：将 Query 头分组，每组共享一个 KV 头。
    - n_q_heads / n_kv_heads = groups
    - 推理效率高于 MHA，表现优于 MQA。
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        # TODO: Initialize projections and groups
        pass

    def forward(self, x, mask=None):
        # TODO: Implement grouping, repeat_interleave and attention
        pass

if __name__ == "__main__":
    d_model, n_q, n_kv = 128, 8, 2
    model = GQA(d_model, n_q, n_kv)
    x = torch.randn(1, 10, d_model)
    print(f"GQA output shape: {model(x).shape}")
