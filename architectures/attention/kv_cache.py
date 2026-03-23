import torch
import torch.nn as nn

class KVCache(nn.Module):
    """
    KV Cache (Key-Value Cache)
    
    原理：在自回归推理中，将之前步生成的 K, V 缓存起来，跳过冗余注意力计算。
    - new_k, new_v: (B, H, 1, D)
    - output: (B, H, L_prev + 1, D) (拼接后的结果)
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # TODO: Define storage for layers
        pass
        
    def reset_cache(self):
        # TODO: Clear caches
        pass
        
    def forward(self, new_k, new_v, layer_idx):
        # TODO: Concatenate with previous cache
        pass

if __name__ == "__main__":
    kv_cache = KVCache(d_model=128, n_heads=4)
    B, H, L, D = 1, 4, 1, 32
    for i in range(5):
        k, v = torch.randn(B, H, 1, D), torch.randn(B, H, 1, D)
        full_k, full_v = kv_cache(k, v, layer_idx=0)
        print(f"Token {i} generated. Cache K shape: {full_k.shape}")
