import torch
import torch.nn as nn

class KVCache(nn.Module):
    """
    KV Cache (Key-Value Cache)
    
    原理：
    在自回归（Autoregressive）生成模型中，推理过程是按字（Token）生成的。
    每个新生成的字只需要跟之前的字计算注意力（Attention）。
    - 之前的字的 Q 是用不到的。
    - 之前的字的 K, V 在每次生成新字时都会被重新计算，造成巨大浪费。
    因此，可以通过缓存 K, V 来跳过重复计算，使推理速度不再随序列长度平方级增长。
    
    Shapes:
    - K, V: (B, H, L_prev, D)
    - new_k, new_v: (B, H, 1, D)
    - output: (B, H, L_prev+1, D)
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 定义一个字典存储每个 layer 的 KV 缓存
        self.cache_k = {}
        self.cache_v = {}
        
    def reset_cache(self):
        self.cache_k.clear()
        self.cache_v.clear()
        
    def forward(self, new_k, new_v, layer_idx):
        # new_k, new_v shape: (B, H, 1, D)
        
        if layer_idx not in self.cache_k:
            # 首次进入，直接存储
            self.cache_k[layer_idx] = new_k
            self.cache_v[layer_idx] = new_v
        else:
            # 拼接之前的缓存
            # self.cache_k[layer_idx] shape: (B, H, L_prev, D)
            # 拼接后: (B, H, L_prev + 1, D)
            self.cache_k[layer_idx] = torch.cat([self.cache_k[layer_idx], new_k], dim=2)
            self.cache_v[layer_idx] = torch.cat([self.cache_v[layer_idx], new_v], dim=2)
            
        return self.cache_k[layer_idx], self.cache_v[layer_idx]

# 模拟推理过程
if __name__ == "__main__":
    kv_cache = KVCache(d_model=128, n_heads=4)
    B, H, L, D = 1, 4, 1, 32
    
    # 模拟生成 5 个词
    for i in range(5):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        
        # 获取拼接后的完整 KV
        full_k, full_v = kv_cache(k, v, layer_idx=0)
        print(f"Token {i} generated. Cache K shape: {full_k.shape}")
