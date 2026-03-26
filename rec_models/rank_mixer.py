import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTokenMixing(nn.Module):
    """
    Multi-Head Token Mixing (RankMixer)
    
    原理：通过 $L \times L$ 矩阵在每个 Head 内混合跨 Token (Field) 的特征信息。
    优势：取代 Self-Attention，硬件利用率 (MFU) 高。
    """
    def __init__(self, n_tokens, d_model, n_heads):
        super().__init__()
        # TODO: Initialize mixing weights (n_heads, n_tokens, n_tokens)
        pass
        
    def forward(self, x):
        # TODO: Mix across token dimensions using batch matmul
        pass

class PerTokenFFN(nn.Module):
    """
    Per-Token FFN (PFFN)
    
    原理：为每个 Token (Field) 提供独立的 MLP 权重，适应异构特征空间。
    """
    def __init__(self, n_tokens, d_model, d_ff):
        super().__init__()
        # TODO: Initialize per-token weights W1, W2
        pass
        
    def forward(self, x):
        # TODO: Implement independent MLP for each token
        pass

if __name__ == "__main__":
    B, L, D = 4, 20, 64
    x = torch.randn(B, L, D)
    # 实例化并运行测试
    pass
