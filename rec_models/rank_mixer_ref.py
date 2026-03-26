import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTokenMixing(nn.Module):
    """
    Multi-Head Token Mixing (RankMixer 核心组件)
    
    原理：
    取代 Transformer 的 Self-Attention。Self-Attention 是 O(L^2 * D)，且在不同特征空间
    算相似度可能存在语义不通的问题。
    Token Mixing 直接对 Embedding 的每一维或每一组维度 (Head) 进行跨 Token (Field) 的线性混合。
    
    Shapes:
    - Input x: (B, L, D) - B: Batch, L: Num Fields/Tokens, D: Dim
    - Output: (B, L, D)
    """
    def __init__(self, n_tokens, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 每个 Head 拥有一个独立的 L x L 混合矩阵
        # (H, L, L)
        self.mixing_weights = nn.Parameter(torch.randn(n_heads, n_tokens, n_tokens) * 0.02)
        
    def forward(self, x):
        B, L, D = x.shape
        # 1. 切分多头: (B, L, D) -> (B, L, H, d_h) -> (B, H, d_h, L)
        x = x.view(B, L, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        
        # 2. 跨 Token 混合: (B, H, d_h, L) @ (H, L, L) -> (B, H, d_h, L)
        # 这里实际上是在每个 Head 内，让 L 个 Token 的信息进行全连接混合
        out = torch.matmul(x, self.mixing_weights)
        
        # 3. 还原形状: (B, H, d_h, L) -> (B, L, H, d_h) -> (B, L, D)
        out = out.permute(0, 3, 1, 2).contiguous().view(B, L, D)
        return out

class PerTokenFFN(nn.Module):
    """
    Per-Token FFN (PFFN)
    
    原理：
    Transformer 中所有 Token 共享同一个 FFN。
    RankMixer 为每个 Token (Field) 提供独立的 FFN 权重。
    这允许模型为不同的特征空间 (如 UserID 空间 vs Item 空间) 学习完全不同的非线性变换。
    """
    def __init__(self, n_tokens, d_model, d_ff):
        super().__init__()
        # 使用批量矩阵乘法实现：每个 Token 独享一组 W1, W2
        # W1: (L, D, D_ff), W2: (L, D_ff, D)
        self.w1 = nn.Parameter(torch.randn(n_tokens, d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_tokens, d_ff, d_model) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(n_tokens, 1, d_ff))
        self.b2 = nn.Parameter(torch.zeros(n_tokens, 1, d_model))
        
    def forward(self, x):
        # x: (B, L, D) -> 转置为 (L, B, D) 以匹配批量乘法
        x = x.transpose(0, 1)
        
        # 第一层: (L, B, D) @ (L, D, D_ff) -> (L, B, D_ff)
        x = torch.bmm(x, self.w1) + self.b1
        x = F.silu(x) # 常用 SiLU/GELU
        
        # 第二层: (L, B, D_ff) @ (L, D_ff, D) -> (L, B, D)
        x = torch.bmm(x, self.w2) + self.b2
        
        # 还原为 (B, L, D)
        return x.transpose(0, 1)

class RankMixerBlock(nn.Module):
    def __init__(self, n_tokens, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.token_mixing = MultiHeadTokenMixing(n_tokens, d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.pffn = PerTokenFFN(n_tokens, d_model, d_ff)
        
    def forward(self, x):
        # 残差结构
        x = x + self.token_mixing(self.ln1(x))
        x = x + self.pffn(self.ln2(x))
        return x

class RankMixer(nn.Module):
    """
    RankMixer (arXiv:2507.15551) - ByteDance
    
    面试核心点：
    1. 硬件友好 (Hardware-Aware): 抛弃 Self-Attention (O(L^2))，使用线性复杂度的 Token Mixing。
    2. 特征建模：通过 PFFN 为每个特有 Field 建模独立子空间。
    3. 极高性能：相比 Transformer，GPU 显存利用率 (MFU) 从 4% 提升到 40%+。
    """
    def __init__(self, n_tokens, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            RankMixerBlock(n_tokens, d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        # 最终输出层: 展平所有 Token 后接 MLP
        self.head = nn.Sequential(
            nn.Linear(n_tokens * d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, L, D) - 已完成 Embedding 的特征
        for block in self.blocks:
            x = block(x)
            
        # 展平输出
        out = x.view(x.size(0), -1)
        return self.head(out)

if __name__ == "__main__":
    B, L, D = 4, 20, 64
    model = RankMixer(n_tokens=L, d_model=D, n_heads=4, d_ff=128, n_layers=2)
    x = torch.randn(B, L, D)
    y = model(x)
    print(f"RankMixer Input: {x.shape}")
    print(f"RankMixer Output: {y.shape}")
