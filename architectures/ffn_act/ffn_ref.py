import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network (Transformer / BERT 风格)
    
    结构：
    1. Linear 上投影: d_model -> d_ff (通常 d_ff = 4 * d_model)
    2. Activation: ReLU / GELU
    3. Linear 下投影: d_ff -> d_model
    
    公式：
    $ \text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2 $
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() # 或 nn.ReLU()
        
    def forward(self, x):
        # x: (B, L, d_model)
        # 1. 第一层线性变换并激活
        x = self.act(self.w1(x))
        # 2. 第二层线性变换
        return self.w2(x)

if __name__ == "__main__":
    B, L, D = 1, 10, 128
    D_ff = 4 * D
    model = StandardFFN(D, D_ff)
    x = torch.randn(B, L, D)
    print(f"Standard FFN output shape: {model(x).shape}")
