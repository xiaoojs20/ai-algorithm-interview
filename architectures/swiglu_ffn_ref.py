import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """
    FFN with SwiGLU Activation (LLaMA / PaLM 风格)
    
    原理：
    传统的 FFN 是两层线性变换中间夹一个 ReLU/GELU。
    SwiGLU 是门控线性单元 (GLU) 的变体，它将激活函数替换为 Swish (SiLU)。
    
    公式：
    $ \text{SwiGLU}(x, W, V) = \text{SiLU}(xW) \otimes (xV) $
    $ \text{FFN}_{SwiGLU}(x, W, V, W_2) = (\text{SiLU}(xW) \otimes (xV)) W_2 $
    
    优势：
    - 具有门控特性，模型可以选择性地让信息通过。
    - 在多项 LLM 研究中被证明比 GELU/ReLU 具有更好的收敛性能。
    """
    def __init__(self, d_model, d_ff, multiple_of=256):
        super().__init__()
        # 往往 d_ff 会根据 LLaMA 策略调整为 2/3 的隐藏层大小并对齐到 multiple_of
        # 这里简化为直接传入
        
        # 1. 投影层：将 x 投影到 2 路分支 (Gate + Content)
        # 用一个大的矩阵直接完成两路投影更高效 (d_model -> 2 * d_ff)
        self.w_gate_content = nn.Linear(d_model, 2 * d_ff, bias=False)
        
        # 2. 下投影层：将合并后的维度映射回 d_model
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        # x: (B, L, d_model)
        
        # 1. 第一步：线性投影得到两组分量 (Gate 和 Value/Content)
        # (B, L, 2 * d_ff)
        gate_content = self.w_gate_content(x)
        
        # 2. 第二步：切分为两部分
        # gate: 用于产生门控信号；content: 存储内容
        gate, content = gate_content.chunk(2, dim=-1) # (B, L, d_ff)
        
        # 3. 第三步：SwiGLU 激活计算
        # SiLU (Swish) 激活门控信号，再与内容对应位置相乘 (Hadamard Product)
        # F.silu(gate) * content
        swiglu_out = F.silu(gate) * content
        
        # 4. 第四步：输出投影 (Down Projection)
        return self.w_down(swiglu_out)

if __name__ == "__main__":
    B, L, D = 2, 10, 128
    D_ff = (D * 8 // 3) # 模拟 LLaMA 样式的 FFN 维度
    
    model = SwiGLUFFN(D, D_ff)
    x = torch.randn(B, L, D)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
