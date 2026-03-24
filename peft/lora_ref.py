import torch
import torch.nn as nn

class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) - 低秩自适应
    
    原理介绍：
    LoRA 是目前最流行的参数高效微调（PEFT）方法。其核心逻辑是在冻结预训练权重 $W$ 的基础上，引入旁路分支进行低秩分解。
    
    核心优势：
    1. 极低参数量：只训练 A 和 B 两个小矩阵，参数量通常只有原模型的 0.01% - 1%。
    2. 推理无延迟：训练完成后，可以通过 $W' = W + \frac{\alpha}{r}BA$ 将旁路参数合并回主路径。
    3. 模块化：可以为不同任务训练不同的 LoRA 权重，推理时无缝切换。

    LaTeX 公式:
    $ h = Wx + \Delta Wx = Wx + \frac{\alpha}{r} (B A x) $
    其中 $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$，$r$ 是秩（Rank），通常 $r \ll d$。
    
    初始化策略（面试必考）：
    - A 矩阵：采用**随机高斯初始化 (Random Gaussian)**，提供初始权重。
    - B 矩阵：必须初始化为 **全 0**。
    - 结果：通过 $B=0$，保证在训练刚开始时 $\Delta W = BA = 0$，此时模型输出与原始预训练模型完全一致，确保训练起点的稳定性。
    """
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        
        # Scaling Factor (缩放因子): alpha / r
        # 目的是减少因改变秩 r 而需要频繁调优学习率的需求
        self.scaling = alpha / r
        
        # 1. 旁路 A 矩阵: [d_in, r] - 高斯初始化
        # 按照 LoRA 论文原文，A 矩阵应采用随机高斯初始化 (Random Gaussian)
        self.lora_A = nn.Parameter(torch.empty(d_in, r))
        # nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02) 

        # 2. 旁路 B 矩阵: [r, d_out] - 初始化为全 0
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        
    def forward(self, x, W_base):
        # x: (B, L, d_in), W_base: 原模型冻结后的权重 [d_in, d_out]
        
        # 路径 1: 原始路径 (Frozen)
        base_out = x @ W_base
        
        # 路径 2: LoRA 低秩路径 (Trainable)
        # 实现顺序: (x @ A) @ B 比 x @ (A @ B) 更高效，因为它避免了构造大的 [d_in, d_out] 矩阵
        # 形状变化: (B, L, d_in) -> (B, L, r) -> (B, L, d_out)
        lora_delta = (x @ self.lora_A) @ self.lora_B
        
        # 最终输出 = 原始输出 + (低秩差值 * 缩放因子)
        return base_out + lora_delta * self.scaling
