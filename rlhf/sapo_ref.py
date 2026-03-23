import torch
import torch.nn.functional as F

class SAPO(torch.nn.Module):
    """
    SAPO (Soft Adaptive Policy Optimization)
    
    原理 (DeepSeek 推理模型核心优化之一):
    PPO 的硬裁剪 (Hard Clipping) 在处理长序列或采样差异大时会导致训练不稳定。
    SAPO 提出“软自适应门控” (Soft Adaptive Gating)，取代简单的 min(ratio, clip)。
    
    关键实现：
    1. 平滑增益函数: 利用类似 sec^2(x) 或温度控制的函数，对偏离策略较远的采样点进行软性的梯度衰减，而不是直接抛弃。
    2. 细粒度控制: 针对每个 token 进行动态自适应调整权重，保证更新不偏离序列整体的置信域 (Trust Region)。
    
    公式简化示意：
    L = - (Gate(ratio) * Advantage)
    其中 Gate(r) = 1 / (r^2 + (1-r)^2) 等类似形式。
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, log_probs, old_log_probs, advantages):
        # ratio: pi / pi_old
        ratio = torch.exp(log_probs - old_log_probs)
        
        # 1. 软自适应门控 (Soft Gating)
        # 这里用类似 Soft-clipping 的形式：对比 PPO 的 clamp，使用连续可微的门控函数
        # SAPO 核心逻辑：对过大或过小的 ratio，自适应地衰减优势贡献 (而不是暴力截断)
        gate = 1.0 / (1.0 + torch.pow(ratio - 1.0, 2) / self.temperature)
        
        # 2. 计算最终 Loss
        # Gate 的形状与 ratio 一致，起到根据偏离程度动态分配权重的效果
        weighted_surrogate = gate * ratio * advantages
        
        # 为了展示对比，可以返回一个 policy loss
        return -weighted_surrogate.mean()

if __name__ == "__main__":
    B, L = 4, 128
    log_probs = torch.randn(B, L)
    old_log_probs = torch.randn(B, L)
    advantages = torch.randn(B, L)
    
    criterion = SAPO(temperature=0.1)
    loss = criterion(log_probs, old_log_probs, advantages)
    
    print(f"SAPO Loss: {loss.item():.4f}")
