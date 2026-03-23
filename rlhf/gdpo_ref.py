import torch
import torch.nn.functional as F

def gdpo_loss(reward_tensor, log_probs, old_log_probs, ref_log_probs, eps=0.2, beta=0.1):
    """
    GDPO (Group reward-Decoupled Normalization Policy Optimization)
    
    原理 (DeepSeek-V3 核心算法之一):
    在处理多目标奖励时（如：正确性、格式、安全性），传统的做法是先求和再归一化。
    但这会导致不同的奖励信号被抹平。GDPO 提出“奖励解耦归一化”：
    1. 分解：将综合奖励分解为多个独立的奖励分量。
    2. 独立归一化：在组内 (Group) 对每一个奖励分量分别进行标准化。
    3. 合并：将标准化后的各分量优势 (Advantages) 相加。
    
    Shapes:
    - reward_tensor: (Batch, Group_size, Num_Rewards) - 多个维度的奖励输入
    - log_probs: (Batch, Group_size)
    """
    # 1. 组内奖励解耦归一化 (Reward-Decoupled Normalization)
    B, G, R = reward_tensor.shape
    
    # 对每一个奖励类型分别计算均值和标准差
    # (B, 1, R)
    mean_r = reward_tensor.mean(dim=1, keepdim=True)
    std_r = reward_tensor.std(dim=1, keepdim=True)
    
    # 独立归一化: (B, G, R)
    decoupled_advantages = (reward_tensor - mean_r) / (std_r + 1e-8)
    
    # 合并优势: 简单求和
    # (B, G)
    total_advantages = decoupled_advantages.sum(dim=-1)
    
    # 2. 策略优化 (同 PPO/GRPO 逻辑)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * total_advantages
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * total_advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 3. KL 约束
    kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    return policy_loss + beta * kl_div.mean()

if __name__ == "__main__":
    B, G, R = 2, 8, 3 # Batch=2, 每组 8 个样本, 3 种奖励(正确性、美观度、安全性)
    reward_tensor = torch.randn(B, G, R)
    log_probs = torch.randn(B, G)
    old_log_probs = torch.randn(B, G)
    ref_log_probs = torch.randn(B, G)
    
    loss = gdpo_loss(reward_tensor, log_probs, old_log_probs, ref_log_probs)
    print(f"GDPO Loss: {loss.item():.4f}")
