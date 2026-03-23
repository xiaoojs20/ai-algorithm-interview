import torch
import torch.nn.functional as F

def grpo_loss(rewards, log_probs, old_log_probs, ref_log_probs, eps=0.2, beta=0.1):
    """
    GRPO (Group Relative Policy Optimization)
    
    Shapes:
    - rewards: (Batch, Group_size) - 对于每个 Prompt 的采样组奖励分数
    - log_probs, old_log_probs, ref_log_probs: (Batch, Group_size) - 对数概率
    
    原理 (DeepSeek 核心创新)：
    1. 取消 Critic: 不再需要训练复杂的 Value 模型，大大节省显存。
    2. 组内得分 (Group Relative Reward): 对同一个 Prompt 采样多个 Output (G 个)，
       用这组输出的均值作为投影基准来衡量相对优势 (Advantage)。
    """
    # 1. 组内相对优势计算 (Group Advantage)
    # rewards: (Batch_size, Group_size)
    # 计算每组内所有样本的均值和标准差
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8) # (B, G)
    
    # 2. 计算策略项 (Policy Surrogates)
    # ratio: pi / pi_old
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean() # (B, G)
    
    # 3. 计算 KL 散度约束项 (与 Reference 模型保持距离)
    # 通常使用近似公式: exp(ref_log - log) - (ref_log - log) - 1
    kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    kl_loss = kl_div.mean()
    
    # 总损失 = 策略项 + beta * KL 项
    return policy_loss + beta * kl_loss

if __name__ == "__main__":
    B, G = 2, 8 # Batch=2, 每组采样 8 个回复
    rewards = torch.randn(B, G)
    log_probs = torch.randn(B, G)
    old_log_probs = torch.randn(B, G)
    ref_log_probs = torch.randn(B, G)
    
    loss = grpo_loss(rewards, log_probs, old_log_probs, ref_log_probs)
    print(f"GRPO Loss: {loss.item():.4f}")
