import torch
import torch.nn.functional as F

def ppo_loss(log_probs, old_log_probs, advantages, values=None, old_values=None, returns=None, 
             eps_clip=0.2, c1=0.5, c2=0.01):
    """
    PPO (Proximal Policy Optimization)
    
    Shapes:
    - log_probs, old_log_probs: (N,) - 为 Batch_size * Seq_len 或采样点总数
    - advantages: (N,) - 优势函数估计值
    - values, returns: (N,) - 价值函数预测与目标值
    
    原理：
    PPO 是 RLHF 中的标准策略优化算法。它通过“裁剪”（Clipping）机制限制策略更新的幅度。
    """
    # 1. 计算概率比率 ratio: r_t(theta) = pi_theta / pi_old
    ratio = torch.exp(log_probs - old_log_probs)
    
    # 2. 策略裁剪损失 (Policy Clipping)
    # 当优势 A > 0，目标是增加 ratio，但不能超过 1+eps
    # 当优势 A < 0，目标是减小 ratio，但不能低于 1-eps
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 3. 价值损失 (Value Loss - Optional in some implementations)
    # 在标准 PPO 中，通常还会对 Value Function 进行裁剪
    value_loss = 0
    if values is not None and returns is not None:
        value_loss = F.mse_loss(values, returns)
        
    # 4. 熵正则项 (Entropy)
    # 用于增加策略的多样性 (防止过早收敛到局部最优)
    entropy_loss = - (torch.exp(log_probs) * log_probs).mean()
    
    # 总损失 = 策略损失 + c1 * 价值损失 - c2 * 熵损失
    total_loss = policy_loss + c1 * value_loss - c2 * entropy_loss
    
    return total_loss, policy_loss, value_loss

if __name__ == "__main__":
    # Simulate data
    B = 4
    log_probs = torch.randn(B)
    old_log_probs = torch.randn(B)
    advantages = torch.randn(B)
    
    loss, p_loss, _ = ppo_loss(log_probs, old_log_probs, advantages)
    print(f"PPO Total Loss: {loss.item():.4f}, Policy Loss: {p_loss.item():.4f}")
