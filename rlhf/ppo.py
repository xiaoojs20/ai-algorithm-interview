import torch
import torch.nn.functional as F

def ppo_loss(log_probs, old_log_probs, advantages, values=None, returns=None, eps_clip=0.2):
    """
    PPO (Proximal Policy Optimization)
    
    原理：通过 Clipping 限制策略更新幅度。
    包含：1. Policy Loss (Clipped) 2. Value Loss (MSE) 3. Entropy Loss
    """
    # TODO: Implement PPO total loss
    pass

if __name__ == "__main__":
    B = 4
    log_probs, old_log_probs, advantages = torch.randn(B), torch.randn(B), torch.randn(B)
    print(f"PPO Loss: {ppo_loss(log_probs, old_log_probs, advantages)}")
