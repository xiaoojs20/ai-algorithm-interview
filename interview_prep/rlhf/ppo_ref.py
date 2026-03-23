import torch
import torch.nn.functional as F

def ppo_loss(log_probs, old_log_probs, advantages, eps_clip=0.2):
    """
    PPO (Proximal Policy Optimization)
    
    LaTeX Formula:
    $ \mathcal{L}_{PPO} = - \mathbb{E} [ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) ] $
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
    
    return -torch.min(surr1, surr2).mean()
