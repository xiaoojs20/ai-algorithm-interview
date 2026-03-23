import torch

def grpo_step(rewards, log_probs, old_log_probs, eps=0.2):
    """
    GRPO (Group Relative Policy Optimization) - DeepSeek 独家高频
    
    LaTeX Formula:
    $ A_i = \frac{r_i - \text{mean}(G)}{\text{std}(G) + \epsilon} $
    """
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    
    return -torch.min(surr1, surr2).mean()
