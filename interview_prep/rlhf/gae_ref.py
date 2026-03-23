import torch

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    GAE (Generalized Advantage Estimation)
    
    LaTeX Formula:
    $ A_t = \delta_t + (\gamma\lambda) A_{t+1} $
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(rewards.size(1))):
        mask = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * next_values[:, t] * mask - values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae * mask
    return advantages
