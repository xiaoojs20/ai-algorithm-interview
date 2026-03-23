import torch

class AdamWOptimizer:
    """
    AdamW (Decoupled Weight Decay)
    
    LaTeX Formula:
    $ \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right) $
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01):
        self.params = list(params)
        self.lr, self.betas, self.wd = lr, betas, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            update = m_hat / (torch.sqrt(v_hat) + 1e-8)
            p.data -= self.lr * (update + self.wd * p.data)
