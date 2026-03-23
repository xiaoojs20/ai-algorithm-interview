import torch

class SGDOptimizer:
    """
    SGD with Momentum
    
    LaTeX Formula:
    $ v_{t+1} = \mu v_t + \eta g_{t+1} $
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr, self.mu = lr, momentum
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.v[i] = self.mu * self.v[i] + self.lr * p.grad
            p.data -= self.v[i]
