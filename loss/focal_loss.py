import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (用于样本不平衡)
    
    原理：对容易分类的样本降权，通过 (1-p_t)^gamma 调节聚焦于难分类样本。
    公式：FL = -alpha * (1-p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # TODO: Implement Focal Loss logic
        pass

if __name__ == "__main__":
    B, C = 4, 3
    inputs, targets = torch.randn(B, C), torch.tensor([1, 0, 2, 1])
    print(f"Focal Loss: {FocalLoss()(inputs, targets)}")
