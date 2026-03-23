import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet Loss (三元组损失)
    
    原理：距离学习。L = max(d(a, p) - d(a, n) + m, 0)
    目标：使 Anchor 离正样本比离负样本近至少 margin 距离。
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # TODO: Implement squared distance and triplet loss
        pass

if __name__ == "__main__":
    B, D = 4, 128
    a, p, n = torch.randn(B, D), torch.randn(B, D), torch.randn(B, D)
    criterion = TripletLoss(margin=1.0)
    print(f"Triplet Loss: {criterion(a, p, n)}")
