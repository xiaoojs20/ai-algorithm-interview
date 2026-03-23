import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet Loss (三元组损失)
    
    原理：
    用于学习度量空间中的嵌入表现。输入为一个三元组：Anchor (锚点), Positive (正样本), Negative (负样本)。
    目标是使 Anchor 到 Positive 的距离小于 Anchor 到 Negative 的距离，且至少相差一个 Margin。
    
    公式：
    $ L = \max(d(a, p) - d(a, n) + \text{margin}, 0) $
    
    应用：
    人脸识别 (FaceNet)、图像检索、语音指纹。
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 1. 计算欧几里得距离的平方
        d_ap = torch.sum((anchor - positive).pow(2), dim=-1)
        d_an = torch.sum((anchor - negative).pow(2), dim=-1)
        
        # 2. 计算损失
        # 只有在 d_ap - d_an + margin > 0 时才贡献 loss
        loss = F.relu(d_ap - d_an + self.margin)
        
        return loss.mean()

if __name__ == "__main__":
    B, D = 4, 128
    a = torch.randn(B, D)
    p = torch.randn(B, D)
    n = torch.randn(B, D)
    
    criterion = TripletLoss(margin=1.0)
    print(f"Triplet Loss: {criterion(a, p, n).item():.4f}")
