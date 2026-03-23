import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss
    
    原理：
    旨在解决单阶段目标检测或高度不平衡分类任务中正负样本比例悬殊的问题。
    它通过在交叉熵损失前增加一个调制因子 (1 - p_t)^gamma，
    从而降低“容易分类（置信度高）”样本的权重，使模型专注于“难分类”的样本。
    
    公式：
    $ FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t) $
    其中 alpha 是类别权重，gamma 是平滑因子。
    
    Shapes:
    - inputs: (B, C)
    - targets: (B)
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. 计算交叉熵损失
        # 得到每个样本的第 y_i 类的预测概率 p_t
        # 用 log_softmax + nll_loss 更稳定
        log_p = F.log_softmax(inputs, dim=-1)
        ce_loss = F.nll_loss(log_p, targets, reduction='none')
        
        # 2. 计算 p_t (实际标签对应的预测概率)
        # log_p[targets]
        p_t = torch.exp(-ce_loss)
        
        # 3. 计算 Focal Loss 权重
        # (1 - p_t)^gamma
        loss = self.alpha * (1 - p_t)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

if __name__ == "__main__":
    B, C = 4, 3
    inputs = torch.randn(B, C)
    targets = torch.tensor([1, 0, 2, 1])
    
    criterion = FocalLoss(gamma=2)
    print(f"Focal Loss: {criterion(inputs, targets).item():.4f}")
