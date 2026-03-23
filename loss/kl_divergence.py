import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence_loss(y_true_prob, y_pred_log_prob):
    """
    KL Divergence (Kullback-Leibler 散度)
    
    原理：衡量两个概率分布的距离。D_KL(P||Q) = \sum P * log(P/Q)
    注意：PyTorch F.kl_div(input=log_probs, target=probs)
    """
    # TODO: Implement KL divergence with reduction='batchmean'
    pass

if __name__ == "__main__":
    B, C = 2, 5
    y_true, y_pred = F.softmax(torch.randn(B, C), dim=-1), F.log_softmax(torch.randn(B, C), dim=-1)
    print(f"KL Divergence: {kl_divergence_loss(y_true, y_pred)}")
