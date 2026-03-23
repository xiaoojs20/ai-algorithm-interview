import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence_loss(y_true_prob, y_pred_log_prob):
    """
    KL Divergence (Kullback-Leibler 散度)
    
    原理：
    用于衡量两个概率分布之间的差异。
    若 y_true 为目标分布，y_pred 为模型输出分布。
    注意：在 PyTorch 中使用 F.kl_div，输入的 preds 应已经是 log_softmax 值。
    
    公式：
    $ D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)} = \sum P(x) (\log P(x) - \log Q(x)) $
    
    应用：
    变分自编码器 (VAE)、知识蒸馏 (Knowledge Distillation)。
    """
    # y_true_prob: (B, C) - 真实概率分布
    # y_pred_log_prob: (B, C) - 预测分布的对数概率 (log_softmax)
    
    # 选项：reduction='batchmean' 是 KL 散度的数学定义常用项
    loss = F.kl_div(y_pred_log_prob, y_true_prob, reduction='batchmean')
    
    return loss

if __name__ == "__main__":
    B, C = 2, 5
    # 生成概率分布
    y_true = F.softmax(torch.randn(B, C), dim=-1)
    
    # 生成对数概率分布
    y_pred = F.log_softmax(torch.randn(B, C), dim=-1)
    
    loss = kl_divergence_loss(y_true, y_pred)
    print(f"KL Divergence: {loss.item():.4f}")
