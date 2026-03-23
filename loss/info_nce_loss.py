import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(query, positive, negatives, temperature=0.1):
    """
    InfoNCE Loss (对比学习损失函数)
    
    原理：正样本对相似度最大化，负样本对相似度最小化。
    公式：L = -log(exp(sim(q,p)/tau) / (\sum exp(sim(q,n)/tau)))
    """
    # TODO: Implement InfoNCE
    pass

if __name__ == "__main__":
    B, D, K = 4, 128, 10
    q = F.normalize(torch.randn(B, D), dim=-1)
    p = F.normalize(torch.randn(B, D), dim=-1)
    n = F.normalize(torch.randn(B, K, D), dim=-1)
    
    loss = info_nce_loss(q, p, n)
    print(f"InfoNCE Loss: {loss}")
