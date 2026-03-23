import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(query, positive, negatives, temperature=0.1):
    """
    InfoNCE Loss (对比学习中的高频损失函数)
    
    原理：
    通过对比正样本和负样本对来学习表示。目标是使正样本对之间的相似度尽可能大，
    而负样本对之间的相似度尽可能小。本质上是 $(k+1)$ 类分类任务的交叉熵。
    
    公式：
    $ L = -\log \frac{\exp(sim(q, p) / \tau)}{\exp(sim(q, p) / \tau) + \sum_{i=1}^k \exp(sim(q, n_i) / \tau)} $
    
    Shapes:
    - query: (B, D)
    - positive: (B, D)
    - negatives: (B, K, D) 其中 K 是负样本数量
    """
    # 1. 计算正样本相似度: (B, D) * (B, D) -> (B, 1)
    pos_sim = torch.sum(query * positive, dim=-1, keepdim=True) / temperature
    
    # 2. 计算负样本相似度: (B, 1, D) @ (B, D, K) -> (B, K)
    # query.unsqueeze(1): (B, 1, D)
    # negatives.transpose(1, 2): (B, D, K)
    neg_sim = torch.bmm(query.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature
    
    # 3. 拼接正负样本相似度: (B, 1 + K)
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    
    # 4. 标签全为 0 (因为正样本总是在第 0 位)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
    
    # 5. 计算交叉熵
    return F.cross_entropy(logits, labels)

if __name__ == "__main__":
    B, D, K = 4, 128, 10
    q = F.normalize(torch.randn(B, D), dim=-1)
    p = F.normalize(torch.randn(B, D), dim=-1)
    n = F.normalize(torch.randn(B, K, D), dim=-1)
    
    loss = info_nce_loss(q, p, n)
    print(f"InfoNCE Loss: {loss.item():.4f}")
