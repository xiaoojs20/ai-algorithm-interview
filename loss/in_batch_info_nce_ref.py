import torch
import torch.nn as nn
import torch.nn.functional as F

def in_batch_info_nce_loss(query, positive, temperature=0.1):
    """
    In-Batch InfoNCE Loss (常见于 CLIP, SimCLR, Sentence-BERT)
    
    原理：
    在双塔模型中，一个 Batch 包含 B 对 (query, positive)。
    对于第 i 个 query，只有第 i 个 positive 是它的正样本，
    而 Batch 内其余所有的 B-1 个 positive 都被视作它的负样本（In-batch Negatives）。
    这种方法极大地提高了负样本的利用率，无需显式构造负样本。
    
    公式：
    $ S_{ij} = \frac{sim(q_i, p_j)}{\tau} $
    $ L = \text{CrossEntropy}(S, \text{arange}(B)) $
    
    Shapes:
    - query: (Batch, Dim)
    - positive: (Batch, Dim)
    """
    batch_size = query.size(0)
    
    # 1. 归一化 (L2 Normalize)
    query = F.normalize(query, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    
    # 2. 计算点积相似度矩阵 (B, B)
    # logits[i, j] 表示第 i 个 query 与第 j 个 positive 的相似度
    logits = torch.matmul(query, positive.transpose(0, 1)) / temperature
    
    # 3. 构造标签：对于第 i 行，正样本在第 i 列
    labels = torch.arange(batch_size, device=query.device)
    
    # 4. 计算对称损失 (可选，双向匹配)
    # q -> p 的损失
    loss_q = F.cross_entropy(logits, labels)
    # p -> q 的损失 (对偶项)
    loss_p = F.cross_entropy(logits.transpose(0, 1), labels)
    
    return (loss_q + loss_p) / 2

if __name__ == "__main__":
    B, D = 4, 128
    q = torch.randn(B, D)
    p = torch.randn(B, D)
    
    loss = in_batch_info_nce_loss(q, p)
    print(f"In-Batch InfoNCE Loss: {loss.item():.4f}")
