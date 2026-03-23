import numpy as np

def compute_auc(y_true, y_score):
    """
    AUC 计算 (手写版本 - 面试高频)
    
    LaTeX Formula (Rank-based):
    $ AUC = \frac{\sum_{i \in \text{pos}} \text{rank}_i - \frac{M(M+1)}{2}}{M \times N} $
    M: 正样本数, N: 负样本数
    
    核心逻辑:
    1. 对概率分数排序
    2. 计算正样本的 Rank 和 (即正样本排在负样本前的概率)
    3. 代入公式
    """
    # 1. 组合 label 和 score 并按 score 从小到大排序
    data = sorted(zip(y_score, y_true), key=lambda x: x[0])
    
    pos_indices = [i + 1 for i, (s, l) in enumerate(data) if l == 1]
    M = len(pos_indices)
    N = len(data) - M
    
    if M == 0 or N == 0:
        return 0.0
    
    # 2. 计算 Rank 和并应用公式
    rank_sum = sum(pos_indices)
    auc = (rank_sum - (M * (M + 1)) / 2) / (M * N)
    
    return auc
