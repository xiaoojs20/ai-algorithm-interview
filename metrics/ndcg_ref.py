import numpy as np

def calculate_dcg(relevances, k):
    """
    DCG (Discounted Cumulative Gain)
    公式：DCG_k = \sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}
    """
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def calculate_ndcg(relevances, k):
    """
    NDCG (Normalized Discounted Cumulative Gain)
    
    原理：
    用于评估排序结果。考虑了文档的相关性分级（不仅是二分类）和在结果列表中的位置。
    NDCG = DCG / IDCG
    IDCG 是理想排序下的 DCG（将相关性从高到低排列）。
    """
    # 1. 计算实际 DCG
    dcg = calculate_dcg(relevances, k)
    
    # 2. 计算理想 IDCG
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    
    # 3. 归一化
    if not idcg:
        return 0.0
    return dcg / idcg

if __name__ == "__main__":
    relevances = [3, 2, 3, 0, 1, 2]
    k = 6
    print(f"NDCG@{k}: {calculate_ndcg(relevances, k)}")
