import numpy as np

def calculate_ndcg(relevances, k):
    """
    NDCG (Normalized Discounted Cumulative Gain)
    
    原理：考虑相关性分级（不仅是二分类）和位置折扣因子。
    公式：DCG_k = \sum_{i=1}^k rel_i / \log_2(i+1), NDCG = DCG / IDCG
    """
    # TODO: Implement NDCG
    pass

if __name__ == "__main__":
    relevances = [3, 2, 3, 0, 1, 2]
    k = 6
    print(f"NDCG@{k}: {calculate_ndcg(relevances, k)}")
