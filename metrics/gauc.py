import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_gauc(labels, preds, user_ids):
    """
    GAUC (Group AUC)
    
    原理：对每个用户分别计算各自的 AUC，然后根据每个用户的样本量进行加权平均。
    公式：GAUC = (\sum_u w_u * AUC_u) / (\sum_u w_u)
    """
    # TODO: Implement GAUC
    pass

if __name__ == "__main__":
    labels = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    preds = np.array([0.8, 0.2, 0.4, 0.1, 0.9, 0.7, 0.3, 0.5])
    user_ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    
    print(f"GAUC: {calculate_gauc(labels, preds, user_ids)}")
