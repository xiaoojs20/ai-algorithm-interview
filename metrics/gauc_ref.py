import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_gauc(labels, preds, user_ids):
    """
    GAUC (Group AUC)
    
    原理：
    传统的 AUC 是全局的，但在推荐系统中，更关注模型在“每个用户”内部的排序能力。
    GAUC 对每个用户分别计算 AUC，然后根据每个用户的样本量（或者点击量）进行加权平均。
    
    公式：
    GAUC = \frac{\sum_u w_u \times AUC_u}{\sum_u w_u}
    其中 w_u 是用户 u 的样本数。
    """
    user_auc_list = []
    user_weight_list = []
    
    # 按用户分组处理
    unique_users = np.unique(user_ids)
    for user in unique_users:
        user_mask = (user_ids == user)
        u_labels = labels[user_mask]
        u_preds = preds[user_mask]
        
        # 只有同时包含正负样本的用户才能计算 AUC
        if len(np.unique(u_labels)) == 2:
            auc = roc_auc_score(u_labels, u_preds)
            user_auc_list.append(auc)
            user_weight_list.append(len(u_labels))
            
    if not user_auc_list:
        return 0.5
        
    return np.average(user_auc_list, weights=user_weight_list)

if __name__ == "__main__":
    labels = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    preds = np.array([0.8, 0.2, 0.4, 0.1, 0.9, 0.7, 0.3, 0.5])
    user_ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    
    print(f"Global AUC: {roc_auc_score(labels, preds)}")
    print(f"GAUC: {calculate_gauc(labels, preds, user_ids)}")
