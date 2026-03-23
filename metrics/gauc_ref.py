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
    
    # 1. 按用户 ID 分组并行处理
    # unique_users 获取数据集中所有出现过的唯一用户
    unique_users = np.unique(user_ids)
    for user in unique_users:
        # 使用 mask 提取该用户对应的所有样本及其预测值
        user_mask = (user_ids == user)
        u_labels = labels[user_mask]
        u_preds = preds[user_mask]
        
        # 2. 关键过滤逻辑：只有同时包含“正样本”和“负样本”的用户才能计算 AUC
        # 因为 AUC 是衡量排序能力的（正样本排在负样本前的概率），
        # 如果一个用户只有正样本或只有负样本，该用户的 AUC 无定义，必须剔除。
        if len(np.unique(u_labels)) == 2:
            auc = roc_auc_score(u_labels, u_preds)
            user_auc_list.append(auc)
            # 3. 记录该用户的样本数（权重），后续用于加权平均
            # 通常样本越多的活跃用户，其 AUC 对整体结果贡献越大
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
