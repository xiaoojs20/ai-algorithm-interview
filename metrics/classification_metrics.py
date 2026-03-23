import numpy as np

def calculate_basic_metrics(y_true, y_pred):
    """
    TODO: 手撕 TP, FP, TN, FN 统计。
    1. 计算 Precision, Recall, F1。
    """
    pass

def calculate_auc_manual(y_true, y_scores):
    """
    TODO: 手撕基于秩 (Rank) 的 AUC 计算。
    公式：AUC = (sum(rank_pos) - m*(m+1)/2) / (m * n)
    其中 m 为正样本数，n 为负样本数。
    """
    pass

def calculate_multi_f1(y_true, y_pred, average='macro'):
    """
    TODO: 手撕多分类 F1。
    1. Macro-F1 (每类 F1 的均值)
    2. Micro-F1 (全局 TP/FP/FN 的 F1)
    """
    pass

if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    # 练习代码
    print(f"Basic Metrics: {calculate_basic_metrics(y_true, y_pred)}")
    print(f"Manual AUC: {calculate_auc_manual(y_true, y_scores)}")
    print(f"Macro-F1: {calculate_multi_f1(y_true, y_pred, 'macro')}")
