import numpy as np

def calculate_basic_metrics(y_true, y_pred):
    """
    手撕基础分类指标 (二分类)
    
    1. TP (True Positive): 预测为 1，实际为 1
    2. FP (False Positive): 预测为 1，实际为 0 (误报)
    3. TN (True Negative): 预测为 0，实际为 0
    4. FN (False Negative): 预测为 0，实际为 1 (漏报)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": precision, "Recall": recall, "F1": f1
    }

def calculate_auc_manual(y_true, y_scores):
    """
    手撕 AUC (基于秩/排序的计算方法)
    
    原理：
    AUC 等于随机抽取一个正样本和随机抽取一个负样本，正样本得分大于负样本得分的概率。
    公式：AUC = (sum(rank_pos) - M*(M+1)/2) / (M * N)
    其中 M 为正样本数，N 为负样本数。
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 1. 样本分类
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    m = len(pos_indices)
    n = len(neg_indices)
    
    if m == 0 or n == 0:
        return 0.5
    
    # 2. 排序并计算秩 (Rank)
    # 对所有得分进行排序，得分最小的秩为 1
    sorted_indices = np.argsort(y_scores)
    ranks = np.zeros_like(y_scores)
    ranks[sorted_indices] = np.arange(1, len(y_scores) + 1)
    
    # 3. 计算正样本的秩和
    rank_sum_pos = np.sum(ranks[pos_indices])
    
    # 4. 公式计算 AUC
    auc = (rank_sum_pos - m * (m + 1) / 2) / (m * n)
    return auc

def calculate_multi_f1(y_true, y_pred, num_classes, average='macro'):
    """
    多分类 F1 (Macro vs Micro)
    
    1. Macro-F1: 每一个类别先算 F1，再取平均（平等看待每个类，适合类别不平衡但同等重要）。
    2. Micro-F1: 先统计所有的 TP, FP, FN 再算 F1（平等看待每个样本，受大类影响大）。
    """
    # 每类的评估结果
    f1_list = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        # 为 Micro 累加
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 为 Macro 算单类 F1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        f1_list.append(f1)
        
    if average == 'macro':
        return np.mean(f1_list)
    else: # micro
        p_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        return 2*p_micro*r_micro/(p_micro+r_micro) if (p_micro+r_micro) > 0 else 0

if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_scores = [0.1, 0.9, 0.4, 0.2, 0.8]
    
    print("Basic Metrics:", calculate_basic_metrics(y_true, y_pred))
    print("Manual AUC:", calculate_auc_manual(y_true, y_scores))
    
    y_true_multi = [0, 1, 2, 0, 1, 2]
    y_pred_multi = [0, 1, 1, 0, 2, 2]
    print("Macro-F1:", calculate_multi_f1(y_true_multi, y_pred_multi, num_classes=3, average='macro'))
    print("Micro-F1:", calculate_multi_f1(y_true_multi, y_pred_multi, num_classes=3, average='micro'))
