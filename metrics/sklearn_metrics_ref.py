from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

def classification_metrics_demo():
    """
    分类指标调用案例
    """
    y_true = [0, 1, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 1, 0, 1, 1, 1] # 离散预测值
    y_scores = [0.1, 0.4, 0.8, 0.2, 0.7, 0.9, 0.6] # 概率预测值 (Score)
    
    # 1. 准确率 Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # 2. 精确率 (Precision) 和 召回率 (Recall)
    # 精确率 (查准率)：TP / (TP + FP)
    # 原理：在所有模型预测为“正”的样本里，有多少是真的“正”样本？
    # 意义：衡量模型“说准”的能力，防止“错杀”（FP 要低）。
    precision = precision_score(y_true, y_pred)
    
    # 召回率 (查全率)：TP / (TP + FN)
    # 原理：在所有实际为“正”的样本里，模型成功抓到了多少个？
    # 意义：衡量模型“找全”的能力，防止“漏网之鱼”（FN 要低）。
    recall = recall_score(y_true, y_pred)
    
    # 3. F1 Score (2 * P * R / (P + R))
    # 原理：精确率和召回率的调和平均数。
    # 意义：用于综合评估。当 P 和 R 出现矛盾时（一个高一个低），F1 会偏向较低的那个值，
    # 从而迫使模型在查准和查全之间寻找平衡点。常用于类别不平衡场景。
    f1 = f1_score(y_true, y_pred)
    
    # 4. AUC (Area Under ROC Curve) - 使用概率得分
    auc = roc_auc_score(y_true, y_scores)
    
    # 5. 混淆矩阵 (Confusion Matrix)
    # 返回 [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    
    # 6. 综合报告
    report = classification_report(y_true, y_pred)
    
    print("--- Classification Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    # print(f"Report:\n{report}")


def regression_metrics_demo():
    """
    回归指标调用案例
    """
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5,  0.0, 2.1, 7.8]
    
    # 1. 均方误差 (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)
    
    # 2. 平均绝对误差 (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 3. R2 Score (决定系数) - 越接近 1 说明拟合越好
    r2 = r2_score(y_true, y_pred)
    
    print("\n--- Regression Metrics ---")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    classification_metrics_demo()
    regression_metrics_demo()
