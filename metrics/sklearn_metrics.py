# 练习调用 sklearn.metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

def practice_classification_metrics():
    """
    TODO: 练习计算以下指标
    1. Accuracy (准确率)
    2. Precision (精确率), Recall (召回率), F1-score
    3. ROC AUC Score (使用概率得分)
    4. Confusion Matrix (混淆矩阵)
    """
    y_true = [0, 1, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 1, 0, 1, 1, 1] 
    y_scores = [0.1, 0.4, 0.8, 0.2, 0.7, 0.9, 0.6] 
    pass

def practice_regression_metrics():
    """
    TODO: 练习计算以下指标
    1. Mean Squared Error (MSE)
    2. Mean Absolute Error (MAE)
    3. R2 Score (R方)
    """
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5,  0.0, 2.1, 7.8]
    pass

if __name__ == "__main__":
    practice_classification_metrics()
    practice_regression_metrics()
