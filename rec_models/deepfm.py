import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """
    DeepFM (Deep Factorization Machines)
    
    原理：结合了 FM（提取低阶交叉特征）和 MLP（提取高阶稠密特征）。
    核心公式：y = sigmoid(y_linear + y_fm_second + y_deep)
    """
    def __init__(self, feature_dims, embedding_dim=4):
        super().__init__()
        # TODO: Initialize components
        pass
        
    def forward(self, x):
        # x: (Batch, Num_features)
        # TODO: Implement FM (Linear + Second-order) and Deep parts
        pass

if __name__ == "__main__":
    feature_dims = [2, 3, 4]
    model = DeepFM(feature_dims)
    x = torch.LongTensor([[0, 1, 2], [1, 2, 3]])
    print(f"Prediction: {model(x)}")
