import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMLoss(nn.Module):
    """
    Language Model Loss (Next Token Prediction)
    
    原理：通过平移对齐输入和标签，对最后一个预测维进行交叉熵计算。
    关键：1. Shift Align (平移对齐) 2. Masking (忽略 PAD)
    """
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # TODO: Implement shift and masking logic
        pass

if __name__ == "__main__":
    B, L, V = 2, 8, 10
    logits, labels = torch.randn(B, L, V), torch.randint(0, V, (B, L))
    labels[0, 5:] = -100
    
    criterion = LLMLoss(ignore_index=-100)
    loss = criterion(logits, labels)
    print(f"LLM Loss: {loss}")
