import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMLoss(nn.Module):
    """
    Language Model Loss (Next Token Prediction)
    
    原理：
    因果语言模型（Causal LLM）的目标是根据前文预测下一个词。
    核心逻辑：交叉熵 (Cross Entropy) + 掩码处理 (Masking)。
    
    关键实现：
    1. 平移对齐 (Shift Align): 预测位置 i 的输入对应的是 i+1 的标签。
    2. 忽略填充 (Ignore Index): 对于 PAD 字符不计算损失。
    
    Shapes:
    - logits: (Batch, Seq_len, Vocab_size)
    - labels: (Batch, Seq_len)
    """
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # 1. 平移对齐 (Shift Align)
        # logits 包含从 0 到 L-2 的输入预测
        # labels 包含从 1 到 L-1 的目标词
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 2. 形状展平 (Flatten) 以适应 CrossEntropyLoss
        # (Batch * Seq_len-1, Vocab_size)
        # (Batch * Seq_len-1)
        B, L, V = shift_logits.shape
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)
        
        # 3. 计算交叉熵
        # ignore_index 会自动过滤掉掩码位置
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=self.ignore_index)

if __name__ == "__main__":
    B, L, V = 2, 8, 10
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    
    # 模拟填充部分标签
    labels[0, 5:] = -100
    
    criterion = LLMLoss(ignore_index=-100)
    loss = criterion(logits, labels)
    
    print(f"LLM Loss: {loss.item():.4f}")
