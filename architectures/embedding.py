import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    """
    nn.Embedding (词嵌入层)
    
    原理：词嵌入是一个可学习的查找表 (Look-up Table)。
    公式：E(x) = W[x]，即直接取权重矩阵中对应的行向量。
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # TODO: Initialize embedding matrix
        pass
        
    def forward(self, indices):
        # TODO: Lookup index in the matrix
        pass

if __name__ == "__main__":
    vocab_size, dim = 10, 4
    tokens = torch.LongTensor([[1, 2, 4, 3]])
    
    # 模拟使用方式
    embed = nn.Embedding(vocab_size, dim)
    vector = embed(tokens)
    print(f"Embedding shape: {vector.shape}")
