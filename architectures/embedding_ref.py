import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    """
    nn.Embedding (词嵌入层)
    
    原理：
    Embedding 层本质上是一个可学习的查找表（Look-up Table）。
    它不进行复杂的特征提取，而是将稀疏的 ID（如词索引）映射为稠密的连续向量。
    
    逻辑公式：
    $ E(x) = W[x] $
    其中 W 是权重矩阵 (Vocab_size, Embedding_dim)，x 是索引。
    
    优势：
    相比 One-hot 编码，Embedding 大大缩小了特征空间的维度。
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 内部实际上是一个可梯度的参数矩阵 (Vocab_size, Dim)
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        
    def forward(self, indices):
        # indices: (B, L) - 词索引
        # 1. 查找对应的行向量
        # Output: (B, L, Embedding_dim)
        return self.weight[indices]

# 模拟使用方式
if __name__ == "__main__":
    vocab_size = 10
    dim = 4
    
    # 1. 标准 nn.Embedding
    embed = nn.Embedding(vocab_size, dim)
    
    # 2. 调用方式
    tokens = torch.LongTensor([[1, 2, 4, 3]]) # Batch=1, Length=4
    vector = embed(tokens)
    
    print(f"Tokens: {tokens}")
    print(f"Embedding shape: {vector.shape}")
    print(f"Embedding values: {vector}")
    
    # 3. 验证自定义 Embedding 逻辑
    custom_embed = CustomEmbedding(vocab_size, dim)
    custom_vector = custom_embed(tokens)
    print(f"Custom embedding shape: {custom_vector.shape}")
