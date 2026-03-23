import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """
    DeepFM (Deep Factorization Machines)
    
    原理：
    结合了 FM（因子分解机）和 Deep（深度神经网络）。
    - Wide（FM）部分：负责提取低阶特征（一阶特征和二阶交叉特征）。
    - Deep 部分：负责提取高阶特征。
    - 两部分共享输入（Embedding），可以同时学习低阶和高阶特征，无需人工特征工程。
    
    公式：
    $ y = \text{sigmoid}(y_{FM} + y_{Deep}) $
    $ y_{FM} = w_0 + \sum w_i x_i + \sum_{i<j} <v_i, v_j> x_i x_j $
    """
    def __init__(self, feature_dims, embedding_dim=4):
        super().__init__()
        # feature_dims 为每个特征的维度（通常是 category 数）
        self.num_features = len(feature_dims)
        self.total_feature_dim = sum(feature_dims)
        
        # 1. 一阶权重 (Linear Part)
        self.linear = nn.ModuleList([nn.Embedding(f, 1) for f in feature_dims])
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 2. 二阶/Deep 共享的 Embedding
        self.embeddings = nn.ModuleList([nn.Embedding(f, embedding_dim) for f in feature_dims])
        
        # 3. Deep 部分 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (Batch, Num_features) - 特征的索引
        
        # --- FM 部分 (线性项) ---
        # (Batch, 1) -> sum(Embedding(1))
        linear_part = torch.cat([self.linear[i](x[:, i]) for i in range(self.num_features)], dim=1)
        linear_out = torch.sum(linear_part, dim=1, keepdim=True) + self.bias
        
        # --- FM 部分 (二阶项) ---
        # 提取 Embedding: (Batch, Num_features, Embedding_dim)
        embeds = torch.stack([self.embeddings[i](x[:, i]) for i in range(self.num_features)], dim=1)
        
        # 使用简洁公式: sum(<v_i, v_j>) = 0.5 * ( (sum V)^2 - sum(V^2) )
        sum_of_embed = torch.sum(embeds, dim=1) # (Batch, Embedding_dim)
        sum_of_embed_square = sum_of_embed.pow(2)
        square_of_sum_embed = torch.sum(embeds.pow(2), dim=1)
        
        fm_second_order = 0.5 * torch.sum(sum_of_embed_square - square_of_sum_embed, dim=1, keepdim=True)
        
        # --- Deep 部分 ---
        mlp_in = embeds.view(x.size(0), -1) # Flatten (Batch, Num_features * Embedding_dim)
        mlp_out = self.mlp(mlp_in)
        
        # --- Total ---
        y = torch.sigmoid(linear_out + fm_second_order + mlp_out)
        return y

if __name__ == "__main__":
    feature_dims = [2, 3, 4] # 虽然是 3 个分类特征，每个特征分别有 2, 3, 4 种分类值
    model = DeepFM(feature_dims)
    x = torch.LongTensor([[0, 1, 2], [1, 2, 3]]) # Batch = 2
    print(f"Prediction: {model(x)}")
    # print(model)
