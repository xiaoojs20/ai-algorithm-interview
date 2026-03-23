import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationUnit(nn.Module):
    """
    DIN (Deep Interest Network) - Activation Unit (Attention)
    
    LaTeX Formula:
    $ a(q, e) = w \cdot [q, e, q-e, q \times e] + b $
    
    主要考查点: 用户的点击序列历史行为 (Interest Sequence) 与 候选广告 (Target Item) 之间的权重计算。
    """
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 4, 36),
            nn.PReLU(),
            nn.Linear(36, 1)
        )

    def forward(self, query, facts):
        # query: (batch, 1, d_model) - 候选广告
        # facts: (batch, seq_len, d_model) - 历史点击序列
        seq_len = facts.size(1)
        # 将 query 复制 seq_len 遍
        queries = query.expand(-1, seq_len, -1) # (batch, seq_len, d_model)
        
        # 拼接特征: (q, f, q-f, q*f) 大模型与推荐系统的差异点就在这种局部特征工程
        combined = torch.cat([queries, facts, queries - facts, queries * facts], dim=-1)
        
        # 计算每个历史点击的权重
        scores = self.fc(combined) # (batch, seq_len, 1)
        return scores
