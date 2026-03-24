import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    """
    Batch Normalization (BN) - 卷积网络常用组件
    
    原理：对每个 Batch 的所有样本的同一维度 (C/D 维) 进行归一化。
    训练和推理行为不同：
    - 训练：计算当前 Batch 的 mean/var，并更新全局 running_mean/running_var (Exponential Moving Average)。
    - 推理：使用训练阶段累计的全局 running_mean/running_var。
    
    公式：y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Shapes:
    - 输入 x: (B, C)
    - 计算均值 mean 和方差 var 的维度: dim=0 (Batch 维)
    - gamma, beta, running_mean, running_var: (C,)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 动量参数，用于 EMA 更新全局统计值
        self.momentum = momentum
        
        # 1. 可学习参数 (Affine transformation)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 2. 全局统计值 (非权重，不参与梯度下降)
        # register_buffer 使这些张量在模型保存/加载时被包含，但不会出现在 model.parameters() 中
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        # x: (B, C)
        
        if self.training:
            # 训练阶段: 计算当前 Batch 的统计信息
            # mean: (C,)
            mean = x.mean(dim=0)
            # var: (C,) - 无偏估计 unbiased=False 符合主流算法定义
            var = x.var(dim=0, unbiased=False)
            
            # 更新全局运行统计量 (EMA 逻辑): 
            # new_running = (1 - momentum) * old_running + momentum * current
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # 使用当前 Batch 的统计量归一化
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # 推理阶段: 直接使用 EMA 累积的全局均值和方差
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            
        # 线性映射 y = x_hat * gamma + beta
        return self.gamma * x_hat + self.beta

if __name__ == "__main__":
    B, C = 4, 16
    bn = BatchNorm1d(C)
    
    # --- 训练模式 ---
    bn.train()
    x_train = torch.randn(B, C) * 10 + 5 # 模拟有偏数据
    out_train = bn(x_train)
    print(f"Train mode Mean: {out_train.mean(dim=0).mean():.4f}")
    
    # --- 推理模式 ---
    bn.eval()
    x_test = torch.randn(B, C)
    out_test = bn(x_test)
    print(f"Eval mode Mean (should use running stats): {out_test.mean(dim=0).mean():.4f}")
