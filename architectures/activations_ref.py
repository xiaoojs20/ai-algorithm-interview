import torch
import math

def sigmoid(x):
    """
    Sigmoid 激活函数
    公式：1 / (1 + exp(-x))
    """
    return 1 / (1 + torch.exp(-x))

def relu(x):
    """
    ReLU (Rectified Linear Unit)
    公式：max(0, x)
    """
    return torch.maximum(torch.zeros_like(x), x)

def silu(x):
    """
    SiLU (Sigmoid Linear Unit / Swish)
    公式：x * sigmoid(x)
    """
    return x * sigmoid(x)

def tanh(x):
    """
    Tanh (双曲正切)
    公式：(exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    exp_pos = torch.exp(x)
    exp_neg = torch.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

def gelu(x):
    """
    GELU (Gaussian Error Linear Unit) - 高斯误差线性单元
    
    原理：
    通过高斯分布的累积分布函数 (CDF) 对输入进行加权。
    在大模型 (GPT, BERT) 中被广泛使用。
    
    近似公式：
    $ 0.5x \times (1 + \tanh(\sqrt{2/\pi} \times (x + 0.044715x^3))) $
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"Input: {x}")
    print(f"Sigmoid: {sigmoid(x)}")
    print(f"ReLU: {relu(x)}")
    print(f"SiLU: {silu(x)}")
    print(f"Tanh: {tanh(x)}")
    print(f"GELU: {gelu(x)}")
