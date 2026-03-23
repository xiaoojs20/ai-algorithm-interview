import torch
import math

def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    # TODO: Implement sigmoid
    pass

def relu(x):
    """ReLU: max(0, x)"""
    # TODO: Implement relu
    pass

def silu(x):
    """SiLU: x * sigmoid(x)"""
    # TODO: Implement silu (swish)
    pass

def gelu(x):
    """GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    # TODO: Implement gelu
    pass

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"Sigmoid: {sigmoid(x)}")
    print(f"ReLU: {relu(x)}")
    print(f"SiLU: {silu(x)}")
    print(f"GELU: {gelu(x)}")
