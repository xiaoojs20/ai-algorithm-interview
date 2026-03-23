import numpy as np

class SimpleMLP:
    """
    NumPy 手撕反向传播 (MLP with 1 Hidden Layer)
    
    结构：Input -> W1 -> ReLU -> W2 -> Softmax -> Loss(CE)
    """
    def __init__(self, d_in, d_hidden, d_out):
        self.W1, self.b1 = np.random.randn(d_in, d_hidden) * 0.01, np.zeros((1, d_hidden))
        self.W2, self.b2 = np.random.randn(d_hidden, d_out) * 0.01, np.zeros((1, d_out))
        
    def forward(self, X):
        # TODO: Implement forward pass
        pass
        
    def backward(self, X, y_true, lr=0.01):
        # y_true 为 one-hot 编码
        # TODO: Implement backward pass and parameter update
        pass

if __name__ == "__main__":
    X, y = np.random.randn(10, 4), [0, 1, 2, 1, 0, 2, 1, 0, 0, 2]
    mlp = SimpleMLP(4, 8, 3)
    # y_onehot calculation...
    pass
    # mlp.backward(...)
