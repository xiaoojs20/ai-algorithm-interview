import numpy as np

class SimpleMLP:
    """
    NumPy 手撕反向传播 (MLP with 1 Hidden Layer)
    
    网络结构：
    Input (D_in) -> Linear (W1, b1) -> ReLU -> Linear (W2, b2) -> Softmax -> Loss(CE)
    
    梯度推导核心公式：
    1. dL/dz2 = y_pred - y_true (Softmax + CrossEntropy 联合导数)
    2. dW2 = a1.T @ dL/dz2
    3. db2 = sum(dL/dz2, axis=0)
    4. da1 = dL/dz2 @ W2.T
    5. dz1 = da1 * (z1 > 0) (ReLU 导数)
    6. dW1 = X.T @ dz1
    7. db1 = sum(dz1, axis=0)
    """
    def __init__(self, d_in, d_hidden, d_out):
        # 随机初始化权重
        self.W1 = np.random.randn(d_in, d_hidden) * 0.01
        self.b1 = np.zeros((1, d_hidden))
        self.W2 = np.random.randn(d_hidden, d_out) * 0.01
        self.b2 = np.zeros((1, d_out))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        # 减去 max 以保证数值稳定性
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # 1. 第一层前向
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # 2. 第二层前向
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs
        
    def backward(self, X, y_true, lr=0.01):
        # y_true 为 one-hot 编码 (B, D_out)
        batch_size = X.shape[0]
        
        # 1. 计算输出层误差 delta2 (dL/dz2)
        # 巧合且优美的公式：Softmax + CrossEntropy 的偏导 = y_pred - y_true
        delta2 = (self.probs - y_true) / batch_size
        
        # 2. 计算 W2, b2 的梯度
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # 3. 计算隐藏层误差 delta1 (dL/dz1)
        # 反向传播误差：delta2 @ W2.T
        # 再乘以激活层导数：* (z1 > 0)
        delta1 = np.dot(delta2, self.W2.T)
        delta1[self.z1 <= 0] = 0 # ReLU 导数：z1 <= 0 则为 0，否则为原值
        
        # 4. 计算 W1, b1 的梯度
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # 5. 更新参数 (梯度下降)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

if __name__ == "__main__":
    # 生成模拟数据
    X = np.random.randn(10, 4) # 10 个样本, 输入 4 维
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0, 0, 2])
    y_true_onehot = one_hot(y, num_classes=3)
    
    mlp = SimpleMLP(d_in=4, d_hidden=8, d_out=3)
    
    # 模拟训练 10 轮
    for i in range(10):
        probs = mlp.forward(X)
        loss = -np.mean(np.sum(y_true_onehot * np.log(probs + 1e-8), axis=1))
        mlp.backward(X, y_true_onehot, lr=0.1)
        print(f"Epoch {i}, Loss: {loss:.4f}")
