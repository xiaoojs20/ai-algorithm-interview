import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-means 聚类算法
    
    原理：
    硬性聚类（Hard Clustering）。目标是将数据划分为 K 个簇，使得同一个簇内的数据点到该簇质心的欧氏距离平方和最小。
    
    流程：
    1. 随机选取 k 个作为初始质心 (Centroids)。
    2. 分配 (Assign): 计算每个点到各质心的距离，将点分配到最近的质心所属的簇。
    3. 更新 (Update): 重新计算每个簇内所有点的均值，作为该簇的新质心。
    4. 重复 2-3，直到质心位置不再变化或达到最大迭代次数。
    
    复杂度分析：
    - 时间复杂度: O(iter * k * n * d) (n:点数, d:维度)
    - 空间复杂度: O(n * d + k * d)
    """
    # 1. 随机初始化质心 (从数据点中随机选 k 个)
    n_samples, n_features = X.shape
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx]
    
    for i in range(max_iters):
        # 2. 分配：计算每个样本到各质心的距离 (n, k)
        # 用广播机制 (n, 1, d) - (k, d) -> (n, k, d) -> sum_sq(n, k)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1) # (n,)
        
        # 3. 更新：计算每簇均值
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 4. 判断收敛
        if np.all(np.abs(new_centroids - centroids) < tol):
            print(f"Converged at iteration {i}")
            break
            
        centroids = new_centroids
        
    return centroids, labels

if __name__ == "__main__":
    # 生成测试数据 (3 个簇)
    X = np.r_[np.random.randn(50, 2) + [2, 2],
              np.random.randn(50, 2) + [0, -2],
              np.random.randn(50, 2) + [-2, 2]]
    
    centroids, labels = kmeans(X, k=3)
    print(f"Centroids:\n{centroids}")
