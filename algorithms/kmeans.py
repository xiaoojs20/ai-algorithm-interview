import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-means 聚类算法
    
    原理：硬性聚类。
    流程：1. 随机找 K 个质心 2. 分配距离最近的簇 3. 更新质心为簇均值。
    """
    # 1. 随机选取初始质心
    # 2. 迭代分配和更新质心
    # TODO: Implement K-means
    pass

if __name__ == "__main__":
    X = np.r_[np.random.randn(50, 2) + [2, 2],
              np.random.randn(50, 2) + [0, -2],
              np.random.randn(50, 2) + [-2, 2]]
    
    centroids, labels = kmeans(X, k=3)
    print(f"Centroids:\n{centroids}")
