import torch
import torch.nn as nn
import math

def flash_attention_sim(q, k, v, block_size=128):
    """
    FlashAttention 原理模拟 (Simulation)
    
    原理：
    FlashAttention 的核心是 Tiling（分块）。
    它不是一次性计算整个 NxN 的 Attention 矩阵（造成显存爆炸），
    而是将 Q, K, V 分成小块加载到高效的 SRAM 中，逐步更新 softmax 的归一化常数。
    
    关键技术：
    1. Tiling: 将计算拆分为子块。
    2. Online Softmax: 动态更新局部最大值和归一化和，最后统一。
    3. Recompution: 反向传播时不保存中间状态，而是利用前向计算过程。
    
    Shapes:
    - q, k, v: (B, H, L, D)
    - block_size: 分块大小
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    # 逻辑上的分块处理 (逻辑模拟，并非真实并行)
    # 实际上 Flash 还会区分不同的 tiling 因子 (Tr, Tc)
    output = torch.zeros_like(q)
    
    # 对 Batch 和 Head 并行
    for b in range(B):
        for h in range(H):
            qi = q[b, h]
            ki = k[b, h]
            vi = v[b, h]
            
            # 维护全局 softmax 信息：当前最大值 m_i 和累加和 l_i
            # 这里的 L 分块逻辑
            m_prev = torch.full((L, 1), -float('inf'), device=q.device)
            l_prev = torch.zeros((L, 1), device=q.device)
            o_prev = torch.zeros((L, D), device=q.device)
            
            # 遍历 K, V 的序列分块 (Outer loop over keys)
            for j in range(0, L, block_size):
                kj = ki[j : j + block_size]
                vj = vi[j : j + block_size]
                
                # 遍历 Q 的序列分块 (Inner loop over queries)
                for i in range(0, L, block_size):
                    qi_block = qi[i : i + block_size] # (Br, D)
                    
                    # 1. 计算局部 Score: (Br, D) @ (D, Bc) -> (Br, Bc)
                    S_ij = (qi_block @ kj.transpose(-2, -1)) * scale
                    
                    # 2. 计算当前块的最大值
                    m_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]
                    
                    # 3. 计算指数项和累积和
                    P_ij = torch.exp(S_ij - m_ij)
                    l_ij = torch.sum(P_ij, dim=-1, keepdim=True)
                    
                    # 4. 更新全局 softmax 信息 (Online Softmax)
                    # 这里的逻辑是融合旧的最大值和新块的最大值
                    m_curr = torch.max(m_prev[i : i + block_size], m_ij)
                    l_new = torch.exp(m_prev[i : i + block_size] - m_curr) * l_prev[i : i + block_size] + \
                            torch.exp(m_ij - m_curr) * l_ij
                    
                    # 5. 更新 O (公式：O_new = (O_old * alpha + O_new * beta) / l_new)
                    alpha = torch.exp(m_prev[i : i + block_size] - m_curr)
                    beta = torch.exp(m_ij - m_curr)
                    
                    o_curr = (alpha * o_prev[i : i + block_size] * l_prev[i : i + block_size] + \
                              beta * (P_ij @ vj)) / l_new
                    
                    # 状态回填
                    o_prev[i : i + block_size] = o_curr
                    m_prev[i : i + block_size] = m_curr
                    l_prev[i : i + block_size] = l_new
            
            output[b, h] = o_prev
            
    return output

if __name__ == "__main__":
    q = torch.randn(1, 1, 128, 64)
    k = torch.randn(1, 1, 128, 64)
    v = torch.randn(1, 1, 128, 64)
    
    # 模拟分块大小 32
    res = flash_attention_sim(q, k, v, block_size=32)
    print(f"Result shape: {res.shape}")
    
    # 验证常规 SDPA
    attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(64), dim=-1)
    ref = attn @ v
    
    diff = torch.abs(res - ref).max()
    print(f"Difference with standard attention: {diff}")
