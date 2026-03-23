# AI 算法面试：核心代码复习手册 (AI/LLM Interview Prep)

本仓库包含了大模型 (LLM)、搜索 (Search) 与推荐 (RecSys) 领域最核心、考频最高的算法实现。

每个算法均配有：
- **xx_ref.py**: 带有公式注释及详细面试考点说明的参考实现。
- **xx.py**: 空白练习模板，用于手写代码训练。

---

## 📁 目录结构

### 1. [模型架构 (Architectures)](./architectures/)
- **[Attention (注意力机制)](./architectures/attention/)**: MHA, SDPA, GQA, MLA, FlashAttention (Sim), KV Cache。
- **[FFN & Activations (前馈网络与激活)](./architectures/ffn_act/)**: Standard FFN, SwiGLU, ReLU, Sigmoid, SiLU, GELU。
- **[Positional Encoding (位置编码)](./architectures/pos_encoding/)**: RoPE (Rotary Positional Embedding)。
- **[Norms (归一化)](./architectures/norms/)**: RMSNorm。
- **[MoE (混合专家模型)](./architectures/moe/)**: MoE Layer & Router。
- **[Basic Layers (基础层)](./architectures/layers/)**: Embedding。

### 2. [搜索与推荐 (Search & RecSys)](./rec_models/)
- **DIN (Deep Interest Network)**: 用户行为序列注意力机制 (Target Attention)。
- **DeepFM**: 结合 FM (低阶特征交叉) 与 Deep (高阶特征提取) 的端到端模型。

### 3. [评价指标 (Metrics)](./metrics/)
- **AUC & GAUC**: 衡量全局与用户分组内的排序能力。
- **Classification (Hand-written)**: 手写 TP/FP/TN/FN 统计、Macro/Micro F1、基于秩 (Rank) 的 AUC 计算。
- **NDCG**: 考虑相关性分级与位置折扣因子的排序指标。
- **Sklearn Metrics 调用**: Accuracy, Precision, Recall, F1, MSE, R2 等标准 API 用例。

### 4. [优化器与损失 (Optimizers & Loss)](./optimizers/) 与 [损失函数](./loss/)
- **Adam / AdamW**: 偏差修正 (Bias Correction)、解耦后的权重衰减 (Weight Decay)。
- **InfoNCE Loss**: 标准 InfoNCE (带负样本)。
- **In-Batch InfoNCE**: CLIP/SimCLR 风格，利用 Batch 内其余样本作为负样本。
- **LLM Loss (Next Token Prediction)**: 包含 Shift Align (平移对齐) 与 Masking 处理。
- **Focal Loss / Triplet Loss / KL Div**: 处理不平衡样本、三元组距离学习及分布差异衡量。

### 5. [偏好对齐与强化学习 (RLHF/Alignment)](./rlhf/)
- **PPO & GRPO**: 组内相对奖励计算、裁剪损失 (Clipped Loss)、重要性采样。
- **DPO (Direct Preference Optimization)**: 消除 Reward Model，直接利用 Log-Ratio 进行偏好学习。

### 6. [参数高效微调 (PEFT)](./peft/)
- **LoRA**: 矩阵低秩分解 ($W = W_0 + BA$)、缩放因子 ($\alpha/r$)、参数初始化策略。

### 7. [经典算法 (Algorithms & Sorting)](./sort/)
- **Sorting (9 种排序)**: 快速排序、归并排序、堆排序、计数排序、基数排序等。
- **K-means**: 质心初始化、聚类分配与均值更新流程。

---

## 💡 如何使用
1. **先看 `xx_ref.py`**: 理解算法的数学公式、张量形状 (Shape) 变换以及面试常问的实现细节。
2. **手写练习 `xx.py`**: 在没有任何提示的情况下复现代码逻辑，确保掌握基本原理。

祝你面试通关，拿到心仪 Offer！🚀
