# AI 算法面试：核心代码复习手册 (AI/LLM Interview Prep)

本仓库包含了大模型 (LLM)、搜索 (Search) 与推荐 (RecSys) 领域最核心、考频最高的算法实现。

每个算法均配有：
- **xx_ref.py**: 带有公式注释及详细面试考点说明的参考实现。
- **xx.py**: 空白练习模板，用于手写代码训练。

---

## 📁 目录结构

### 1. [模型架构 (Architectures)](./architectures/)
- **MHA & SDPA**: 注意力机制核心公式、Mask 逻辑、多头切分。
- **GQA (Grouped Query Attention)**: KV 共享组、推理效率优化。
- **MLA (Multi-head Latent Attention - DeepSeek)**: KV 压缩潜变量、RoPE 解耦合、KV Cache 极限优化。
- **RoPE (Rotary Positional Embedding)**: 旋转矩阵应用、复数域旋转实数化、相对位置编码。
- **MoE (Mixture of Experts)**: Router Gating 逻辑、Top-K 专家分发、负载均衡考点。
- **RMSNorm**: 相比 LayerNorm 的简化点、方差计算逻辑。
- **FlashAttention (Simulated)**: Tiling 分块计算原理、Online Softmax 动态更新。
- **KV Cache**: 自回归推理时的增量生成逻辑。
- **Embedding**: 可学习查找表 (Look-up Table) 的本质与调用方式。

### 2. [搜索与推荐 (Search & RecSys)](./rec_models/)
- **DIN (Deep Interest Network)**: 用户行为序列注意力机制 (Target Attention)。
- **DeepFM**: 结合 FM (低阶特征交叉) 与 Deep (高阶特征提取) 的端到端模型。

### 3. [评价指标 (Metrics)](./metrics/)
- **AUC & GAUC**: 衡量全局与用户分组内的排序能力。
- **NDCG**: 考虑相关性分级与位置折扣因子的排序指标。
- **Sklearn Metrics 调用**: Accuracy, Precision, Recall, F1, MSE, R2 等标准 API 用例。

### 4. [优化器与损失 (Optimizers & Loss)](./optimizers/) 与 [损失函数](./loss/)
- **Adam / AdamW**: 偏差修正 (Bias Correction)、解耦后的权重衰减 (Weight Decay)。
- **InfoNCE (InfoNCE Loss)**: 对比学习 (Contrastive Learning) 中的正负样本相似度判别。
- **LLM Loss (Next Token Prediction)**: 包含 Shift Align (平移对齐) 与 Masking 处理。
- **Focal Loss**: 解决类别不平衡，通过 $(1-p_t)^\gamma$ 对易分类样本降权。
- **Triplet Loss**: 三元组损失，使 Anchor 离正样本比离负样本更近。
- **KL Divergence**: 衡量概率分布差异，常用于 VAE 和 知识蒸馏。
- **SGD / CrossEntropy**: 基础随机梯度下降与交叉熵实现。

### 5. [偏好对齐与强化学习 (RLHF/Alignment)](./rlhf/)
- **PPO & GRPO**: 组内相对奖励计算、裁剪损失 (Clipped Loss)、重要性采样。
- **DPO (Direct Preference Optimization)**: 消除 Reward Model，直接利用 Log-Ratio 进行偏好学习。
- **GAE (Generalized Advantage Estimation)**: 优势估计的偏差与方差平衡。

### 6. [参数高效微调 (PEFT)](./peft/)
- **LoRA**: 矩阵低秩分解 ($W = W_0 + BA$)、缩放因子 ($\alpha/r$)、参数初始化策略。

### 7. [经典算法 (Algorithms & Sorting)](./sort/)
- **Sorting (9 种排序)**: 快速排序、归并排序、堆排序、希尔排序、计数排序、基数排序等。
- **K-means**: 质心初始化、聚类分配与均值更新流程。

### 8. [推理解码 (Decoding)](./decoding/)
- **Beam Search**: Top-K 维持候选、搜索权重评分逻辑。

---

## 💡 如何使用
1. **先看 `xx_ref.py`**: 理解算法的数学公式、张量形状 (Shape) 变换以及面试常问的实现细节。
2. **手写练习 `xx.py`**: 在没有任何提示的情况下复现代码逻辑，确保掌握边界条件（如 Mask 处理或数值稳定性 Trick）。
3. **重点突破**: 关注模型推理效率（KV Cache/FlashAttention）以及不平衡数据集处理（GAUC/F1）。

祝你面试通关，拿到心仪 Offer！🚀
