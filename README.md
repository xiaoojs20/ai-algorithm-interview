# 大模型算法工程师面试：核心代码复习手册 (LLM Interview Prep)

本仓库包含了大模型 (LLM) 领域最核心、考频最高的算法实现。每个算法均配有 LaTeX 公式注释及详细的面试考点说明。

## 📁 目录结构

### 1. [模型架构 (Architectures)](./architectures/)
- **MHA & SDPA**: 注意力机制核心公式、Mask 逻辑、多头切分。
- **RoPE (Rotary Positional Embedding)**: 旋转矩阵应用、复数域旋转实数化、相对位置编码。
- **MoE (Mixture of Experts)**: Router Gating 逻辑、Top-K 专家分发、负载均衡考点。
- **RMSNorm**: 相比 LayerNorm 的简化点、方差计算逻辑。
- **SwiGLU**: 门控线性单元、SiLU 激活函数。

### 2. [偏好对齐与强化学习 (RLHF/Alignment)](./rlhf/)
- **PPO & GRPO (DeepSeek Core)**: 组内相对奖励计算、裁剪损失 (Clipped Loss)、重要性采样比率。
- **DPO (Direct Preference Optimization)**: Log-Ratio 计算、直接利用 Logps 计算 Loss、消除 Reward Model 的优势。
- **GAE (Generalized Advantage Estimation)**: 优势估计的偏差与方差平衡。

### 3. [训练、优化器与损失 (Optimizers & Loss)](./optimizers_loss/)
- **Adam / AdamW**: 偏差修正 (Bias Correction)、解耦后的 Weight Decay (AdamW)。
- **SGD**: 动力系数 (Momentum) 的物理意义。
- **Stable Cross Entropy**: 数值稳定性 (Log-Sum-Exp Trick)。

### 4. [参数高效微调 (PEFT)](./peft/)
- **LoRA**: 矩阵低秩分解逻辑、初始化策略 ($B=0, A \sim \mathcal{N}$)、Scaling Factor ($\alpha/r$)。

### 5. [推理策略 (Decoding)](./decoding/)
- **Beam Search**: Top-K 维持候选、Log-Score 累加、搜索逻辑。

---

## 💡 如何使用
1. 先看 `xx_ref.py` (带有公式和注释的参考代码)，理解算法的每一行物理含义。
2. 打开 `xx.py` (空白模板)，尝试在不看参考代码的情况下复现实现细节。
3. **重点关注**：数值稳定性、张量形状 (Shape) 变换、以及特殊初始化 (如 LoRA B 矩阵)。

祝你面试通关，拿到心仪 Offer！
