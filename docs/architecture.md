# TFDCL 架构详解

> 本文档描述模型的完整架构、训练流程和推理路径。CLAUDE.md 仅保留红线和索引。

---

## Pipeline 总览

```
数据加载 (datautils)
→ DCMR.encode_freq()/encode(concat_freq=False) → simple_kmeans_clustering() (伪标签)
→ DCMR.fit()
→ DCMR.encode(concat_freq=True) → 下游评估 (tasks)
```

---

## 训练循环（每 epoch）

### 0. 数据驱动 μ 初始化（仅 epoch 0）

`_init_mu_from_data()`：分批 rfft → 平均幅度谱 → `scipy.signal.find_peaks` 找 Top-K 谱峰 → 归一化到 [0, 1] → `LearnableBandpassDecomposition.init_mu_from_peaks()`。确保 K 个带通滤波器从数据真实谱峰起步，替代均匀 linspace 初始化。

### 1. 伪标签更新（每 20 epoch）

- 时域特征 `encode(return_global=True, concat_freq=False)` → **L2 归一化** → `simple_kmeans_clustering()` → `time_plabels` + `time_centroids`
- 频域特征 `encode_freq()` → **L2 归一化** → `simple_kmeans_clustering()` → `freq_plabels` + `freq_centroids`
- Auto K-Means++：k-means++ 初始化 + 轮廓系数自动选最优 k，子采样 (max 3000) 评估
- 聚类中心缓存为 GPU tensor 用作非参数化分类器

### 2. 频域分支前向 (FEI 频域掩码增强)

去均值 → `rfft` → 随机频域掩码 → 频域特征：

- **FEI 频域掩码** (`generate_frequency_mask`)：随机选择 k 个频率分量置零，k/F ~ Uniform(0.0, 0.7)
- **freq_net**：接收掩码后损坏频谱 + FEI 掩码提示
- **能量偏置注意力**：所选 token 的 log1p 幅值作为 pre-softmax bias 注入注意力 score
- **log1p 幅值编码**：`log1p(abs)` 替代 `log(abs+1e-8)`，零值严格映射到 0

→ `z_global_f [B, output_dims]`

### 3. 时域 Student 前向

`HeteroMoEStudentEncoder`（无掩码）：

- `LearnableBandpassDecomposition`：K 个可学习高斯带通滤波器 → PoU 归一化 → K 个子带信号
- subband_proj → 4 个共享专家各自处理所有 K 个子带 → expert_norms 归一化
- `SparseFreqAwareRouter`：子带频域特征 → MLP → Gumbel-Softmax Top-1 over (K, E+1) → one-hot 加权组合 → `component_norm` → B 个分量
- 跨分量注意力融合 → `z_t_concat [B, num_branches * D]`

### 4. 损失计算（三层级联合）

| 层级 | 损失函数 | 作用 |
|------|---------|------|
| 语义 | `complematch_cross_modal_ce_loss` | 非参数化跨模态伪标签 CE + 置信度过滤 |
| 物理 | `vmd_parametric_loss` | 峰值引导自适应带宽 + 重叠惩罚 + μ 边界约束 |
| 路由 | `sparse_load_balancing_loss` | Switch Transformer 风格，防止专家垄断 |
| 路由 | `sparse_diversity_loss` | Gram 矩阵非对角惩罚，鼓励分量差异化 |

总损失 = `1.0 * loss_comple + 1.0 * loss_vmd + λ_balance * loss_balance + λ_diversity * loss_diversity`

梯度裁剪 (max_norm=1.0)，AdamW 优化。每 epoch 末执行 Gumbel tau 退火 (仅 sparse 模式)。

### 5. 编码阶段（推理）

| 调用方式 | 输出 | 用途 |
|---------|------|------|
| `encode()` (默认 `concat_freq=True`) | `[N, B_br*D + D]` | 下游评估 |
| `encode(return_global=True)` | `[N, 2*D]` | 时频全局融合 |
| `encode(concat_freq=False, return_global=True)` | `[N, D]` | 训练内伪标签生成 |
| `encode_freq()` | `[N, D]` | 训练内频域伪标签 |

> **关键区分**：训练内伪标签用 `concat_freq=False`（仅时域），下游评估用默认 `concat_freq=True`（时频拼接）。

---

## 核心模块

### 全局频域编码器 (FreqGlobalEncoder)

掩码频谱的"频域视角"，输出全局频域表征，与完整时域表征进行跨模态对齐。

- **架构**：残差增强 → 动态位置编码 → FEI 掩码提示注入 → Top-K 频段选择 → [CLS] + 手动多头自注意力（能量偏置 pre-softmax）→ 投影头
- **能量偏置**：所选 token 的 log1p 幅值作为加性偏置 (`energy_bias_scale=0.1`) 注入注意力 score，引导 CLS 聚焦高能频段
- **输出**：`z_global_f [B, output_dims]`

### 异构 MoE 时域 Student 编码器 (HeteroMoEStudentEncoder)

VMD 频带分解替代时域掩码，频域感知路由替代时域统计路由。

```
x [B, T, C=1]
→ LearnableBandpassDecomposition → [B, K, T, C]
→ subband_proj → [B, K, T, H]
→ flatten [B*K, T, H]
→ 4 个共享专家 → 4 × [B*K, T, D]
→ reshape [B, E=4, K, T, D]
→ temporal_pool → [B, E, K, T_c, D]
→ SparseFreqAwareRouter → components [B, T_c, D] × B_br
→ cross_attention → z_t_concat [B, B_br * D]
```

**4 个固定异构专家**：
- Expert 0 — 空洞卷积 (DecompositionBlock)：局部多尺度感受野
- Expert 1 — FEB 频域增强 (FEBDecompositionBlock)：全局频域感受野
- Expert 2 — 多尺度卷积 (MSTConvDecompositionBlock)：深度可分离并行核宽 [5,15,31]
- Expert 3 — 紧凑 TS-Mixer (TSMixerExpert)：SE 风格时序门控 + channel mixing

### LearnableBandpassDecomposition

K 个可学习参数 (μ_k, σ_k) 高斯滤波器，在频域做 element-wise 乘法后 IFFT 回时域。

- **PoU 归一化**：`G_k / (Σ G_k + noise_floor)`，强制 Σ_k G_k(f) ≈ 1
- **数据驱动 μ 初始化**：`find_peaks` → `init_mu_from_peaks()`，从真实谱峰起步
- **σ 初始化**：softplus(-3.0) + eps ≈ 0.06，避免全通滤波器

### SparseFreqAwareRouter / FreqAwareRouter

- **Sparse (默认)**：Gumbel-Softmax Top-1 over (K, E+1)，含子带剪枝 (null slot)
- **Dense**：Joint Softmax over (K, E)，`--dense-routing` CLI 切换
- **频域特征驱动**：中心频率 ω_k、带内能量 log(E)、带宽 σ_k、频谱偏度 skew

### GatedAttnProjection

注意力加权池化 + 非线性投影，返回 `(z_proj [B, D], concentration [B])`。

---

## Auto K-Means++ 聚类

- KMeans with `init='k-means++'` + silhouette_score 自动选最优 k
- 候选 k 集：`[k_min, k_base//2, k_base, k_base*2, k_max]`，k_base = clamp(sqrt(N), 16, 256)
- 时域和频域各自独立聚类，每 20 epoch 重新聚类

---

## Checkpoint 格式

| Key | 对应模块 | 说明 |
|-----|---------|------|
| `time_net` | `self._net` (HeteroMoEStudentEncoder) | 时域编码器全部参数 |
| `freq_net` | `self.freq_net` (FreqGlobalEncoder) | 频域编码器参数 |
| `gated_proj_t` | `self.gated_proj_t` (ModuleList) | num_branches 个投影 |
| `num_branches` | int | 解耦分量数 |
| `num_subbands` | int | 子带数 K |
| `output_dims` | int | 表征维度 |
| `use_sparse_routing` | bool | 路由模式 |

`load()` 向后兼容旧 checkpoint：自动从 router 权重形状推断路由模式，必要时 `_rebuild_router()`。

---

## 下游评估

| 任务 | 评估函数 | 分类器 | 指标 |
|------|---------|--------|------|
| classification | `eval_classification3()` | SVM (RBF, C=100) | acc, auprc, f1, precision, recall |
| forecasting | `eval_forecasting()` | Ridge | — |
| anomaly | `eval_anomaly_detection()` | 阈值法 | — |

分类流程：`DCMR.encode(concat_freq=True)` → `[N, B_br*D + D]` → `StandardScaler` + `SVC`。
