# SAST v4：结构感知同步压缩变换

## 设计文档 (v4 更新)

> 2026-07-31 | 从 v3 演进至 v4：可微性修复、SupCon 自监督替代分类、per-region 选择性熵、物理约束重构

---

## 目录

- [1. 概述与 v3->v4 变更总览](#1-概述与-v3-v4-变更总览)
- [2. 核心数据流](#2-核心数据流)
- [3. 三大职责分离](#3-三大职责分离)
- [4. 静态原型软匹配（固定不学习）](#4-静态原型软匹配固定不学习)
- [5. 异构物理图 + GAT](#5-异构物理图--gat)
- [6. 可微稀疏高斯重排（v4 关键修复）](#6-可微稀疏高斯重排v4-关键修复)
- [7. 损失函数（v4 重构）](#7-损失函数v4-重构)
- [8. 可微性分析](#8-可微性分析)
- [9. 端到端训练与评估](#9-端到端训练与评估)
- [10. 显存与性能评估](#10-显存与性能评估)
- [11. v3->v4 改动对照表](#11-v3-v4-改动对照表)

---

## 1. 概述与 v3->v4 变更总览

SAST (Structure-Aware Synchrosqueezing Transform) 是面向水泵水轮机振动信号的时频分析方法，核心创新在于**用物理图自洽性驱动挤压策略的自适应选择**——对谐波分量（2xBPF）硬挤、对调制分量（BPF）适度挤压保留调制结构、对宽带水力分量（LOW_FREQ）不挤压保留本征带宽。

v4 在 v3 的基础上做了以下关键改动：

| 改动项 | v3 | v4 | 动机 |
|--------|----|----|------|
| 可微重排 | `argmin` 量化 sigma 到离散 level，梯度断裂 | 连续高斯核 `w=exp(-0.5*(k/sigma)^2)/Z`，sigma 可微 | 修复 `tfr_enhanced` 对 `sigma_sq` 不可微，`L_task`/`RE_2D` 梯度到不了 GAT |
| 主监督 | `L_task` (分类 CE) | `L_supcon` (监督对比，Khosla 2020) | 同工况拉近、不同工况推远，用工况标签定义正负对但不分类，梯度经 TFR 回流 GAT |
| 频域编码器 | `TFRClassifier` (BatchNorm) | `FreqEncoder` (LayerNorm) | 小 batch 时 BN 统计不稳定 |
| 评估方式 | 分类头 softmax | KNN (z_freq 到工况中心最近邻) | 无分类头，纯表示学习评估 |
| 合成预训练 | 方案A/B 曾实现 | 已删除 | 方案A 信号重建损失平凡（梯度~1e-7），方案B 被 SupCon 替代 |
| Rényi 熵 | 全局 RE_2D | per-region 选择性（HYDRAULIC 不惩罚） | 修复全局熵"鼓励所有能量集中"与选择性挤压的冲突 |
| 物理约束 | 约束 `w_i`，`gate_edge × ratio_dev` 互消致~0 | 约束 `A_ij`（GAT 注意力），按边类型差异化 ℓ | 修复 v3 物理约束失效（~0），GAT 通过 A_ij 收到梯度 |
| gate_node | inplace `gate_node[:,s]=max(...)` | `scatter_reduce(amax)` | autograd 安全 |
| 总损失 | 4 项 | 5 项：SupCon + RE_2D + L_physics + L_smooth + L_balance | SupCon 主监督，其余辅 |
| 工况匹配 | 可学习参数 | 固定不学习（alpha 无可学习参数） | 职责分离：原型负责工况识别（固定），GAT 负责挤压策略（学习） |

---

## 2. 核心数据流

```
信号 x[t] (T=2000, fs=1000 Hz)
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  MSST (N_max=5, save_trajectory)                         │
│  output: STFT [F,T], omega_final [F,T], omegas [N,F,T]  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  MSSTNodeExtractor — 按 3 频段聚合                       │
│  每物理节点: IF(t), energy(t), bandwidth(t), persistence │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  V_obs 计算 + StaticPrototypeMatcher（固定，不学习）       │
│  V_obs(t) = [R_LF, R_BPF, R_2xBPF, log10(Eb/E2x)]       │
│  alpha = softmax(cosine(V_obs, frozen_prototypes) / τ)   │
│  cond_ctx(t) = Σ α_k × P_embed[k]                       │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  异构物理图 + PPM + EdgeConditionedGAT                    │
│  4 节点 (OP 虚拟 + 3 物理), 10 边, 4 边类型               │
│  PPM: 类型嵌入 + 交叉注意力 + 门控融合                     │
│  GAT: 2 层 × 4 头, edge-conditioned attention            │
│  output: w_i [3, T] (IF 信任度), A_ij [10, H, T]         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  sigma_i = sigma_min + (1 - w_i) × Δσ                    │
│  broadcast to per-bin: sigma_sq [B, F, T]                │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  SparseGaussianReassigner (连续可微高斯核)                │
│  w_k = exp(-0.5*(k/sigma_sq)^2) / Z                     │
│  tfr_enhanced = Σ_k w_k * STFT[omega_hat + k]           │
│  → 输出 TFR_sast [B, F, T] (对 sigma_sq 可微)            │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  FreqEncoder: TFR_sast → z_freq [B, D] (L2 归一化)      │
│  GlobalAvgPool(T) → log1p → MLP(LayerNorm) → normalize   │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   L_supcon(z_freq)  RE_2D(TFR_sast)  L_physics(A_ij)
   (监督对比)         (per-region)     (注意力约束)
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 L_total = λ_sc·L_supcon + λ_e·RE_2D
                         + λ_p·L_physics + λ_s·L_smooth
                         + λ_b·L_balance
```

---

## 3. 三大职责分离

SAST 将传统 SST 的"IF 估计 + 挤压"两步拆分为三大职责，分别由不同模块承担：

```
┌──────────────────────────────────────────────────────────────────┐
│  职责                    │ 模块              │ 可学习？│ 输出      │
├──────────────────────────────────────────────────────────────────┤
│  1. IF 估计 (HMST定位)   │ MSST              │  否     │ omega_5   │
│     "能量在哪个频率？"    │ 所有 bin 统一用     │         │           │
│                          │ 最优 IF (lookup 5次)│         │           │
├──────────────────────────────────────────────────────────────────┤
│  2. 策略决策 (Graph决策) │ PPM + GAT          │  是     │ w_i, A_ij │
│     "该不该挤？挤多宽？"  │ 物理图自洽性 →       │         │           │
│                          │ 信任度 w_i ∈ (0,1)  │         │           │
├──────────────────────────────────────────────────────────────────┤
│  3. 挤压执行 (Squeeze)   │ SparseGaussian       │  否     │ TFR_sast  │
│     "怎么挤？"            │ Reassigner          │  (但sigma│           │
│                          │ 连续高斯核, sigma 可微│  可微)   │           │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计原则**：
- MSST 的 lookup 迭代对所有 bin 无害（总是朝能量集中方向走），因此所有 bin 统一使用 omega_5（5 次 lookup 的最优 IF），无需为不同 bin 选择不同 IF 阶数。
- GAT 不输出"频率是多少"，只输出"此刻的 IF 有多可信"（w_i）和"边注意力如何分配"（A_ij）。
- 挤压核宽 sigma 连续依赖 w_i，梯度通过 SparseGaussianReassigner 回流到 GAT。

---

## 4. 静态原型软匹配（固定不学习）

### 4.1 设计原则

**工况匹配与挤压策略决策完全分离**：

| 模块 | 职责 | 可学习？ |
|------|------|:---:|
| StaticPrototypeMatcher | 工况识别：当前信号能量分布最像哪种典型工况？ | **否**（原型 frozen，alpha 无可学习参数） |
| PPM + GAT | 挤压策略：根据物理图自洽性决定 w_i | **是**（P_embed 可学习，但 alpha 不参与） |

工况标签**不用于教模型识别工况**，而用于 SupCon 定义正负对（同工况 TFR 表示拉近）。

### 4.2 匹配流程

```
离线 EDA → 5 个 frozen prototypes:
  V_proto_k = [R_LF, R_BPF, R_2xBPF, log10(E_BPF/E_2xBPF)]

在线推理:
  V_obs(t) = 从 node_energy 实时计算 4D 向量
  sim(t, k) = cosine(V_obs(t), prototype_k)    k=0..4
  alpha(t) = softmax(sim / τ)                  τ=0.1
  cond_ctx(t) = Σ_k alpha_k(t) × P_embed[k]   ← P_embed 可学习

  alpha 只用于加权混合 P_embed，不参与分类决策。
  obs_proj 是 dead code（alpha 从 cosine 直接计算，无可学习参数）。
```

### 4.3 五类原型

| 原型 | 工况 | V_proto = [R_LF, R_BPF, R_2xBPF, log10(Eb/E2x)] |
|------|------|------|
| 0 | 空转 | [0.038, 0.209, 0.027, +0.99] |
| 1 | 低负荷 | [0.068, 0.833, 0.010, +1.99] |
| 2 | 中负荷 | [0.046, 0.270, 0.304, +0.00] |
| 3 | 高负荷 | [0.013, 0.153, 0.652, -0.65] |
| 4 | 抽水 | [0.021, 0.686, 0.099, +0.85] |

---

## 5. 异构物理图 + GAT

### 5.1 图拓扑（4 节点，10 边）

```
                ┌─────────────────────┐
                │   OP (virtual)       │  工况上下文嵌入
                └──┬──┬──┬────────────┘
          CONDITION│  │  │  CONDITION
          (w=0.6)  │  │  │  (w=0.6)
                   ▼  ▼  ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │ LOW_FREQ │ │   BPF    │ │  2xBPF   │
      │  (idx=1) │ │  (idx=2) │ │  (idx=3) │
      │ 2-25 Hz  │ │ 42-55 Hz │ │ 90-105 Hz│
      └──┬───┬───┘ └──┬───┬───┘ └──┬───┬───┘
         │   │        │   │        ▲   ▲
         │   │ DRIFT  │   └────────┘   │
         │   │(w=0.15)│   HARMONIC     │
         │   │        │   (r=2.0,w=0.8)│
         │   │        │                │
         │   └────────┼────────────────┘
         │  COMPETITION (w=0.25)  LOW_FREQ<->2xBPF
         └────────────┼────────────────┘
           COMPETITION (w=0.20)  BPF<->2xBPF
```

### 5.2 完整边表

| 边 | 类型 | 物理含义 | w | 边特征 dim0 语义 |
|----|------|---------|:---:|------|
| OP -> LOW_FREQ | CONDITION | 工况信息广播 | 0.6 | cond_sim |
| OP -> BPF | CONDITION | 工况信息广播 | 0.6 | cond_sim |
| OP -> 2xBPF | CONDITION | 工况信息广播 | 0.6 | cond_sim |
| LOW_FREQ -> BPF | DRIFT | 涡带能量 -> BPF 调制 | 0.15 | Corr_E |
| BPF -> LOW_FREQ | DRIFT | 调制反馈 | 0.15 | Corr_E |
| BPF -> 2xBPF | HARMONIC | 确定性倍频 r=2.0 | 0.8 | r_obs |
| LOW_FREQ -> 2xBPF | COMPETITION | 涡带 vs RSI 此消彼长 | 0.25 | -Corr_E |
| 2xBPF -> LOW_FREQ | COMPETITION | 反向 | 0.25 | -Corr_E |
| BPF -> 2xBPF | COMPETITION | BPF vs RSI 此消彼长 | 0.20 | -Corr_E |
| 2xBPF -> BPF | COMPETITION | 反向 | 0.20 | -Corr_E |

### 5.3 GAT 输出

**w_i**：每物理节点每帧的 IF 信任度，`w_i = sigmoid(MLP(h_phys)) ∈ (0,1)`：

```
w_i → 1 (2xBPF):  σ → σ_min (0.5 bin)  — 窄核，能量高度集中
w_i → 0.7 (BPF):  σ → 中 (4 bin)       — 适度保留调制结构
w_i → 0 (LOW_FREQ): σ → σ_max (15 bin)  — 宽核，几乎不改变原始能量分布
```

**A_ij**：每边每头的 GAT 注意力权重，`A_ij ∈ [0,1]`，由 L_physics 约束（见 §7.4）。

---

## 6. 可微稀疏高斯重排（v4 关键修复）

### 6.1 v3 问题：argmin 量化导致梯度断裂

v3 的实现中，`sigma_sq`（逐 bin 核宽）先被量化到离散 sigma level：

```python
# v3 旧实现（已废弃）
level_idx = argmin(|sigma_sq - sigma_levels|)  # 离散，不可微
w_k = precomputed_weights[level_idx]            # 查表
tfr_enhanced = scatter_add(w_k * STFT, target)  # requires_grad=False
```

`argmin` 操作不可微，导致 `tfr_enhanced` 对 `sigma_sq` 的梯度恒为 0。`tfr_enhanced.requires_grad` 为 `False`，意味着 `L_task`（当时是分类 CE）和 `RE_2D` 的梯度都无法通过 TFR 回流到 `sigma_sq` → `w_i` → GAT。**w_i 实际上处于无监督状态**——GAT 的参数更新完全依赖 L_physics 和 L_smooth/L_balance 等正则项，没有来自 TFR 质量的梯度信号。

### 6.2 v4 修复：连续可微高斯核

```python
# v4 新实现：sigma_sq 直接进入高斯权重（连续，可微）
sigma = sigma_sq.clamp(sigma_min, sigma_max)   # [B, F, T]

# Pass 1: 计算归一化常数 Z
Z = Σ_k exp(-0.5 * (k/sigma)^2)               # [B, F, T]

# Pass 2: scatter 分配
for k in offsets:
    w_k = exp(-0.5 * (k/sigma)^2) / Z          # [B, F, T] 可微
    tfr_enhanced.scatter_add(target, w_k * tfr_weighted)
```

**关键设计决策**：

1. **连续 sigma 直接进权重**：`w_k = exp(-0.5*(k/sigma)^2) / Z`，sigma 作为连续变量参与高斯核计算，梯度经链式法则自然回流。

2. **归一化核**：`Σ_k w_k = 1`，保证频率求和守恒（每帧总能量不变），避免"重建损失"方案中损失平凡的问题（SST 归一化软核频率求和守恒，重建损失的梯度仅 ~1e-7）。

3. **两遍循环避免 OOM**：旧方案若预计算 `[2K+1, B, F, T]` 大张量（K=45 时约 91×B×F×T 个 float32），在 `batch=4, F=1000, T=2000` 时内存超 2.7 GB。两遍循环（Pass1 算 Z，Pass2 scatter）内存开销仅 `[B, F, T]` 的 3 倍（Z + sigma + tfr_enhanced），约 24 MB/batch。

4. **目标位置 detach**：`omega_hat` 来自 MSST（离散常量），不携带梯度。这符合设计原则——IF 估计本身不参与梯度，梯度只流向"怎么挤"（sigma_sq → w_i → GAT）。

### 6.3 可微性验证

修复后，`tfr_enhanced.requires_grad = True`，TFR loss 能回流到 w_i：

```
修复前: tfr_enhanced.requires_grad = False
        w_i.grad = 0  (来自 TFR 相关 loss)

修复后: tfr_enhanced.requires_grad = True
        w_i.grad ≈ 9.7e-4  (来自 SupCon + RE_2D，经 TFR → sigma → w_i)
```

### 6.4 gate_node 修复

v3 中使用 inplace 操作 `gate_node[:, s] = max(...)` 修改 tensor，autograd 不安全。v4 改为 `scatter_reduce(amax)`：

```python
# v4 新实现
gate_node = gate_node.scatter_reduce(1, src_idx, gate_edge, reduce='amax')
gate_node = gate_node.scatter_reduce(1, dst_idx, gate_edge, reduce='amax')
```

---

## 7. 损失函数（v4 重构）

### 7.1 总损失公式

$$L_{total} = \lambda_{sc} \cdot L_{supcon} + \lambda_e \cdot RE_{2D} + \lambda_p \cdot L_{physics} + \lambda_s \cdot L_{smooth} + \lambda_b \cdot L_{balance}$$

| 项 | 名称 | 监督来源 | 需要标签？ | λ |
|----|------|:---:|:---:|:---:|
| L_supcon | 监督对比损失 | 工况标签（定义正负对，不分类） | 是（仅正负对） | 1.0 |
| RE_2D | per-region 选择性 Rényi 熵 | TFR 自身集中度 | 否 | 0.1 |
| L_physics | 注意力物理一致性 | 叶片数 Z=9 等机械设计参数 | 否 | 0.5 |
| L_smooth | w_i 帧间差分 + A_ij 帧间差分 | 正则化 | 否 | 0.05 |
| L_balance | w_mean 区间约束 [0.3, 0.8] | 正则化 | 否 | 0.01 |

### 7.2 L_supcon — 监督对比损失（替代 L_task）

**来源**：Khosla et al. (2020) "Supervised Contrastive Learning"

**设计思路**：
- 同工况样本的 TFR 表示 z_freq 拉近，不同工况推远
- 用工况标签定义正负对，但**不分类**（无分类头，无 softmax）
- 梯度经 `z_freq → FreqEncoder → TFR_sast → sigma_sq → w_i → GAT` 监督 GAT
- 评估改用 KNN（z_freq 到各类工况中心最近邻）

**公式**：

$$L_{supcon} = -\frac{1}{|P(i)|} \sum_{i} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}$$

其中 $P(i)$ 是 anchor i 的同工况正样本集合，$\tau = 0.1$ 是温度。

**FreqEncoder**（替代 TFRClassifier）：

```
TFR_sast [B, F, T] → GlobalAvgPool(T) → [B, F]
→ log1p → Linear(F, 128) → LayerNorm → GELU → Dropout
→ Linear(128, 64) → LayerNorm → GELU → Dropout
→ Linear(64, 128) → L2 normalize → z_freq [B, 128]
```

用 LayerNorm 替代 BatchNorm——小 batch 时 BN 统计不稳定，LayerNorm 按样本归一化不受 batch size 影响。

**评估**：KNN 最近邻到工况中心：

```python
centroids[c] = mean(normalize(z_freq)) for samples of class c  # 训练集
pred = argmax(cosine_similarity(z_freq_test, centroids))      # 最近邻
```

### 7.3 RE_2D — per-region 选择性 Rényi 熵（修复全局熵问题）

**v3 问题**：全局 Rényi 熵鼓励所有频段的能量集中。但 LOW_FREQ（涡带，本征带宽 10-15 Hz）**应该保留展宽**——全局熵会惩罚 LOW_FREQ 的展宽，与选择性挤压的设计目标冲突。

**v4 修复**：仅对需要挤压的频段计算熵：

$$RE_{2D}(TFR) = \frac{1}{1-\alpha} \log \sum_{m \in \mathcal{R}_{squeeze}} \sum_{n} \left( \frac{|TFR[m,n]|}{\sum |TFR|} \right)^\alpha$$

其中 $\mathcal{R}_{squeeze}$ = {BPF, 2xBPF}（BLADE_PASS 和 BLADE_HARMONIC 类型），**HYDRAULIC（LOW_FREQ）被排除**。

```python
for region in regions:
    if region.f_type == 'HYDRAULIC':
        continue  # 不惩罚，保留展宽
    # 计算该频段的 Rényi 熵
```

### 7.4 L_physics — 注意力物理一致性（约束 A_ij，非 w_i）

**v3 问题**：L_physics 约束 `w_i × gate_edge × ratio_dev`，但 gate_edge 和 ratio_dev 在 HARMONIC 边上相互抵消（gate 高时 ratio_dev 低，反之亦然），导致 L_physics ≈ 0，无有效梯度信号。

**v4 修复**：约束对象改为 `A_ij`（GAT 注意力权重），按边类型差异化偏差 ℓ：

$$L_{physics} = \frac{\sum_{edges} w_{type} \cdot A_{ij} \cdot \ell_{ij}}{\sum A_{ij}}$$

| 边类型 | ℓ（偏差度量） | 物理含义 |
|--------|------|------|
| HARMONIC | $|r_{obs} - r_{nom}| / r_{nom}$ | 比值须等于标称值（如 2.0） |
| CONDITION | $1 - cond_{sim}$ | 上下文匹配应高 |
| DRIFT | $1 - Corr_E$ | 能量共变应正相关 |
| COMPETITION | $1 - (-Corr_E)$ | 此消彼长应负相关（dim0 = -Corr_E，大表示负相关强） |

**关键效果**：
- 高注意力 + 物理不一致 → 大惩罚 → 推动 GAT 学会利用边特征分配注意力
- 低注意力 + 物理不一致 → 小惩罚（A_ij 小 → 该项自动衰减）
- v4 修复后 L_physics ≈ 0.31（非零），GAT 通过 A_ij 收到有效梯度

### 7.5 L_smooth — 时序平滑

$$L_{smooth} = \frac{1}{B \cdot N \cdot (T-1)} \sum (w_i[:,:,t+1] - w_i[:,:,t])^2 + \lambda_A \cdot \frac{1}{B \cdot M \cdot H \cdot (T-1)} \sum (A_{ij}[:,:,:,t+1] - A_{ij}[:,:,:,t])^2$$

同时约束 w_i 和 A_ij 的时序平滑性（$\lambda_A = 0.1$）。

### 7.6 L_balance — 防退化

$$L_{balance} = \max(0, w_{min} - \bar{w}) + \max(0, \bar{w} - w_{max})$$

$w_{min}=0.3, w_{max}=0.8$。防止所有 w_i 退化到 0（什么都不挤）或 1（什么都挤）。

---

## 8. 可微性分析

### 8.1 梯度路径图

```
L_total
  ├── L_supcon
  │     └── z_freq ─── FreqEncoder ─── TFR_sast ─── sigma_sq ─── w_i ─── GAT
  │           [可微]        [可微]        [可微]      [可微]     [可微]
  │
  ├── RE_2D
  │     └── TFR_sast ─── sigma_sq ─── w_i ─── GAT
  │           [可微]        [可微]     [可微]
  │
  ├── L_physics
  │     └── A_ij ─── GAT
  │         [可微]
  │
  ├── L_smooth
  │     └── w_i ─── GAT
  │         [可微]
  │
  └── L_balance
        └── w_i ─── GAT
            [可微]
```

### 8.2 不可微的环节（设计决策）

| 环节 | 不可微原因 | 替代方案 |
|------|------|------|
| MSST IF 估计 (omega_5) | 离散 bin 索引 + argmax/round | 作为常量输入，不参与梯度（IF 本身不应被优化） |
| scatter_add 的目标索引 | 离散整数索引 | 梯度只沿 value 传播，不沿 index 传播（符合设计） |
| 挤压轮数 lambda_i | round(w_i × N_max) 离散 | 训练时固定 lambda=1，推理时启用；w_i 的排序直接继承 |
| 静态原型匹配 | cosine 相似度计算不可学习 | 设计上固定不学习，职责分离 |

### 8.3 梯度验证

| 检查项 | v3 状态 | v4 状态 |
|--------|:---:|:---:|
| tfr_enhanced.requires_grad | False | **True** |
| w_i 从 TFR loss 收到梯度 | 0 | **~9.7e-4** |
| GAT 参数从 SupCon 收到梯度 | 无此路径 | **有**（经 TFR → sigma → w_i） |
| GAT 参数从 L_physics 收到梯度 | ~0（gate×ratio 互消） | **非零**（~0.31, 经 A_ij） |

---

## 9. 端到端训练与评估

### 9.1 训练流程

```
python train_sast.py --batch_size 8 --epochs 20 --device cuda

Input: signal [T] + 工况标签 y (仅用于 SupCon 正负对定义，不用于分类)
"""

for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        # 1. Forward
        results = model(batch_x)                    # SAST full pipeline
        tfr_sast = results['tfr_enhanced']
        w_i = results['w_i']
        A_ij = results['A_ij']

        # 2. FreqEncoder
        z_freq = freq_encoder(tfr_sast)            # [B, 128]

        # 3. 5-component loss
        L_supcon = supcon_loss(z_freq, batch_y)    # 监督对比
        RE_2D = renyi_2d_loss(tfr_sast, freqs)     # per-region 选择性
        L_physics = physics_loss(A_ij, edge_feats) # 注意力物理约束
        L_smooth = smoothness_loss(w_i, A_ij)      # 时序平滑
        L_balance = balance_loss(w_i)              # 防退化

        L_total = λ_sc·L_supcon + λ_e·RE_2D + λ_p·L_physics
                + λ_s·L_smooth + λ_b·L_balance

        # 4. Backward — 梯度经 TFR → sigma → w_i → GAT
        L_total.backward()

    # 5. Validation (KNN via z_freq centroids)
    centroids = compute_class_centroids(model, freq_encoder, X_train, y_train)
    val_acc = evaluate_accuracy(model, freq_encoder, X_val, y_val, centroids)
```

### 9.2 关键训练参数

| 参数 | 推荐值 | 说明 |
|------|:---:|------|
| batch_size | >= 8 | SupCon 需 batch 内同工况正样本对 |
| lr | 0.001 | AdamW optimizer |
| λ_sc | 1.0 | SupCon 主监督 |
| λ_e | 0.1 | RE_2D 辅 |
| λ_p | 0.5 | L_physics 辅 |
| λ_s | 0.05 | 时序平滑 |
| λ_b | 0.01 | 防退化 |
| supcon_temperature | 0.1 | 越小越聚焦硬正样本 |
| grad_clip | 1.0 | 梯度裁剪 |

### 9.3 评估：KNN 最近邻到工况中心

```python
# 训练后：计算各类 z_freq 中心
centroids = compute_class_centroids(model, freq_encoder, X_train, y_train)
# centroids: [5, 128] L2-normalized

# 验证/测试：最近邻分类
def predict(x):
    z = freq_encoder(model(x)['tfr_enhanced'])
    return argmax(cosine_similarity(z, centroids))

accuracy = evaluate_accuracy(model, freq_encoder, X_val, y_val, centroids)
```

无分类头——纯表示学习评估，验证 z_freq 是否学到了工况判别性的 TFR 表示。

---

## 10. 显存与性能评估

### 10.1 显存分析（max_len=2000, F=1000, T=2000）

| 张量 | 约大小 | 说明 |
|------|:---:|------|
| MSST 中间量 (omegas) | 24 MB | 5 层 × F × T × 4 bytes |
| tfr_stft | 16 MB | F × T × 4 bytes |
| 节点特征 (node_feats) | < 1 MB | 3 × T × 4 |
| GAT 中间激活 | ~10 MB | 4 节点 × 128 dim × 4 头 |
| SparseGaussianReassigner | ~24 MB | 两遍循环，仅 [B,F,T] 的 3 倍 |
| **per-sample 总计** | **~320 MB** | 线性增长 |
| **batch=8 总显存** | **~2.6 GB** | 8GB GPU 建议 batch ≤ 16 |

### 10.2 内存优化

- **两遍循环**：旧方案 `[2K+1, B, F, T]` 大张量（K=45 时约 91×B×F×T）曾导致 batch=4 OOM。两遍循环仅需 `[B, F, T]` 的 3 倍内存。
- **MSST 在 GPU 上运行**：CUDA 加速全流程（STFT + IF + 挤压），115x vs numpy。
- **FreqEncoder 轻量**：仅 128 维输出，~0.1M 参数。

---

## 11. v3->v4 改动对照表

| 维度 | v3 | v4 |
|------|----|----|
| **可微重排** | argmin 量化 sigma，梯度断裂 | 连续高斯核，sigma 可微 |
| **主监督** | L_task (分类 CE) | L_supcon (监督对比) |
| **频域编码器** | TFRClassifier (BN) | FreqEncoder (LN) |
| **评估** | 分类头 softmax | KNN 到类中心 |
| **预训练** | 曾实现合成信号预训练 | 已删除 |
| **Rényi 熵** | 全局 | per-region 选择性（HYDRAULIC 排除） |
| **物理约束对象** | w_i（互消致 ~0） | A_ij（有效梯度，~0.31） |
| **物理约束按边类型** | 统一 ratio_dev | 按边类型差异化 ℓ |
| **gate_node** | inplace max | scatter_reduce(amax) |
| **总损失项数** | 4 项 | 5 项（+SupCon） |
| **工况匹配** | 可学习参数 | 固定不学习 |
| **梯度到 GAT 的路径** | 仅 L_physics(~0) + 正则项 | SupCon(TFR→σ→w_i) + L_physics(A_ij) + 正则项 |

---

## 参考文献

1. Khosla, P., et al. "Supervised Contrastive Learning." NeurIPS, 2020.
2. Colominas, M.A., Meignen, S. "Adaptive Order Synchrosqueezing Transform." 2025.
3. Yu, G., Wang, Z., Zhao, P. "Multi-Synchrosqueezing Transform." IEEE TIE, 2019.
4. Oberlin, T., Meignen, S. "Second-Order Synchrosqueezing Transform." IEEE TSP, 2015.
5. Pham, D-H., Meignen, S. "High-Order Synchrosqueezing Transform." IEEE TSP, 2017.

---

*本文档描述 SAST v4 架构，于 2026-07-31 从 v3 演进。v3 设计文档见 `SAST_完整设计文档.md`，讲解文档见 `SAST_讲解文档.md`。*