# SAST-MSST：结构感知多重同步压缩变换

## —— 设计原理、完整流程与可解释性分析

> 2026-07-26 | 整合自 SAST_MSST_必要性分析.md + SAST_完整流程说明.md

---

## 目录

**Part I — 为什么需要 SAST**
- [1. WSST2 vs MSST：两种 SST 基底的本质差异](#1-wsst2-vs-msst两种-sst-基底的本质差异)
- [2. MSST 多重挤压的本质与局限](#2-msst-多重挤压的本质与局限)
- [3. SAST 核心思路：物理先验引导的信任分配](#3-sast-核心思路物理先验引导的信任分配)
- [4. SST 的理论边界与 IF 估计失效区](#4-sst-的理论边界与-if-估计失效区)

**Part II — 水泵水轮机物理背景**
- [5. 机组参数与频率结构](#5-机组参数与频率结构)
- [6. 五种流态与频率演化](#6-五种流态与频率演化)
- [7. 物理边类型体系](#7-物理边类型体系)
- [8. 图自洽作为区分标准](#8-图自洽作为区分标准)

**Part III — 实测数据验证**
- [9. 频率稳定性分化](#9-频率稳定性分化)
- [10. 跨类倍频检验](#10-跨类倍频检验)

**Part IV — 方法论辨析**
- [11. 物理先验是否构成循环论证](#11-物理先验是否构成循环论证)

**Part V — SAST 完整设计**
- [12. 总体架构](#12-总体架构)
- [13. MSST：IF 估计基座](#13-msstif-估计基座)
- [14. 节点特征提取](#14-节点特征提取)
- [15. 静态原型软匹配](#15-静态原型软匹配)
- [16. 异构物理图](#16-异构物理图)
- [17. PPM + GAT：信任度推理](#17-ppm--gat信任度推理)
- [18. 逐 bin 阶数选择](#18-逐-bin-阶数选择)
- [19. 稀疏高斯重排](#19-稀疏高斯重排)
- [20. 训练与推理流程](#20-训练与推理流程)
- [21. 损失函数](#21-损失函数)

**Part VI — 实施与加速**
- [22. 实施路线与当前状态](#22-实施路线与当前状态)
- [23. CUDA 加速方案](#23-cuda-加速方案)

**Part VII — 可解释性分析**
- [24. 六大诊断维度与集成面板](#24-六大诊断维度与集成面板)

**Part VIII — 核心论证回顾**
- [25. 核心论证回顾](#25-核心论证回顾)

---

## Part I — 为什么需要 SAST

### 1. 核心问题：不同 TF bin 需要不同的 SST 阶数

SST 的 IF 估计基于一个局部假设：**信号在短时窗内可被 N 阶多项式相位近似**。

```
SST1 (N=1): 假设信号是纯简谐波  s(t) = A*exp(j*omega0*t)
            对线性 chirp 有偏, 偏置正比于 chirp_rate * sigma^2

SST2 (N=2): 假设信号是线性 chirp  s(t) = A*exp(j*(omega0*t + 0.5*alpha*t^2))
            对线性 chirp 无偏, 对二次 chirp 有偏

SSTN (N>=3): 假设是更高阶多项式相位
            对对应阶数无偏, 对更高阶有偏；且高阶导数放大噪声
```

**Colominas & Meignen (2025) Fig.1 的发现**：对同一个信号的**不同 TF 位置**，最优阶数不同。

```
同一信号, 同一时刻, 不同频率位置:
  靠近 IF (eta ≈ a): 高阶 IF 估计准确  → N=2 或 N=3 最优
  远离 IF (eta ≈ b): 高阶放大噪声      → N=1 反而最好

结论: 不存在全局最优的固定阶数。最优阶数是 (频率, 时间) 的函数。
```

这是一个逐 bin 的决策问题——时频平面上每个有能量的点 $(eta, t)$ 都应该有自己独立的 SST 阶数 $N^*(eta, t)$。

### 2. 水泵水轮机的三个分量正好对应三种需求

我们的信号中存在三类物理性质完全不同的频率分量：

| 分量 | 频段 | 信号特征 | 物理上正确的 N | Rényi 会选什么 |
|------|:---:|------|:---:|------|
| **2xBPF** | 90-105 Hz | 纯谐波, IF 极度稳定 (CV<1%) | **N=1 即可** | N=5（但增益极小, ΔR≈0.5） |
| **BPF** | 42-55 Hz | 受工况调制的 FM 信号, IF 在 ~15 Hz 带宽内漂移 | **N=3~5** | N=5（增益显著, ΔR≈2.5） |
| **LOW_FREQ** | 8-20 Hz | 水力涡带, 宽带非稳态, 无明确定义的 IF | **N=0** | **N=5**（Rényi 下降 1.9, 但这是破坏性的） |


### 3. 现有方法的局限

**固定阶 SST** (Oberlin & Meignen, 2015; Yu et al. 2019 MSST):
全频段统一 N，无法区分上述三种情况。MSST 的多次迭代虽然在逐步精化 IF，但迭代次数仍然是全局统一的——对 2xBPF 浪费计算且引入噪声，对 LOW_FREQ 强行挤压破坏信号。

**自适应阶 SST** (Colominas & Meignen, 2025):
做到了逐 bin 阶数选择，但用的是坐标下降——对每列 STFT 的每个非零系数穷举 $N_{max}$ 种阶数，选 $RE_{1D}$ 最小的。这是 NP-hard 组合优化问题，**无法用梯度下降**，且每个信号需要从头跑完整优化（数秒到数十秒）。不能泛化，不能注入物理先验。

### 4. SAST 的方案：从"逐 bin 穷举阶数"到"神经网络控制挤压"

SAST 不直接优化离散阶数，而是利用 MSST 的一个关键性质：**lookup 迭代对所有 bin 无害**。传统 SST 的高阶导数放大噪声，但 MSST 的 `omega_{k+1} = omega_k(omega_k)` 是沿频率轴向能量集中方向走的——纯谐波 bin 立即收敛，噪声 bin 也不会发散。

因此 SAST 让**所有 bin 统一使用 omega_5**（5 次 lookup 后的最优 IF），GAT 控制的是**挤压这个动作本身**：

```
论文 (Colominas): 为每个 bin 选 N ∈ {1,2,3,4} → 哪个 IF 估计?
SAST:             所有 bin 用 omega_5 (最优 IF)
                  GAT 控制: sigma_i (挤多宽) + lambda_i (挤几轮)
                  梯度通过 sigma_i → w_i → GAT 回传
```

```
w_i 的语义层级:
  节点级: GAT 输出 w_i → 决定该分量的挤压策略
          w_i → 1 (2xBPF):    sigma -> sigma_min (窄核), lambda -> 1 (一挤到位)
          w_i → 0.7 (BPF):    sigma -> 中, lambda -> 3~5 (多轮追 chirp)
          w_i → 0 (LOW_FREQ): sigma -> sigma_max (宽核), lambda -> 0 (不挤)

  bin 级 (推理): ridge_factor = energy_ratio * ridge_decay
          脊线 bin → 参与全部 lambda_i 轮; 远离 bin → 跳过
```

**为什么梯度下降可行**：训练时 lambda_i=1（单轮挤压），w_i 通过连续的 sigma_i 影响 TFR，RE_2D 对 sigma_i 可微，梯度完整回传。推理时 lambda_i = round(w_i * N_max) 自然继承 w_i 的排序。详见 §18。

### 5. 为什么选 STFT 而非 CWT 作为基座

| | CWT (WSST2) | STFT (MSST) |
|---|---|---|
| 频率轴 | 对数间隔 | **线性间隔** |
| 物理图 | 倍频关系在对数轴上非等距 | **倍频关系等距, 图直观** |
| 挤压方式 | δ 函数硬重排 | δ 函数硬重排 (**相同**) |
| 分析窗 | Q 恒定 (频率自适应宽度) | 固定长度 |
| 计算效率 | 低 | **高** (GPU cuFFT) |

两者的挤压操作完全相同（δ 函数硬重排），区别仅在分析窗。选 STFT 的理由是线性频率轴天然适合构建物理图（BPF→2xBPF 的 2:1 关系在 Hz 域等距），且 GPU 加速成熟（cuFFT batch FFT）。

**SAST 的创新不在分析阶段，在挤压阶段**——用可变宽度高斯核替代 δ 函数，核宽由 GAT 根据物理图自洽性逐 bin 控制。这是 WSST2 和 MSST 都不具备的能力。

### 6. SAST 的边界：什么它能改善，什么不能

在进入具体设计之前，需要诚实地界定 SAST 的能力边界——避免过度宣称，也避免低估其价值。

#### 6.1 TFR 增强：集中在两个区域

MSST 本身就是优秀的 TFR 基底。SAST 在三个分量上的增量是不对称的：

| 分量 | MSST（固定 δ 硬挤） | SAST（自适应高斯核） | 有视觉改善？ |
|------|------|------|:---:|
| 2xBPF (100Hz) | 已挤到近乎完美（纯谐波, CV<1%） | 窄核，与 MSST 几乎一致 | **几乎没有** |
| BPF (48Hz) | 硬挤到单一 bin，~15 Hz 的工况调制带宽被抹掉 | 中等核宽，保留调制结构 | **有** — TFR 更诚实, 不虚构"稳定谐波" |
| LOW_FREQ (12Hz) | 强行挤压宽带水力信号，制造伪结构 | 宽核/不挤，保留水力带宽 | **有** — 避免虚假的"锐利低频分量" |

SAST 的 TFR 改善不在 2xBPF（那已经不需要改善了），而在 BPF 和 LOW_FREQ——让时频图**不在没有谐波的地方假装有谐波**。

#### 6.2 对下游分类的直接贡献：可能很小

SAST 改变的是**频段内部的能量分布**（TFR 的"形状"），而分类器依赖的主要是**频段之间的能量比**（TFR 的"总量"）。

```
分类器 (GlobalAvgPool → MLP) 实际依赖的特征:
  R_LF  = sum(|TFR|² in 8-20 Hz)    ← 低频能量占比
  R_BPF = sum(|TFR|² in 42-55 Hz)   ← BPF 能量占比
  R_2x  = sum(|TFR|² in 90-105 Hz)  ← 2xBPF 能量占比

  这已被静态原型 V_obs = [R_LF, R_BPF, R_2xBPF, log(E_BPF/E_2x)] 完整捕获。
```

一个具体例子说明为什么全局池化后差异消失：

```
Class 3 (高负荷) 一个样本:

  MSST:
    BPF 频段: 能量全部集中在 48.0 Hz 单一 bin → sum = E_BPF
    2xBPF 频段: 能量集中在 100.7 Hz → sum = E_2x

  SAST:
    BPF 频段: 能量分布在 42-55 Hz (调制保留) → sum = E_BPF' (≈ E_BPF)
    2xBPF 频段: 与 MSST 几乎相同 → sum = E_2x (≈ 不变)

  GlobalAvgPool 后:
    MSST:  [..., E_BPF, ..., E_2x, ...]
    SAST:  [..., E_BPF', ..., E_2x, ...]  ← 几乎一样
```

SAST 重新分配了频段内的能量，但没有改变频段间的能量比。当分类器用 GlobalAvgPool 沿频率轴压缩时，这个差异被抹掉了。

**SAST 对分类精度的直接提升预计在 1-2% 以内**——前提是 MSST 基线已经足够高。如果基线已经在 95%+，这个提升在统计上甚至可能不显著。

#### 6.3 SAST 真正的价值：在分类之外

```
┌──────────────────────────────────────────────────────────────────┐
│ SAST 的价值层级 (由上到下: 从"谁都能做"到"只有 SAST 能做")         │
│                                                                   │
│ L1 — TFR 保真度: MSST 已解决 80%。                                  │
│      SAST 补充: BPF 调制保留 + LOW_FREQ 伪影抑制。                  │
│                                                                   │
│ L2 — 分类精度: SAST 的直接贡献很小 (预计 1-2%)。                    │
│      频段间能量比已是强特征，MSST 已保留了它。                       │
│      SAST 的价值不在让分类器从 95% 变成 97%。                       │
│                                                                   │
│ L3 — 可解释性: ★ 这是 SAST 的核心增量                               │
│      w_i(t): "此刻模型选择信任哪个分量？"                            │
│      gate_edge(t): "倍频关系此刻成立吗？" → 滑差/失速早期预警        │
│      alpha(t): "当前最像哪种工况原型？" → 工况漂移追踪               │
│      sigma_sq(eta,b): 逐 bin 展示"哪里在硬挤, 哪里在保留"            │
│      N*(eta,b): "每个 bin 用了什么阶数？"                            │
│      这些输出对故障诊断工程师的价值远超 TFR 本身。                    │
│                                                                   │
│ L4 — 鲁棒性与泛化:                                                  │
│      - 跨工况泛化: GAT 学的是物理关系 (比值自洽性), 不是频率值       │
│      - 异常检测: gate_edge 突然下降 → 物理关系破坏 → 预警            │
│      - OOD 检测: 所有 prototype 的 alpha 都低 → 工况不在包络内       │
│      - 跨设备迁移: 换叶片数 Z_r → 改图模板即可, 不需重新训练         │
└──────────────────────────────────────────────────────────────────┘
```

#### 6.4 两类目标的两种路线

| 目标 | 推荐方案 | SAST 的角色 |
|------|---------|------------|
| **纯分类**: 区分 5 种流态 | MSST + 静态原型 (V_obs → MLP) | 可有可无，增益很小 |
| **诊断与理解**: 知道模型为什么这样判断、在异常工况下不瞎猜、给出可解释的物理诊断 | MSST + SAST | 核心组件，不可替代 |

SAST 的设计目标从一开始就是后者——**不是为了把 95% 的分类精度提到 97%，而是让模型的判断有物理依据可循、在训练包络外不盲目自信。**

---

## Part II — 水泵水轮机物理背景

### 5. 机组参数与频率结构

#### 5.1 机组设计参数

| 参数 | 取值 | 单位 |
|------|:---:|------|
| 额定转速 | 333.3 | rpm |
| 转轮叶片数 Z_r | 9 | — |
| 活动导叶数 Z_s | 20 | — |
| 额定输出功率 | 306.12 | MW |
| 最大输入功率 | 322.0 | MW |

**转频**: fr = 333.3 / 60 = **5.555 Hz**

**理论倍频**:
- BPF = Z_r * fr = 9 * 5.555 = **50.0 Hz**（叶片通过频率）
- RSI 耦合阶 nu = floor(Z_s / Z_r) = floor(20 / 9) = **2**
- RSI = nu * BPF = 2 * 50.0 = **100.0 Hz**（动静干涉频率）— 与 2*BPF 重合

#### 5.2 实测频谱分析结论

基于 5 类 9669 样本 FFT 均值频谱分析，确定 3 个物理节点：

| 节点 | 频段 | f_type | C_prior | bw_expected | persist_expected |
|------|:---:|------|:---:|:---:|:---:|
| LOW_FREQ | 8-20 Hz | HYDRAULIC | 0.30 | 10 Hz | 0.50 |
| BPF | 42-55 Hz | BLADE_PASS | 0.60 | 12 Hz | 0.85 |
| 2xBPF | 90-105 Hz | BLADE_HARMONIC | 0.90 | 2 Hz | 0.95 |

**精简依据**（从原始 7 节点设计精简为 3）:

| 原节点 | 删除原因 |
|------|------|
| fr (~5.5 Hz) | 加速度传感器 omega^2 衰减 81 倍, 不可观测 |
| 3xBPF (150 Hz) | 与 50 Hz 相关性 ~0, 能量占比 0.07-0.32% |
| GPF (111 Hz) | 频谱中 100 Hz 区域无独立峰 |
| HIGH_HARMONIC (468 Hz) | 仅 31-45% 样本出现, 能量 <0.08%, 判定为噪声 |

> **关键发现**: 实测数据只支持一个可靠的倍频关系 (BPF->2xBPF, r~2.0)。468 Hz 与 100 Hz 的比值 4.68 在小样本中显得稳定，但全样本均值谱中能量太低。

**频率结构图**:

```
                         LOW_FREQ (水力低频)
                         8-20 Hz
                         [来源: 涡带/压力脉动, 与机械路径独立]
                         [无边 — 独立分量]

                    BPF ----------[ r=2.0 ]----------> 2xBPF
              (叶片通过频率)     INTEGER_HARMONIC     (二倍叶片通过频率)
               42-55 Hz, 宽峰      w=0.8              90-105 Hz, 尖峰
               [工况调制 -> 展宽]                      [极度稳定, CV<1%]
               [C_prior=0.60]                         [C_prior=0.90]
```

#### 5.3 关键设计约束：图中不嵌入具体频率值

物理图的节点只携带**结构身份**（"这是转频"、"这是二倍叶片通过频率"），**不预设任何具体的 Hz 值**。

```
错误做法: 节点携带 fr = 5.56 Hz, BPF = 50.0 Hz 等标称值作为特征
          -> 模型可能直接记住频率值, 学到的是查表而非推理
          -> 跨设备、跨工况泛化失败

正确做法: 节点特征只来自信号:
          [IF_raw(t), energy(t), bandwidth(t), persistence(t)]
          节点身份由它与其他节点的关系定义, 而非预设 Hz 值
          GAT 通过边特征 r_obs = IF_i/IF_j 推断关系是否成立
```

**核心原则**: 图用**无量纲比值**（ratio）而非**有量纲频率**（Hz）编码物理关系。比值 9.0 是"转轮有 9 个叶片"这一机械事实的无量纲表达。

### 6. 五种流态与频率演化

数据集包含 5 种运行工况：

| 流态 | 运行状态 | 频率特征 | 类别 |
|------|------|------|:---:|
| 空转 | 低速旋转, 导叶开度极小 | **低频主导**, BPF 弱, RSI 微弱 | Class 0 |
| 低负荷 (75-125MW) | 导叶开度增加 | **BPF 主导**, 低频+RSI 均存 | Class 1 |
| 中负荷 (150MW) | 导叶进一步开大 | **RSI 增强主导**, 低频+BPF 减弱 | Class 2 |
| 高负荷 (175-300MW) | 导叶开度大, 无叶区小 | **RSI 极强**, 低频几乎消失 | Class 3 |
| 抽水 | 反向旋转, 导叶接近全开 | BPF+RSI **均强且接近** | Class 4 |

**频率演化规律（此消彼长）**:

```
空转         低负荷        中负荷        高负荷        抽水
──────────────────────────────────────────────────────────
低频主导 -> BPF 上升  -> RSI 主导  -> RSI 极强  -> BPF+RSI 均强
RSI 弱      RSI 仍弱    低频减弱     低频消失     双主导
BPF 弱      BPF 主导    BPF 减弱     BPF 弱       BPF 强(回升)
```

**物理本质**: 低负荷时涡带主导（水力能量 -> 低频脉动 + BPF 调制），高负荷时动静干涉主导（流道能量 -> RSI）。这是**流道能量守恒**的体现——能量要么去涡带，要么去动静干涉。

**对 SAST 的启示**: GAT 的 w_i 必须随流态自适应——同一频率分量在不同工况下需要不同的挤压策略。

### 7. 物理边类型体系

基于数据验证和物理机制，定义四种边类型：

#### 7.1 边类型总览

| 边类型 | 约束强度 | 物理含义 | 特征维度语义 |
|------|:---:|------|------|
| **INTEGER_HARMONIC** | 强 (w=0.8) | 整数倍频 (BPF->2xBPF, r=2.0) | [r_obs, sigma_r, Corr_E, w_type, p_min] |
| **CONDITION** | 中 (w=0.6) | 工况上下文广播 (OP -> 频率节点) | [cond_sim, 0, 0, w_type, 0] |
| **DRIFT** | 弱 (w=0.15) | 能量耦合 / 调制关系 (LOW_FREQ <-> BPF) | [Corr_E, E_ratio, bw_coupling, w_type, p_min] |
| **ENERGY_COMPETITION** | 弱-中 (w=0.20-0.25) | 能量竞争 / 此消彼长 | [-Corr_E, E_ratio, E_ratio_stability, w_type, p_min] |

#### 7.2 HARMONIC 边：确定性倍频

唯一确定性边。叶片数 Z=9 决定了 BPF 的二次谐波必然是 BPF 的 2 倍。

```
e_{BPF->2xBPF}(t) = [r_obs, sigma_r, Corr_E, w=0.8, p_min]

r_obs(t)   = IF_2xBPF(t) / IF_BPF(t)          <- 实测比值, 应 ~2.0
sigma_r(t) = std(r_obs over window)             <- 比值稳定性
Corr_E(t)  = corr(energy_BPF, energy_2xBPF)     <- 能量相关性 (应正相关)
p_min      = min(persist_BPF, persist_2xBPF)    <- 最小持续性
```

**门控**: gate = exp(-|r_obs - 2.0| / (2.0 * tau))。比值越接近 2.0，门控越接近 1。

#### 7.3 DRIFT 边：能量调制耦合

LOW_FREQ 和 BPF 之间没有整数倍频关系，但存在**能量调制耦合**：涡带增强 -> 无叶区压力脉动 -> BPF 被调制展宽。

```
e_{LOW_FREQ<->BPF}(t) = [Corr_E, E_ratio, bw_coupling, w=0.15, p_min]

Corr_E(t)     = corr(energy_low, energy_bpf)      <- 应正相关
E_ratio(t)    = log(E_low / E_bpf)                 <- 随工况单调递减
bw_coupling(t)= corr(bw_low, bw_bpf)               <- 带宽是否共变
```

**门控**: gate = sigmoid((|Corr_E| - 0.3) / 0.1)。能量相关性高时门控高。

#### 7.4 CONDITION 边：工况上下文广播

OP 虚拟节点向三个频率节点广播工况上下文。

```
e_{OP->node}(t) = [cond_sim, 0, 0, w=0.6, 0]

cond_sim = cosine_similarity(h_phys_projected, cond_ctx)
```

**门控**: gate = sigmoid(cond_sim / 0.1)。节点特征与工况上下文越匹配，门控越高。

#### 7.5 ENERGY_COMPETITION 边：此消彼长

新增边类型，编码流道能量守恒——能量要么去涡带+BPF（低负荷主导），要么去动静干涉/RSI（高负荷主导）。两者能量**反相关**。

```
e_{LOW_FREQ<->2xBPF}(t) = [-Corr_E, E_ratio, E_ratio_stability, w=0.25, p_min]
e_{BPF<->2xBPF}(t)      = [-Corr_E, E_ratio, E_ratio_stability, w=0.20, p_min]

-Corr_E(t)             = -corr(energy_src, energy_dst)  <- 负相关程度
                                         (越大越"此消彼长")
E_ratio(t)             = log(E_src / E_dst)              <- 阵营能量比
E_ratio_stability(t)   = exp(-std(E_ratio)/0.5)          <- 比值稳定性
                                         (定工况样本内应稳定 ~1.0)
```

**门控**: gate = sigmoid((-Corr_E - 0.3) / 0.1)。显著负相关时门控高。

**权重差异**: LOW_FREQ <-> 2xBPF (w=0.25) 是直接的物理竞争；BPF <-> 2xBPF (w=0.20) 较弱——BPF 本身也是谐波链的一部分，不是纯粹的竞争者。

### 8. 图自洽作为区分标准

**问题**: 在没有真值的情况下，怎么知道一个频率分量该硬挤还是软挤？

**答案**: 图上有边且比值稳定 = 可信 = 硬挤；无边或比值漂移 = 不可信 = 保守。

```
2xBPF (~100 Hz, 尖峰):
  3 条入边 (CONDITION + HARMONIC + ENERGY_COMPETITION)
  实测比值 r_obs ~ 96/48 = 2.0 -> 图自洽 YES
  边特征: sigma_r 小, Corr_E 高
  -> 硬挤: GAT 给高 w_i

LOW_FREQ (~12 Hz, 水力):
  3 条入边 (CONDITION + DRIFT + ENERGY_COMPETITION)
  无边可做倍频验证 -> 无法交叉确认
  -> 保守: GAT 依赖节点自身特征 (能量、带宽、持续性)
  带宽大/能量不稳定 -> w_i 降低 -> 软挤保留
```

**SAST 的五条独立证据链**:

1. **sigma_r** (比值稳定性，时间维度): 噪声比值逐帧乱跳, 连续稳定需全部碰对
2. **Corr_E** (能量相关性，激励源维度): 独立噪声源能量不相关
3. **bw_score** (频谱形态): 噪声宽度、峰度不匹配物理预期
4. **C_prior** (工况一致性): 工况先验阻止不合理的高信任
5. **多边交叉验证** (图拓扑): 多角度独立验证, 联合概率极低

---

## Part III — 实测数据验证

### 9. 频率稳定性分化

以 Class 3 (高负荷) WSST2 脊线追踪结果为例：

| 频率 | IF 变异系数 CV | 瞬时带宽 | Corr(IF, BW) | 分类 |
|------|:---------:|:------:|:-----------:|------|
| **100.7 Hz** (2xBPF) | **0.69%** | 1.35 Hz | **-0.15** | 稳定谐波 -> 硬挤 |
| 47.2 Hz (BPF 邻域) | 3.46% | 2.03 Hz | **+0.16** | 滑差分量 -> 软挤 |
| 468.5 Hz (9xBPF) | 3.02% | 9.67 Hz | -0.09 | 高阶谐波 -> 中间态 |

**关键洞察**: 稳定谐波的 Corr(IF, BW) = -0.15（带宽不随 IF 变化，展宽是测量模糊）；滑差分量的 Corr(IF, BW) = +0.16（带宽与 IF 漂移正相关，展宽含真实物理信息）。

### 10. 跨类倍频检验

对全部 5 个类别的脊线追踪和逐帧比值分析：

| 频率对 | 名义倍数 | 实际比值(均值) | CV 范围 | 稳定性 |
|--------|:---:|:---:|:---:|:---:|
| 468.5 / 100.7 | 4.68 (非整数) | 4.67~4.69 | **0.7~1.8%** | STABLE |
| 100.7 / 47.2 | 2 | 2.03~2.20 | 3.3~6.1% | DRIFT |
| 468.5 / 47.2 | 10 | 9.53~10.29 | 2.7~6.8% | DRIFT |

**发现**: 最稳定的频率关系不是整数（468.5/100.7 ~4.68, CV < 2%）。整数倍频关系在实际数据中普遍有 3-7% 的漂移——这恰恰是物理图需要用比值稳定性（sigma_r）来区分可信与不可信的原因。

---

## Part IV — 方法论辨析

### 11. 物理先验是否构成循环论证

SAST 的物理图约束了时频分析的能量重分配策略。自然的质疑：倍频关系是从信号中分析出来的，又用它约束信号处理——这是循环吗？

**不是。理由如下：**

#### 拓扑结构来自机械设计参数，独立于振动信号

```
Z_r = 9  (转轮叶片数)     -> BPF = 9 * fr
Z_s = 20 (活动导叶数)     -> RSI = 2 * BPF (nu = floor(20/9) = 2)
```

这些是**机械事实**（把机器拆开数叶片，得到的就是 9 和 20）。无论是否测量振动、信号长什么样，这些关系恒成立。

#### GAT 不输出物理值，只输出信任度

```
GAT 输入:  节点特征 (IF, 能量, 带宽) + 边特征 (r_obs, sigma_r, Corr_E)
           物理先验 (拓扑, 边类型, 约束强度)
GAT 输出:  w_i ∈ (0,1) — 每节点每帧的 IF 信任度
GAT 不输出: "这个比值应该是多少"

GAT 问的不是 "BPF 在多少 Hz？"
GAT 问的是 "此时此刻, 我在图上读到的物理关系还有效吗？"
```

这类似于 Kalman 滤波——状态方程 F 和观测矩阵 H 来自物理模型，融合传感器数据后输出状态估计。SAST 的 GAT 同样是"物理模型 + 实测数据 -> 最优策略"的融合架构。

#### 对错误先验的鲁棒性：软门控自动退火

所有经验性约束使用连续的门控而非二值判断。标称值不精确时，gate 平滑下降，原型知识平滑退火到无信息先验 (w_i -> 0.5)，让 GAT 完全依赖实测边特征做决策。

---

## Part V — SAST 完整设计

### 12. 总体架构

```
输入信号 x[T]  (T=1000, fs=1000 Hz)
         |
         v
+---------------------------------------------------+
|  MSST (N_max=5, save_trajectory=True)              |  <- numpy
|  输出: STFT [F, T], omega_final [F, T],            |
|        omegas [N_max, F, T]                        |
+--------------------------+------------------------+
                           |
                           v
+---------------------------------------------------+
|  MSSTNodeExtractor                                 |  <- numpy
|  按 3 个频段聚合 -> node_feats [3, T, 4]            |
+--------------------------+------------------------+
                           |
                           v
+---------------------------------------------------+
|  StaticPrototypeMatcher                            |  <- torch (GPU)
|  V_obs = [R_LF, R_BPF, R_2xBPF, log(Eb/E2x)]       |
|  余弦相似度 + softmax -> cond_ctx, alpha             |
+--------------------------+------------------------+
                           |
                           v
+---------------------------------------------------+
|  异构物理图 + PPM + GAT                             |  <- torch (GPU)
|  10 条有向边, 4 节点, 4 边类型                      |
|  PPM: 原型增强 + 类型嵌入 + 门控融合                 |
|  GAT: 2 层 x 4 头, edge-conditioned attention       |
|  输出: w_i [3, T] (IF 信任度)                       |
+--------------------------+------------------------+
                           |
              +------------+------------+
              |                         |
              v                         v
+--------------------------+  +--------------------------+
|  sigma_i 映射 (连续, 可微)|  |  PerBinOrderSelector      |
|  sigma_i = sigma_min +   |  |  训练: 固定 N_max=5        |
|    (1-w_i)*delta_sigma   |  |  推理: N*(eta,b) =         |
|  -> sigma_sq [F, T]      |  |    round(lambda_i *        |
|                          |  |      conv_decay *          |
|                          |  |      struct_gate)          |
+------------+-------------+  +------------+-------------+
             |                             |
             +-------------+---------------+
                           v
+---------------------------------------------------+
|  SparseGaussianReassigner                          |  <- torch (GPU)
|  论文式稀疏 A_n 矩阵: 每列 ~2*ceil(3*sigma)+1 非零元  |
|  s_n = A_n * f_n  (scatter_add)                     |
|  输出: TFR_sast [F, T]                              |
+---------------------------------------------------+
```

### 13. MSST：IF 估计基座

MSST (Multi-Synchrosqueezing Transform) 是 Yu et al. (2019) 提出的迭代 IF 精化方法。

**算法四步**:

```
Step 1 — STFT:
  高斯窗 g(t) = exp(-pi*t^2/0.32^2), hlength = min(N, 512)
  逐列: x[t-Lh:t+Lh] * g -> FFT -> STFT[t, :]
  输出: STFT [F, T], F=N/2, T=N

Step 2 — 一阶 IF 估计:
  omega[i, :] = round(diff(unwrap(angle(STFT[i, :]))) * N / 2*pi)
  输出: omega_1 [F, T], int32 (1-indexed bin)

Step 3 — IF 迭代精化 (共 num-1 次):
  omega_{k+1}[eta, b] = omega_k[omega_k[eta, b], b]
  输出: omega_final [F, T] + omegas [num, F, T]

Step 4 — 硬挤压:
  若 |STFT[eta,b]| > threshold: Ts[omega_final[eta,b]-1, b] += STFT[eta,b]
  输出: MSST [F, T]
```

**对 SAST 的作用**: MSST 提供 STFT（挤压源数据）和 omega_final + omegas（GAT 的节点特征和逐 bin 阶数选择）。

### 14. 节点特征提取

**不追踪脊线**——MSST 的 omega_final 已经逐 bin 计算好 IF 估计值（解析计算，非追踪）。SAST 只需按频率区域聚合统计量。

对每个物理节点 k 的频段 [f_min, f_max]:

```
IF_raw(t)    = median(omega_final 在频段内的有效值) -> 转为 Hz
energy(t)    = log(1 + sum(|STFT|^2 在区域内))
bandwidth(t) = std(omega_final 在区域内), 钳制到 20 Hz
persistence  = (energy > threshold 的帧数) / 总帧数
```

**频率区域定义**:

| 节点 | f_min | f_max | f_type | C_prior | bw_expected | persist_expected |
|------|:---:|:---:|------|:---:|:---:|:---:|
| LOW_FREQ | 8 Hz | 20 Hz | HYDRAULIC | 0.30 | 10 Hz | 0.50 |
| BPF | 42 Hz | 55 Hz | BLADE_PASS | 0.60 | 12 Hz | 0.85 |
| 2xBPF | 90 Hz | 105 Hz | BLADE_HARMONIC | 0.90 | 2 Hz | 0.95 |

### 15. 静态原型软匹配

#### 15.1 离线 EDA

对 5_dataset.npz 的 5 类逐样本计算 FFT 三个频段的能量占比 -> 每类求均值：

```
V_proto = [R_LF, R_BPF, R_2xBPF, log10(E_BPF/E_2xBPF)]

Class 0 (No-load):  [0.038, 0.209, 0.027, +0.99]
Class 1 (Low load): [0.068, 0.833, 0.010, +1.99]
Class 2 (Mid load): [0.046, 0.270, 0.304, +0.00]
Class 3 (High load):[0.013, 0.153, 0.652, -0.65]
Class 4 (Pumping):  [0.021, 0.686, 0.099, +0.85]
```

硬编码在模型中 (`register_buffer`)，不可学习。

#### 15.2 在线软匹配（余弦相似度，不分类）

```
每帧 t:
  V_obs(t) = 从 node_energy 实时计算 4D 向量
  sim(t, k) = cosine_similarity(V_obs(t), prototype_k)  k=0..4
  alpha(t) = softmax(sim / tau)  -> [5] 注意力权重

  cond_ctx(t) = sum_k alpha_k(t) * P_embed[k]
  其中 P_embed [5, d_cond] 是可学习的 "原型解释" 嵌入
```

**为什么不预测工况类别**: alpha 是对 5 个原型的软注意力权重，不是类别概率。模型不被告知"你现在是 Class 3"。好处：(1) 不需要故障标签；(2) 跨类泛化自然发生；(3) 训练推理完全一致。

### 16. 异构物理图

#### 16.1 图拓扑（10 条有向边）

```
                +---------------------+
                |   OP (virtual)       |  工况上下文嵌入
                +------+--+--+--------+
          CONDITION   |  |  |  CONDITION
          (w=0.6)     |  |  |  (w=0.6)
                       v  v  v
      +----------+  +----------+  +----------+
      | LOW_FREQ |  |   BPF    |  |  2xBPF   |
      |  (idx=1) |  |  (idx=2) |  |  (idx=3) |
      +----+-----+  +--+---+---+  +----+-----+
           |           |   |            ^
           |   DRIFT   |   +------------+
           |  (w=0.15) |   HARMONIC (r=2.0, w=0.8)
           |           v
           +---------->+
         [能量耦合]

         ENERGY_COMPETITION (此消彼长):
           LOW_FREQ <--> 2xBPF  (w=0.25)
           BPF      <--> 2xBPF  (w=0.20)
         [负能量相关: 流道能量守恒]
```

#### 16.2 完整边表

| 边 | 类型 | 物理含义 | w |
|---|------|---------|:---:|
| OP -> LOW_FREQ | CONDITION | 工况信息广播 | 0.6 |
| OP -> BPF | CONDITION | 工况信息广播 | 0.6 |
| OP -> 2xBPF | CONDITION | 工况信息广播 | 0.6 |
| LOW_FREQ -> BPF | DRIFT | 涡带能量 -> BPF 调制 | 0.15 |
| BPF -> LOW_FREQ | DRIFT | 调制反馈 | 0.15 |
| BPF -> 2xBPF | HARMONIC | 确定性倍频 r=2.0 | 0.8 |
| LOW_FREQ -> 2xBPF | ENERGY_COMPETITION | 涡带 vs RSI 此消彼长 | 0.25 |
| 2xBPF -> LOW_FREQ | ENERGY_COMPETITION | 反向 | 0.25 |
| BPF -> 2xBPF | ENERGY_COMPETITION | BPF vs RSI 此消彼长 | 0.20 |
| 2xBPF -> BPF | ENERGY_COMPETITION | 反向 | 0.20 |

#### 16.3 节点-边入度矩阵

每个物理节点至少收到 3 条入边（CONDITION + DRIFT/COMPETITION + SELF 等），scatter_softmax 在入边之间竞争，学习"此时应该更听哪条边的"。

| 节点 | 入边 | 出边 |
|------|------|------|
| OP | 0 | -> LOW_FREQ, BPF, 2xBPF (x3) |
| LOW_FREQ | OP, BPF, 2xBPF (x3) | -> BPF, 2xBPF (x2) |
| BPF | OP, LOW_FREQ, 2xBPF, 2xBPF(comp) (x4) | -> LOW_FREQ, 2xBPF, 2xBPF(comp) (x3) |
| 2xBPF | OP, BPF, LOW_FREQ, BPF(comp) (x4) | -> LOW_FREQ, BPF(comp) (x2) |

### 17. PPM + GAT：信任度推理

#### 17.1 PPM (Physics Prototype Memory)

PPM 做三件事：

**a) 类型嵌入增强**: 每个物理节点的 f_type 映射到可学习嵌入，加到 prototype embedding 上。

**b) 交叉注意力**: 物理节点观测特征 Q 与 prototype K/V 做交叉注意力（带 self-bias=3.0，鼓励关注自身 prototype）。

**c) 门控融合**: h_enhanced = (1-gate) * h_raw + gate * proto_context。gate->1 时更信任 prototype；gate->0 时退回到原始特征。

**C_prior 调制**:
```
C_prior_i(cond) = sigmoid(logit_base_i + W_cond[i] * cond_ctx)
```
cond_ctx 来自 StaticPrototypeMatcher，对不同能量分布模式提供不同先验偏向。

#### 17.2 EdgeConditionedGAT

2 层边条件图注意力，4 头，scatter_softmax over destination nodes。

```
每层:
  Q = W_q(h), K = W_k(h), V = W_v(h), E = W_e(edge_feats)
  attn_logits = a * [q_dst || k_src || e_ij]   <- 边条件
  attn_weights = scatter_softmax(attn_logits, dst_nodes)
  h_new = scatter_add(attn_weights * V_src, dst_nodes)
  h = LayerNorm(h + ReLU(proj(h_new)))
```

**输出头**: w_i = sigmoid(MLP(h_phys)) ∈ (0, 1)，每物理节点每帧。

**GAT 学到的行为**（无需手动编码）:
- 稳定比值 + 高能量相关 -> 高注意力 -> w_i↑ -> sigma↓ -> 硬挤
- 漂移比值 + 低能量相关 -> 低注意力 -> w_i↓ -> sigma↑ -> 软挤
- 显著负能量相关 + 比值稳定 -> COMPETITION 边激活 -> 两阵营区分对待

### 18. 挤压迭代控制

#### 18.0 为什么是挤压次数而不是 IF 阶数

传统 SST 的"阶数"控制 IF 估计的 Taylor 展开阶数——高阶导数放大噪声，所以噪声 bin 需要低阶。但 MSST 的 IF 精化机制不同：

```
传统 SST 高阶: 计算 chirp 率 = f''(t) → 高阶导数的数值估计放大噪声, 尤其在低 SNR 区域
MSST 迭代:    omega_{k+1}[eta,b] = omega_k[omega_k[eta,b], b]
              本质是沿频率轴的能量集中方向做固定点迭代
              朝能量更集中的 bin 走一步 — 对任何 bin 都无害
```

**MSST 的 lookup 迭代对所有 bin 都是安全的**——它总是向能量集中的方向收敛，不会像高阶导数那样在噪声区发散。因此：

```
所有 bin 统一使用 omega_5（5 次 lookup 后的最优 IF 估计）。
不需要为不同 bin 选不同的 IF 层。
```

**真正需要 GAT 控制的不是"用哪个 IF"，而是"挤不挤、挤几次"**。挤压是一个破坏性的动作——它把能量从原始位置挪到 IF 指向的位置。这个动作对谐波是正确的（消除窗函数模糊），对滑差/噪声是错误的（虚构集中）。多次挤压-重估 IF-再挤压的循环，每轮提升 TFR 的 SNR，有助于 BPF 的 chirp 追踪：

```
Algorithm:
  1. 所有 bin: omega ← lookup 5 次 → omega_5 (最优 IF, 一次性, 无梯度)
  2. GAT 输出 w_i → sigma_i (核宽) + lambda_i (挤压轮数)
  3. 逐节点逐帧:
       for round in 1..lambda_i:
         squeeze(STFT, omega_5, sigma_i) → TFR_round
         re-estimate omega_5 on TFR_round  (SNR 逐轮提升)
         (训练时 lambda_i=1, 推理时启用多轮)
  4. ridge_factor: 同节点内, 脊线 bin 全参与, 远离 bin 跳过
```

**sigma 和 lambda 的分工**: sigma 控制"挤多宽"（连续，可微，训练时使用），lambda 控制"挤几次"（离散，推理时使用）。两者都由 w_i 派生——w_i 高的分量（2xBPF）窄核 + 少轮（一轮到位），w_i 中等的分量（BPF）中核 + 多轮（逐轮追 chirp），w_i 低的分量（LOW_FREQ）宽核 + 零轮（不挤）。

#### 18.1 节点级 lambda_i

GAT 输出的 w_i 确定每个物理节点的挤压轮数：

```
lambda_i = round(w_i * N_max)    N_max=5

w_i -> 1 (2xBPF):  lambda_i = 5 → 全轮次 (但一轮已收敛, 多余无害)
w_i -> 0.7 (BPF):  lambda_i = 3~4 → 多轮追 chirp
w_i -> 0 (LOW_FREQ): lambda_i = 0 → 不挤压
```

#### 18.2 逐 bin 分化：ridge_factor

同一节点内，脊线上的 bin 应参与全部 lambda_i 轮挤压，远离脊线的 bin 应跳过：

```
ridge_pos(t)  = argmax_{eta in band_i} |STFT[eta, t]|
energy_ratio  = |STFT[eta, t]| / |STFT[ridge_pos(t), t]|
ridge_dist    = |eta - ridge_pos(t)| / bw_expected_i
ridge_decay   = exp(-ridge_dist^2 / 2)
ridge_factor  = energy_ratio * ridge_decay

bin 级挤压轮数: N_sqz(eta, t) = round(lambda_i * ridge_factor)
```

**效果**: 脊线 bin → ridge_factor≈1 → 全轮次；远离 bin → ridge_factor≈0 → 跳过。与 sigma_i 形成互补——sigma 控制"多宽"，lambda*ridge_factor 控制"挤几轮"。宽核 + 零轮 = 保留原貌；窄核 + 多轮 = 精细集中。

#### 18.3 与 Colominas & Meignen (2025) 论文的对比

**论文的 IF 阶数选择**: 对传统 SST（非 MSST），高阶导数放大噪声，导致远离 IF 的 bin 需要降阶。本质是在"信号近似精度"和"噪声敏感度"之间权衡。

**SAST 的挤压迭代控制**: MSST 的 lookup 迭代对噪声无害（只朝能量集中方向走），因此不需要选 IF 阶数。控制力转移到挤压动作本身——"该挤的 bin 挤几轮"比"该 bin 用几阶 IF"更直接地对应物理需求。

| 维度 | 论文 (IF 阶数选择) | SAST (挤压轮数控制) |
|------|:---:|:---:|
| 控制对象 | 用哪层 IF 估计 (omega_k) | 挤几轮 (squeeze 重复次数) |
| 噪声 bin 行为 | 降阶, 用低阶 IF 减少噪声放大 | 跳过, ridge_factor→0, 不参与挤压 |
| 谐波 bin 行为 | 升阶, 用高阶 IF 追 chirp | 多轮, 每轮重估 IF 提升 SNR |
| 为什么可行 | N/A (穷举, 不可微) | MSST lookup 对所有 bin 无害 → 统一用 omega_5 → GAT 只需学"挤不挤" |

### 19. 稀疏高斯重排

**论文式稀疏矩阵乘法** (Colominas & Meignen 2025):

论文的硬重排: A_n[p, m] = 1 if p == round(omega_hat[m]) else 0（每列 1 个非零元）

SAST 的软重排（高斯核替代 delta 函数）:

```
A_n(p, m; sigma, N*) = (1/Z) * exp[-(p - omega_hat[N*(m)])^2 / (2*sigma^2)]

omega_hat[N*(m)] 来自逐 bin 阶数选择
每列有 2*ceil(3*sigma)+1 个非零元（banded sparse）
```

**稀疏性**:

| sigma = 0.5 (w_i->1) | sigma = 15 (w_i->0) | Dense |
|:---:|:---:|:---:|
| 每列 ~5 非零元 | 每列 ~91 非零元 | 每列 F=500 非零元 |
| 几乎等价于硬挤压 | 严重模糊 (保留滑差) | |

**实现**: 对每个 sigma 量化级别，预计算高斯权重，用 scatter_add 沿频率轴分配。

### 20. 训练与推理流程

#### 20.1 训练（全自监督，不需要标签）

```
输入: signal [T]  <- 仅信号, 无标签

1. MSST (num=5, save_trajectory=True) -> STFT, omegas
2. MSSTNodeExtractor -> node_feats [3, T, 4]
3. V_obs = [R_LF, R_BPF, R_2xBPF, log_ratio]
4. StaticPrototypeMatcher(V_obs) -> cond_ctx, alpha
   (余弦相似度匹配, 不涉及标签)
5. 边特征计算 (numpy -> torch)
6. 逐帧 PPM -> GAT -> w_i [3, T]
7. sigma_i = sigma_min + (1-w_i)*delta_sigma  (可微)
8. 固定 N_max=5, 使用 omegas[-1]
9. SparseGaussianReassigner(omegas[-1], sigma_i) -> TFR_sast
10. L_total = lambda_e*RE_2D + lambda_p*L_physics + lambda_s*L_smooth + lambda_b*L_balance
    <- 全部自监督
11. 反向传播 -> 更新 GAT + P_embed + C_prior_logit
```

**梯度路径**: L_total -> TFR_sast -> sigma_i -> w_i -> GAT -> PPM -> (P_embed, C_prior_logit)

**训练/推理差异**: 训练时固定 N_max=5（统一使用 omegas[-1]），推理时启用逐 bin 阶数选择。lambda_i 的离散选择不可微，但在推理时直接继承 w_i 的排序。

#### 20.2 推理

```
1-6. 与训练相同
7. sigma_i = sigma_min + (1-w_i)*delta_sigma
8. PerBinOrderSelector(w_i, omegas) -> N* [F, T], order_idx [F, T]
9. SparseGaussianReassigner(omegas[order_idx], sigma_i) -> TFR_sast
```

### 21. 损失函数（全自监督）

```
L_total = lambda_e*RE_2D + lambda_p*L_physics + lambda_s*L_smooth + lambda_b*L_balance
```

| 项 | 名称 | 监督来源 | 需要标签？ | lambda |
|---|------|:---:|:---:|:---:|
| RE_2D | Rényi 2D 熵 | TFR 自身集中度 | 否 | 0.1 |
| L_physics | 比值偏差 * w_src * w_dst * gate | 叶片数 Z=9 | 否 | 0.5 |
| L_smooth | w_i(t) 帧间差分 | 正则化 | 否 | 0.05 |
| L_balance | w_mean 区间约束 [0.3, 0.8] | 正则化 | 否 | 0.01 |

**没有分类交叉熵**: 训练完全自监督，不需要故障标签。每个节点都有至少 3 条入边，静态原型已将工况知识编码为软匹配。

#### 21.1 RE_2D — 全局 TFR 集中度

```
RE_2D(TFR) = 1/(1-alpha) * log sum_m sum_n (|TFR[m,n]|/sum|TFR|)^alpha
alpha=2, 越低越集中
```

若只有 RE_2D: GAT 把全部 bin 挤到最窄 -> 滑差抹杀 -> 退化为固定阶 SST。
若只有 L_physics: 仅 HARMONIC 边有监督 -> LOW_FREQ 无约束。
**两者一起**: RE_2D 提供全局梯度，L_physics 按边类型修正——"结构感知"。

#### 21.2 L_physics — 自适应物理约束

```
L_physics = mean_edges [ w_type * |r_obs - r_nom|/r_nom * w_src * w_dst * gate_edge ]

情况 1: r_obs ~ 2.0 (倍频满足) -> gate->1 -> w 高 -> L 小 -> 无惩罚
情况 2: r_obs != 2.0 -> gate->0 -> w 高 -> L 大 -> 梯度迫使 w↓
情况 3: r_obs != 2.0 -> gate->0 -> w 已低 -> w_src*w_dst->0 -> L 自动小
```

自适应效应：不需要手动调权重来区分工况。

#### 21.3 各损失项的退化防御

| 退化场景 | 防御机制 |
|------|------|
| GAT 全部输出 w->1 | L_balance (w_mean > 0.8 -> 惩罚) |
| LOW_FREQ 被过度挤压 | DRIFT gate->0 + COMPETITION -> w 高 -> L_physics 大 |
| BPF 被过度挤压 | HARMONIC gate->0 -> w_BPF*w_2xBPF 高 -> L_physics 大 |
| 什么都不挤 (全部 w->0) | L_balance (w_mean < 0.3) + RE_2D 高 |
| w_i 剧烈抖动 | L_smooth |

---

## Part VI — 实施与加速

### 22. 实施路线与当前状态

#### 22.1 已完成

- [x] MSST (numpy) — `models/tfr.py`，含 save_trajectory
- [x] MSSTNodeExtractor — `models/sast_nodes.py`
- [x] 异构物理图 (10 边, 4 类型) — `models/sast_graph.py`
- [x] StaticPrototypeMatcher — 余弦相似度软匹配, 不分类
- [x] PPM + EdgeConditionedGAT — `models/sast.py`
- [x] PerBinOrderSelector + SparseGaussianReassigner
- [x] 损失函数 (RE_2D + L_physics + L_smooth + L_balance) — `models/sast_losses.py`
- [x] CUDA squeeze kernel (hard + linear) — `deploy/msst_squeeze_*.cu`
- [x] 挤压方式对比 + 速度测试 — `test_squeeze_compare.py` / `plot_squeeze_compare.py`
- [x] 设计文档整合

#### 22.2 待完成

- [ ] MSST 全管线 torch 化 (STFT + IF) — 预期 170x 加速
- [ ] 端到端训练 (全自监督, 无标签)
- [ ] 消融实验: 去掉 RE_2D / L_physics / ENERGY_COMPETITION 边的影响
- [ ] 可解释性 6 面板诊断图生成脚本

### 23. CUDA 加速方案

#### 23.1 当前瓶颈与加速路径

| 步骤 | 当前 | 耗时占比 | 加速手段 |
|------|------|:---:|------|
| STFT | numpy for-loop + np.fft | ~42% | torch.stft (cuFFT) |
| IF 估计 | numpy for-loop + unwrap | ~26% | torch angle/diff |
| IF 迭代 | numpy advanced indexing | ~3% | torch indexing |
| 挤压 | CUDA kernel (已有) | ~26% | 已加速 |
| 其他 | numpy | ~3% | torch |

#### 23.2 CUDA 挤压 kernel

- **`msst_squeeze_hard.cu`**: 硬最近邻 (atomicAdd), 匹配 numpy MSST
- **`msst_squeeze_linear.cu`**: 线性插值 (双 bin), 308x 加速 vs numpy squeeze

**全管线 torch 化预期**: ~170x (50 epoch 训练从 ~30 天 -> ~4.3 小时)

---

## Part VII — 可解释性分析

### 24. 六大诊断维度与集成面板

SAST 的每一个中间量都有明确的物理含义。

#### 24.1 w_i(t)：分量化信任度曲线

w_i ∈ (0,1)，每物理节点每帧的 IF 信任度。

**诊断用途**: 横向对比三节点 w_i 时序、异常检测（w_i 突变）、工况间对比。

#### 24.2 alpha(t)：原型注意力权重

alpha ∈ [0,1]^5，5 个静态原型的 soft 匹配权重。

**诊断用途**: 工况漂移追踪、新工况发现（全均匀 -> 不匹配任何原型）、原型覆盖度评估。

#### 24.3 gate_edge(t)：物理边门控

每条边的逐帧门控值，量化每条物理约束的实时满足程度。

**诊断用途**:
- HARMONIC gate 低 -> 倍频偏差大 -> 滑差/失速
- DRIFT gate 低 -> 涡带与 BPF 解耦 -> 空化风险
- CONDITION gate 低 -> 偏离已知原型 -> 提醒人工检查
- COMPETITION gate 低 -> 能量竞争关系不成立 -> 可能非典型工况

#### 24.4 sigma_sq(eta, b)：逐 bin 核宽图

每个 TF bin 的高斯核宽度。窄核 = GAT 高信任；宽核 = GAT 低信任。

**诊断用途**: 叠加在 STFT 上可视化（红=窄核/谐波, 蓝=宽核/滑差, 透明=无挤压）

#### 24.5 A_ij(t)：GAT 注意力探针

M=10 条边, H=4 头, 逐帧散射注意力权重。

**诊断用途**: 观察 4 个注意力头是否分化出"倍频头"和"工况头"，训练验证（全均匀 -> 没学到, 过分集中 -> 过拟合）

#### 24.6 N*(eta,b)：逐 bin 阶数图（仅推理）

N* ∈ {0,...,N_max}，每 bin 选用的 MSST 阶数。

**诊断用途**: 验证阶数分配的物理合理性（2xBPF 应为低阶, BPF 应为中高阶, LOW_FREQ 应为低/零阶）

#### 24.7 六面板诊断图

```
+-------------------+-------------------+
| 1. TFR_sast       | 2. sigma_sq 叠加图 |  <- "做了什么"
|   增强后时频表示   |   核宽热力图       |
+-------------------+-------------------+
| 3. w_i(t) 三曲线  | 4. alpha(t) 面积图 |  <- "信任了什么"
|   分量信任度时序   |   原型注意力时序   |
+-------------------+-------------------+
| 5. gate_edge(t)   | 6. N*(eta,b) 阶数图|  <- "依据是什么"
|   物理边门控时序   |   逐 bin 阶数分布  |
+-------------------+-------------------+
```

---

## Part VIII — 核心论证回顾

```
问题: MSST 的固定窗 + 硬挤压 + 固定迭代次数 = 三个盲区
      1) 低频滑差分量的漂移 (< 1 STFT bin) 被残酷抹平
         -> 涡带稳定性诊断信息被删除
      2) 极端 chirp / 阶跃 / 低 SNR 区域的硬挤制造阶梯伪影
         -> MSST 迭代会巩固伪影而非消除
      3) 对不同分量、不同工况使用相同的挤压策略
         -> 没有"何时该停"的判断

原因: 水泵水轮机信号中能量展宽有多重来源
      1) 整数谐波: 展宽 = 测量模糊 -> 硬挤消除伪影 YES
      2) 分数阶滑差: 展宽含真实物理漂移 -> 硬挤删除信号 NO
      3) IF 失效区: IF 估计不一致 -> 硬挤制造伪影 NO

方案: 用倍频特征图的图自洽性区分可信与不可信区域
      - 多条边交叉验证 + IF 稳定 -> 图一致 -> w_i↑ -> 硬挤
      - 边在打架 (比值漂移) -> 图矛盾 -> w_i↓ -> 软挤保留
      - 边特征恶化 (sigma_r↑, Corr_E↓) -> 自动检测 IF 失效区 -> 保守

SAST 不是让 TFR "更好看"——是让 TFR "更诚实":
不该挤的能量留着, 因为那本身就是信号。
不确定的区域不动它, 因为 Heisenberg 说你不该知道。
```

---

## 参考文献

1. Pham, D-H., Meignen, S. "Second-Order Synchrosqueezing Transform: The Wavelet Case and Comparisons." IEEE TSP, 2015.
2. Oberlin, T., Meignen, S. "Second-Order Synchrosqueezing Transform or Invertible Reassignment?" IEEE TSP, 2015.
3. Yu, G., Wang, Z., Zhao, P. "Multi-Synchrosqueezing Transform." IEEE TIE, 2019.
4. Colominas, M.A., Meignen, S. "Adaptive Order Synchrosqueezing Transform." 2025.
5. Pham, D-H., Meignen, S. "High-Order Synchrosqueezing Transform." IEEE TSP, 2017.
6. Bao, W., et al. "Application of High-Order Multisynchrosqueezing Transform in Fault Diagnosis." IEEE TIM, 2024.

---

*本文档整合自 `SAST_MSST_必要性分析.md`（理论推导与物理背景）和 `SAST_完整流程说明.md`（流程走线与可解释性分析）。整合日期：2026-07-26。*
