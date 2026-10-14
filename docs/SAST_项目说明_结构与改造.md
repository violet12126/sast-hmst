# SAST 项目说明：结构、端到端流程与本次改造

> 用途：让外部模型快速掌握本项目的模型结构、执行流程，以及一次改造（代号 **B4 拆节点 + A2 可微节点频率**）前后的差异。
> 代码仓库：`D:\aifile\SAST`，分支 `feat/band-split-differentiable-node-if`。
> 改造提交：`942fe85`（主体）、`649f384`（修补遗漏）。改造前的版本即 `942fe85^`。

---

## 1. 项目定位

**SAST = Structure-Aware Synchrosqueezing Transform（结构感知同步压缩变换）**。

把多阶同步压缩变换（MSST）给出的瞬时频率（IF）轨迹，用一张"水泵水轮机物理结构图"组织起来，由图注意力网络（GAT）为每个物理节点、每一帧输出一个可信度 `w_i`，再用 `w_i` 控制时频重排（reassign）时每个频率 bin 的核宽，从而得到**自适应压缩**的时频图。监督信号来自工况标签，但模型**不做工况分类**，只用标签定义对比学习的正负对。

| 项 | 值 |
|---|---|
| 输入 | 单通道振动信号，2000 点，fs = 1000 Hz（2 秒） |
| 数据集 | `5_dataset.npz`，共 9669 样本 |
| 工况类别 | `['空转','低负荷','中负荷','高负荷','抽水']`，样本数 293 / 2637 / 879 / 5274 / 586 |
| 图 | `B, F, T` = batch、1000 频率 bin（0–500 Hz，0.5 Hz/bin）、2000 帧 |

---

## 2. 机器物理与频率约定

| 量 | 值 | 说明 |
|---|---|---|
| 额定转速 `n` | 333.3 rpm | → 转频 `f_rot` = 5.5550 Hz |
| 转轮叶片数 `Z_r` | 9 | |
| 1×BPF | `Z_r·f_rot` = **49.995 Hz** | 与 50 Hz 电网频率相差 0.005 Hz，低于 0.5 Hz 分辨率，二者不可分辨 |
| 2×BPF | 2·`Z_r·f_rot` = **99.990 Hz** | 实测 100.10 Hz（+0.108 Hz），低/中/高/抽水逐样本标准差 0.000 |

⚠ 不要用"空转 ≈ 45.65% 转速 → 22.82 Hz"来区分 1×BPF 与电网频率：**该假设已被实测证伪**，22.82 Hz 处没有任何谱线，空转的主导峰在 45.9–46.4 Hz。详见 §14.4。

**分量分类的正确依据是"是否锁定转速"，不是"机械 / 水力"：**

| 类别 | 判据 | 例子 |
|---|---|---|
| **转速锁定分量** | f = k·`Z_r`·`f_rot`，位置唯一确定 | 1×BPF、2×BPF |
| **工况漂移分量** | 频率由流动不稳定性（水头/负荷/导叶开度）决定，不锁转速 | 涡带、压力脉动、46–49 Hz 宽峰 |

注意：动静干涉（RSI）本身也是流体激励，用"机械/水力"去分类是错的；应看频率是否锁转速。

---

## 3. 核心术语

### 3.1 节点（物理频带）

| 节点 | 频带 | `f_type` | 绝对锚 | `w_i` 允许范围 | 预期带宽 |
|---|---|---|---|---|---|
| LOW_FREQ | 2–18 Hz | HYDRAULIC | 无 | 0.10–0.45 | 12 Hz |
| BPF_DRIFT | 42–48 Hz | BLADE_DRIFT | 无 | 0.10–0.50 | 6 Hz |
| BPF_LINE | 48–53 Hz | BLADE_PASS | 49.995 ± 1.5 | 0.65–0.95 | 1.5 Hz |
| 2xBPF | 90–105 Hz | BLADE_HARMONIC | 99.990 ± 1.5 | 0.65–0.90 | 2 Hz |

`f_type` 中 HYDRAULIC 与 BLADE_DRIFT 被划为**软分量**（频率由流动决定，不该被压缩锐化）。

### 3.2 两类节点频率（本次改造的核心概念）

模型同时维护两个"节点频率"，用途**严格分开**：

| | `IF_obs` | `IF_hat` |
|---|---|---|
| 来源 | MSST 的 `omega_final`，按频带聚合 | **增强后**时频图在频带内的能量质心 |
| 形状 | `[B, N_phys, T]` | `[B, N_phys, T]` |
| 有梯度 | **没有**（只依赖输入信号） | **有**（增强图依赖 `w_i → sigma → 重排`） |
| 用途 | 边特征、边门控、bin→节点分配（**路由**） | `L_anchor`、`L_ratio`（**约束**） |
| 实现 | 类 `MSSTNodeExtractor` | 函数 `band_centroid_if()` |

### 3.3 其他

- **presence（分量存在度）**：`[B, N_phys, T] ∈ [0,1]`，由**原始**（未增强）谱算出的带内能量集中度，用于门控 `w_range` / `balance` / `anchor`。必须用原始谱，否则模型可以靠制造尖锐谱线来抬高它。
- **门控 `gate_edge` / `gate_node`**：由边特征决定的 0–1 权重，控制注意力强弱。

---

## 4. 端到端执行流程（当前版本，逐步骤标注与旧版的差异）

### 第 1 步 STFT —— 无变化

短时傅里叶变换 → `tfr_mag [B, 1000, 2000]`。无可学习参数。

### 第 2 步 MSST 估计瞬时频率 —— 无变化

MSST 多次 lookup 迭代 → `omegas [B, 4, 1000, 2000]`（`num = max(4, N_max)` = 4 层，取**最后一层** `omegas[-1]` 为 `omega_final`）。无可学习参数。

> ⚠ 两处旧表述已更正：① 层数是 **4** 不是 3；② 这不是「多阶同步压缩」——`omegas[k]`
> 是同一个一阶 IF 估计的**固定点 lookup 迭代层**（`omega_{k+1}[η,b] = omega_k[omega_k[η,b], b]`），
> 不是 Taylor 相位导数阶。实测最优层恒为 `omegas[-1]`，即生产路径已在用的那层
> （见 §16.3 与 `papers/SAST_讲解文档.md` §5.1）。

### 第 3 步 节点特征提取 —— **变化最大**

把 0–500 Hz 切成频带，对每个频带、每一帧统计 4 个量：

| 量 | 定义 | 有梯度 |
|---|---|---|
| `node_if` | 带内瞬时频率的代表值 | 否 |
| `node_energy` | 带内总能量，取 `log1p` | 否 |
| `node_bw` | 带内瞬时频率的标准差（带宽） | 否 |
| `node_persist` | 整段信号中有能量的帧占比 | 否 |

**变化 ①：频带数 3 → 4。**
- 旧：`LOW_FREQ [2,18]`、`BPF [42,55]`、`2xBPF [90,105]`
- 新：`LOW_FREQ [2,18]`、**`BPF_DRIFT [42,48]`**、**`BPF_LINE [48,53]`**、`2xBPF [90,105]`

**变化 ②：`node_if` 的聚合方式，带内无权重中位数 → 带内能量加权平均。**
计算式：`node_if = Σ(w·f) / Σw`，`w` = 该 bin 的 STFT 能量；带内无能量时退回算术均值。

### 第 4 步 工况上下文 —— **有修补**

由 4 个节点的能量算 4 维比例向量 `V_obs`，与 5 个离线统计的工况原型 `STATIC_PROTOTYPES [5,4]` 做余弦相似度，`softmax` 得权重 `alpha [B,T,5]`，加权混合可学习嵌入得 `cond_ctx [B,T,32]`。不输出工况类别。

`V_obs = [R_LF, R_BPF, R_2xBPF, log10(E_BPF / E_2xBPF)]`

**修补**：旧代码按位置索引 `E_lin[:,0]/[:,1]/[:,2]`。拆节点后位置语义全变（索引 2 变成 BPF_LINE，2xBPF 在索引 3 完全没被读到）。现改为按节点名取，并把 `E_bpf = E_drift + E_line` 合并回"BPF 族"，使 `V_obs` 的 4 维语义与冻结原型一致，**无需重做 EDA**。

### 第 5 步 边特征 —— **有变化**

每条边每帧算一个 5 维特征，5 维含义随边类型不同：

| 边类型 | 5 维含义 |
|---|---|
| INTEGER_HARMONIC | 实测比值 `r_obs`、`r_obs` 波动、两端能量相关、边权重、两端最小持续性 |
| CONDITION | 工况匹配度 `cond_sim`、其余留空 |
| DRIFT | 能量相关、对数能量比、带宽耦合、边权重、最小持续性 |
| ENERGY_COMPETITION | 负能量相关、对数能量比、比值稳定度、边权重、最小持续性 |

**变化**：边数 10 → 11，倍频边改接。详见第 6 节。

### 第 6 步 PPM + GAT → `w_i` —— 结构不变，规模随节点数变化

**PhysicsPrototypeMemory (PPM)**：
- 物理节点初始表示 = `[频率/BW/能量/持续性]` 线性投影 + 频带类型嵌入（`f_type`）+ 可学习原型嵌入
- 节点 ↔ 原型交叉注意力（带 3.0 的自注意力偏置）→ `proto_context`
- 边门控 `gate_node` 融合：`h = (1-g)·h_raw + g·proto_context`
- 虚拟节点 OP 由 `cond_ctx` 投影得到

**GAT**：2 层、4 头，边特征参与注意力计算。

输出：
- **`w_i [B, 4, 2000]`** — 每个物理节点每一帧的可信度 ∈ (0,1)
- `A_ij [B, 11, 4, 2000]` — 每条边每一帧的注意力

### 第 7 步 `w_i` → 核宽 → 重排 —— 无变化

```
sigma_i = sigma_min + (1 - w_i) × (min(sigma_max, 3.0) - sigma_min)
```
`sigma_min=0.5`、`sigma_max=15.0`，但有效上限被 `sigma_soft_cap=3.0` 截断（否则 `w_i≈0.3` 的保守节点会拿到 `sigma≈10.7` bin，高斯重排把低频脊线摊平）。

每个频率 bin 按 `|f − node_if|` 取 `argmin` 分配给最近节点，用该节点的 `sigma`。

**重排**：把每个 bin 的能量沿 `omega_final` 搬走，落点用宽度 `sigma` 的高斯核铺开。
- 训练：单遍完成，可微
- 推理：按 per-bin 轮数迭代，轮数 = `round(λ_node × ridge_factor)`，由 `SqueezeIterationController` 决定

输出 `tfr_enhanced [B, 1000, 2000]`。

### 第 8 步 损失 —— **改动两项、新增一项**

见第 7 节。

---

## 5. 图结构对照

### 旧图（4 节点 / 10 边）

```
节点: [0] OP, [1] LOW_FREQ, [2] BPF, [3] 2xBPF

OP → LOW_FREQ / BPF / 2xBPF     CONDITION  ×3   w=0.6
LOW_FREQ ↔ BPF                  DRIFT      ×2   w=0.15
BPF → 2xBPF                     HARMONIC   ×1   r=2.0, w=0.8
LOW_FREQ ↔ 2xBPF                COMPETITION ×2  w=0.25
BPF ↔ 2xBPF                     COMPETITION ×2  w=0.20
```

### 新图（5 节点 / 11 边）

```
节点: [0] OP, [1] LOW_FREQ, [2] BPF_DRIFT, [3] BPF_LINE, [4] 2xBPF

OP → LOW_FREQ / BPF_DRIFT / BPF_LINE / 2xBPF   CONDITION  ×4  w=0.6
LOW_FREQ ↔ BPF_DRIFT                            DRIFT      ×2  w=0.15
BPF_LINE → 2xBPF                                HARMONIC   ×1  r=2.0, w=0.8
LOW_FREQ ↔ 2xBPF                                COMPETITION ×2  w=0.25
BPF_DRIFT ↔ 2xBPF                               COMPETITION ×2  w=0.20
```

两处实质变化：
1. **倍频边改接**：`BPF → 2xBPF` 变成 `BPF_LINE → 2xBPF`。只有位置锁转速的那条线才可能是 2xBPF 的基频。
2. **DRIFT 边改接**：`LOW_FREQ ↔ BPF` 变成 `LOW_FREQ ↔ BPF_DRIFT`。漂移边只连两个"不锁转速"的分量。

---

## 6. 为什么要改：旧版的四个问题

### 问题 1：频带粒度混装

旧 `BPF [42,55]` 一个带里装了两类性质完全不同的东西：

| 成分 | 频率 | 锁转速 | 期望核宽 |
|---|---|---|---|
| 转速锁定线 | ~49.995 Hz，宽 ≤1 Hz | 是 | 极窄（该挤） |
| 工况漂移宽峰 | 46–49 Hz，宽 2–3 Hz | 否 | 宽（不该挤） |

但模型对一个带只输出一个标量 `w_i`（进而一个 `sigma`），一个 `sigma` 无法同时"挤窄线"和"留宽峰"，训练只能收敛到折中值。

### 问题 2：唯一的物理锚实际上没起作用

`L_anchor` 的作用是防止两个频带的频率被**同时同比例**高估/低估——比值约束对此完全无感，因为比值是**尺度不变量**（`f_dst/f_src` 在共模缩放下不变）。

但旧版把锚在 `node_if` 上，而 `node_if` **没有 `grad_fn`**（只依赖输入信号）。因此：

- `L_anchor` 在任何频率量上的梯度恒为 **0**
- 模型可以在完全不满足 `f(2xBPF) = 2·f(BPF_LINE)` 的条件下把这一项降到最低

同理 `L_physics` 的 `ℓ` 由 `edge_feats`（源自 `node_if`）构造，也是常数，**唯一可微输入是注意力 `A_ij`**。

数值验证（`scripts/test/probe_physics_grad.py`）：

| 量 | 值 |
|---|---|
| 倍频边 `ℓ` | 0.065 |
| 倍频边 `w·ℓ` | **0.052**（全图 11 条边中最小的） |
| `∂L_physics/∂A_harmonic` | **−0.00093 < 0** |

结论：梯度下降反而**增大**对该边的注意力。也就是说 `L_physics` 没有约束频率，只是产生了一个与样本无关的固定注意力先验。

### 问题 3：`r_obs = 2.13 ≠ 2.00` 是聚合口径造成的假象，不是物理不一致

实测证据（旧聚合口径）：

| 节点 | 带内能量峰 | IF 聚合值（中位数） | 偏差 |
|---|---|---|---|
| LOW_FREQ | 2.00–8.38 Hz | 11.92–14.27 Hz | 被抬到频带几何中心 |
| BPF | 46.42–48.75 Hz | 46.12–48.16 Hz | 被漂移宽峰拉低 |
| 2xBPF | 100.00–100.38 Hz | 97.37–99.46 Hz | 同样被拉向带中 |

中位数与能量无关：宽带（LOW_FREQ 宽 16 Hz）里绝大多数 bin 是噪声，中位数落在频带中间。于是 `r_obs` 被算成 2.13。**问题 3 是问题 2 症状的成因。**

### 问题 4：`compute_V_obs` 的节点索引

拆节点后按位置索引会静默取错节点（见第 4 节第 4 步）。

---

## 7. 损失函数对照

### 旧版：9 项

| 项 | λ | 作用 |
|---|---|---|
| `L_supcon` | 1.0 | 监督对比（Khosla 2020），同工况表示拉近、不同工况推远 |
| `L_entropy`（Rényi 2D） | 0.1 | 鼓励时频图能量集中 |
| `L_physics` | 0.5 | 让注意力与物理一致性对齐 |
| `L_balance` | 0.5 | 防退化 |
| `L_var` | 0.5 | 控制 `w_i` 分布宽度 |
| `L_consistency` | 0.3 | `w_i` 时间平滑 |
| `L_lowfreq` | 0.05 | 低频脊线锐化 |
| `L_range` | 0.3 | `w_i` 逐帧限制在节点允许范围内（按 presence 松弛） |
| `L_anchor` | 0.2 | 绝对频率锚（**作用在 `node_if` 上，无效**） |

### 新版：10 项

| 项 | λ | 变化 |
|---|---|---|
| `L_supcon` | 1.0 | — |
| `L_entropy` | 0.1 | **跳过 HYDRAULIC / BLADE_DRIFT 两个软频带** |
| `L_physics` | 0.5 | 保留（已知只影响注意力先验，不约束频率） |
| `L_balance` | 0.5 | `w_min`/`w_max` 改为按节点的 `[N_phys]` 张量 |
| `L_var` / `L_consistency` / `L_lowfreq` | 0.5 / 0.3 / 0.05 | — |
| `L_range` | 0.3 | — |
| `L_anchor` | 0.2 | **改为作用在可微的 `if_hat` 上** |
| **`L_ratio`** | **0.2** | **新增** |

新增项定义：

```
ell_m = clamp( |f_dst/f_src − r_nom| / r_nom − tol, 0 )        tol = 0.06
L_ratio = Σ_m mean_t[ ell_m × min(presence_src, presence_dst) ]
```

`L_anchor` 与 `L_ratio` 都采用 **flat-bottom hinge**：只有超出容差的漂移才被惩罚，真实工况漂移不罚；且二者互补——比值是尺度不变量（管形状），锚提供绝对参考系（管尺度）。

---

## 8. 本次改造清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `models/sast_nodes.py` | 频带 3 → 4（BPF 拆为 BPF_DRIFT / BPF_LINE） |
| 2 | `models/sast_nodes.py` | `node_if` 聚合：带内中位数 → 带内能量加权平均（numpy 与 GPU 两条路径都改） |
| 3 | `models/sast_nodes.py` | 新增 `band_centroid_if()`，得到可微的 `if_hat` |
| 4 | `models/sast_nodes.py` | 新增 `SOFT_F_TYPES = ('HYDRAULIC','BLADE_DRIFT')` |
| 5 | `models/sast_graph.py` | 图 4 节点 / 10 边 → 5 节点 / 11 边；倍频边改接 BPF_LINE |
| 6 | `models/sast_losses.py` | Rényi 2D 熵跳过软频带；`L_anchor` 改作用在 `if_hat`；新增 `L_ratio` |
| 7 | `models/sast.py` | `compute_V_obs` 按新节点布局取能量（修补遗漏）—— **随后被 §15.7 取代**：`V_obs` 4 → 6 维、`expm1` 修正、`STATIC_PROTOTYPES` 按 4 节点重算 |
| 8 | `sast_config.py` / `train_sast.py` | 新增超参 `lambda_ratio=0.2`、`ratio_tol=0.06` |
| 9 | `scripts/**` | 4 个按 3 节点硬编码的脚本改为从 `PUMP_TURBINE_REGIONS` / `PHYSICS_EDGES` 导出；`analyze_band_compressibility.py` 的测量频带随 B4 一并拆成 4 段 |

> **本节记录的是 B4 拆节点那一轮。其后的边界解耦 / 窗软化 / 均衡采样 / 原型重算 / 频率约束降级见 §15。**

---

## 9. 代码文件地图

| 文件 | 内容 |
|---|---|
| `models/tfr.py` | MSST / WSST2 / SST2 等 TFR 基座 + CUDA STFT wrapper |
| `models/sast_nodes.py` | `FreqRegion` / `PUMP_TURBINE_REGIONS`（节点定义）、`band_centroid_if`（可微 IF）、`MSSTNodeExtractor`（节点特征提取，含 numpy 与 GPU 两条路径） |
| `models/sast_graph.py` | 节点索引、`PHYSICS_EDGES`（图定义）、`compute_edge_features`（numpy）与 `compute_edge_features_torch`（GPU 向量化） |
| `models/sast.py` | `StaticPrototypeMatcher`、`PhysicsPrototypeMemory`(PPM)、`EdgeConditionedGAT`、`SqueezeIterationController`、`SparseGaussianReassigner`、主模块 `SAST.forward` |
| `models/sast_losses.py` | 全部损失函数 + `FreqEncoder` + `total_sast_loss` |
| `sast_config.py` | `SastConfig`（所有超参的唯一来源）+ `set_seed` |
| `sast_model.py` | `create_model` / `create_freq_encoder` / `load_checkpoint` |
| `train_sast.py` | 训练入口（CLI） |
| `deploy/sast_export_model.py` | ONNX / TensorRT 导出（节点数已泛化） |
| `scripts/test/` | `probe_physics_grad.py`（梯度流向探针）、`smoke_anchor_presence.py`（锚 / presence 冒烟）、`benchmark_sast.py` |
| `scripts/analysis/` | `probe_node_if.py`（节点 IF 口径体检）、`analyze_band_compressibility.py`（频带可压缩性）、`summarize_sast_per_class.py`、`check_spectral_bands.py`、`probe_harmonic_comb.py` |

---

## 10. 关键超参

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| `fs` | 1000 | `d_h` | 96 |
| `max_len` | 2000 | `n_heads` | 4 |
| `sigma_min` | 0.5 | `n_layers` | 2 |
| `sigma_max` | 15.0 | `msst_num` | 4 |
| `sigma_soft_cap` | 3.0（代码内常量） | `d_cond` | 32 |
| `n_sqz_max` | 4 | `f_type_embed_dim` | 16 |
| `epochs` | 20 | `ppn_temperature` | 0.08 |
| `batch_size` | 2 | `prototype_temperature` | 0.1 |
| `lr` | 1e-3 | `dropout` | 0.1 |
| `grad_clip` | 1.0 | `seed` | 42 |
| `presence_frac` | 0.1 | `w_low_min` | 0.05 |
| `val_split` | 0.15 | | |

⚠ `batch_size` 有**配置分歧**：`sast_config.py:36` 默认 2，`train_sast.py:79` 的 CLI 默认 8（help 明确写"SupCon 需 batch 内同工况正样本对, 建议 >=8"）。正常训练走 CLI，实际是 8；直接构造 `SastConfig()` 会落到 2，而 B=2 会让 SupCon 在约 61.5% 的 batch 上不产生任何梯度。详见 §14.3。

---

## 11. 验证结果（改造后，均为实测）

| 检查 | 结果 |
|---|---|
| numpy 与 GPU 两条 IF 聚合路径 | 一致到 2 位小数：2.35 / 47.21 / 49.19 / 99.48 Hz |
| LOW_FREQ 的 IF | 旧口径 11.9–14.3 Hz → 新口径 **2.37 Hz**（落在能量实际所在处） |
| 端到端 forward + backward | 形状正确：`w_i [B,4,T]`、`A_ij [B,11,4,T]`、`tfr_enhanced [B,1000,T]` |
| `L_anchor` flat-bottom | 质心在容差带内梯度为 0；越界后非零（1.80 Hz→0.307、2.55 Hz→1.058） |
| `L_ratio` 可微性 | `r_obs=2.133`（相对偏差 6.65% > 6%）时 `dL_ratio/dκ = +4.94e-2 ≠ 0` |
| `L_physics` 对频率量 | 梯度仍恒为 0（这是它本身的性质，未改变） |

合成验证信号：3 / 47 / 50 / 100 Hz 四条正弦线叠加。

> ⚠ **两处引用说明**（防止后续被误读）：
> 1. 上表 49.19 Hz 是**合成信号**上 `IF_obs` 的 numpy / GPU 两条路径一致性测试值，**不是**实测数据的 `IF_hat` 串扰量。实测 `IF_hat[BPF_LINE]` 的五类质心为 49.294–50.033 Hz（改前）/ 49.807–50.298 Hz（改后），见 §15.3。
> 2. 本表的数值均为**改前**口径。§15 引入过渡带加权后，提取器/质心/损失的绝对数值均已变化，本表数字不可与新口径直接对照。

---

## 12. 已知遗留问题

> 状态截至 §15 那一轮之后。已解决项保留在表中并标注修订位置，便于追溯。

1. **旧 checkpoint 不兼容**，必须重训。（仍然成立，且原因增加：`PROTOTYPE_DIM` 4→6）
   `C_prior_logit`、`prototype_embed`、`f_type_embed` 的维度都随节点数变化。
2. ~~**工况原型仍基于旧 3 频带划分离线统计**~~ → **已在 §15.7 解决**。
   按 4 节点重做，`V_obs` 4 → 6 维（新增 `log10(E_DRIFT/E_LINE)` 判别子），并修正了 `exp(log1p(·)) = 1+x` 的换算错误与原型/在线归一化不一致。新表见 §15.7。
3. ~~**`L_physics` 是否保留待定**~~ → **已在 §15.5 定案（方案 A）**。
   `L_physics` / `L_anchor` / `L_ratio` 三项一并降级为诊断量，不进 `L_total`；`--freq_constraints_active` 可一键复现原行为。
4. **论文文档已过时**（**未解决，本轮范围外**）。
   `docs/SAST_汇报与论文初稿.docx` 的 §4.2.4(6) 及第 4/5 章中"4 节点 / 10 条边"的描述与数字（如 `∂L/∂A = −0.00087`）需按新拓扑更新（新值 −0.00093，`ℓ=0.065`、`w·ℓ=0.052` 不变）；还需补 §15 的边界变更、`bw_expected` 新值、P×K 与 6 维 `V_obs`。
   另有 `scripts/doc/build_sast_docx.js:560`（"46.4 Hz 与理论值 2×Z_r×f_rot,空转 = 45.6 Hz 吻合"）承袭已被证伪的 45.65% 前提，需一并改写。
5. **待定设计项**：节点 IF 的聚合口径、频带粒度，记录在 `待补充实验清单.md`（该文件 .gitignore，仅本地）。
   频带粒度已在 §15.2 定案（48.5 + 半开 + 单位分解过渡带）。
6. ~~**"空转 45.65% 转速 → 22.82 Hz"的说法需从论文与脚本注释中清除**~~ → **脚本侧已在 §15.8 清除**。
   `check_spectral_bands.py` 与 `probe_harmonic_comb.py` 已删 `F_ROT_IDLE` / `F_IDLE_ROT` 及其梳齿候选；论文草稿侧仍待处理（见第 4 条）。
7. **ONNX 导出脚本缺依赖**（新发现）。
   `deploy/export_sast_onnx.py` 在 `torch.onnx.export` 处报 `ModuleNotFoundError: No module named 'onnxscript'`（新版 torch 的导出器依赖该包，`my_work` 环境未装）。导出模型的**数值**已另行验证与训练侧逐位一致（§15.8），仅 ONNX 写盘这一步被阻塞。修复：`pip install onnxscript`。

---

## 13. 复现命令

```bash
# 图结构冒烟（输出 5 节点 / 11 边）
python -m models.sast_graph

# 损失冒烟（输出 10 项损失 + 逐节点 presence / if_hat）
python -m models.sast_losses

# 锚 / presence / 比值约束冒烟
python scripts/test/smoke_anchor_presence.py

# L_physics 梯度流向探针
python scripts/test/probe_physics_grad.py --r-obs 2.13

# 节点 IF 聚合口径与存在度体检（需 5_dataset.npz）
python scripts/analysis/probe_node_if.py --per-class 12
```

运行脚本使用 conda 环境 `my_work`；Windows 下打印中文需 `PYTHONUTF8=1`。

---

## 14. 补充核对：三处细节的代码 / 数据级答案

### 14.1 `band_centroid_if()` 是否只在硬边界内求能量加权频率均值

**是，完全如此。** 实现在 `models/sast_nodes.py:143-187`，逐节点做：

```python
m    = (freqs >= r.f_min) & (freqs <= r.f_max)      # 硬边界, 两端闭区间
E    = tfr.abs().pow(2)                              # 能量 = 幅值平方
cent = (E * freqs[m]).sum(dim=band) / E.sum(dim=band)  # 逐帧能量加权频率均值
```

- **矩形硬边界**：无窗函数、无软锥度、无过渡带、无重叠。
- **无任何可学习参数**。它只是把 `tfr_enhanced` 映射成可微频率的一个量；`E` 对 `tfr` 是二阶齐次，梯度良态。
- 带内 `ΣE ≤ 1e-12` 时退化为频带中心 `(f_min + f_max)/2`。
- ⚠ **48.0 Hz 这一个 bin 同时属于 BPF_DRIFT 与 BPF_LINE**（前者 `f_max=48`，后者 `f_min=48`，两端都是闭区间；频率轴步长 0.5 Hz，48.0 是真实 bin）。因此它在两个节点里各被计一次。量级影响很小，但严格说节点划分在该 bin 上不互斥。

### 14.2 `SparseGaussianReassigner` 中是否另有可学习参数改变重排落点频率

**没有。落点频率完全不可学，只有一个通道能影响重排，而它只改变核宽、不改变落点。**

| 量 | 来源 | 可微 |
|---|---|---|
| 落点 `omega_hat_int` | `omegas[:, -1]`（MSST 输出）→ `.round().long()` | **否**（离散整数索引，无 `grad_fn`） |
| `sigma` | `w_i` → N(0,σ) 核宽 | 是 |
| `lambda_i`（推理轮数） | `round(w_i·N_max).long()` | 否（整数） |
| `ridge_factor` | `argmin` 掩码 + `node_if` | 否 |

关键代码 `models/sast.py:788-789`：

```python
omega_final   = omegas[:, -1, :, :].float()                        # MSST 估计, 无 grad_fn
omega_hat_int = (omega_final - 1.0).clamp(0, F - 1).round().long()  # 整数索引
```

`ReassignerFunction.backward` 一律 `return None, grad_sigma, None, None, None`（见 `models/sast.py:708/716/741`）：梯度**只回传给 `sigma`**，对 `omega_hat_int` 返回 `None`，**对 `tfr_weighted` 也返回 `None`**。`SqueezeIterationController` 里没有任何 `nn.Parameter`。

因此：**`w_i` 经由 `sigma_i` 改变的只是高斯核的宽度（能量散布范围），落点的位置完全由 MSST 固定。若把"落点频率"理解为高斯核中心，则没有任何可学习参数参与改变它。**

这与 §3.2 的 `IF_obs` 是一体两面：**路由（bin→节点）与重排落点都来自 MSST，都不参与梯度。**

### 14.3 `L_supcon` 的输入张量形状（batch_size = 2）

**先池化成整段向量，再进 MLP。`batch_size = 2` 时输入 `supcon_loss` 的就是 `z_freq [2, 128]`——只有 2 个向量。**

`models/sast_losses.py:62-72`（`FreqEncoder.forward`）：

```python
feat = tfr.mean(dim=-1)          # [B, F]  —— 沿整条时间轴 GlobalAvgPool
feat = torch.log1p(feat)
z    = MLP(feat)                 # [B, 128], LayerNorm 替代 BatchNorm
z    = F.normalize(z, dim=-1)
```

- **没有在时间帧维度切子窗口，也不存在 `[B·N_win, D]` 这种构造**；全代码库无任何窗口切分逻辑。每个 2 秒样本 → 1 个 128 维向量。
- 后果：`supcon_loss`（`sast_losses.py:96-127`）在 B=2 时，若两样本工况不同，`has_pos.any()` 为 `False` → **直接返回 `0.0`，梯度为零**（`sast_losses.py:119-121`）。
- 按训练集分布（空转 234 / 低负荷 2110 / 中负荷 703 / 高负荷 4219 / 抽水 469，共 7735），随机 2 样本同工况的概率 = `Σ p² ≈ 0.385`。即 **B=2 时约 61.5% 的 batch 完全不产生对比梯度**。
- 配置分歧：`sast_config.py:36` 默认 `batch_size = 2`，而 `train_sast.py:79` 的 CLI 默认是 8。按 CLI 走是 8 没问题；任何直接构造 `SastConfig()` 的入口会落到 2。

### 14.4 空转样本的实际谱峰：不是 22.82 Hz，也不是 50 Hz

**结论：空转（293 个样本）的主导谱峰在 45.90–46.39 Hz。22.82 Hz 处没有谱线。**

本项目自己的普查结果（`results/band_census/`，仅用 train 集）与本次对全量 293 个样本的复测一致。

`peak_census.csv` — 空转的候选峰：

| 候选 | 逐样本峰位中位 | 逐样本 std | 相对电平 | 存在率 |
|---|---|---|---|---|
| 45.90 Hz | **46.39 Hz** | 0.611 Hz | **0.00 dB（最强）** | 1.00 |
| 1.95 Hz | 1.95 Hz | 0.937 Hz | −6.99 dB | 1.00 |
| 15.14 Hz | 13.18 Hz | 2.597 Hz | −11.36 dB | 1.00 |
| 100.10 Hz | 100.10 Hz | 1.322 Hz | −12.94 dB | 1.00 |

`bp_line_check.csv` — 空转在 50.00 Hz 处：窗内峰在 **45.898 Hz**，50.00 Hz 电平比它低 **9.87 dB**。

`comb_check.csv` 直指问题：

| 理论分量 | 理论 Hz | 最近实测峰 | 偏差 |
|---|---|---|---|
| 空转 `Z_r·f_rot` | 22.823 | **15.137** | **−7.69 Hz** ← 无信 |
| 空转 `2·Z_r·f_rot` | 45.645 | **45.898** | +0.25 Hz ← 吻合 |

本次复测（全量 293 = train 234 + test 59）：

| 检查 | 结果 |
|---|---|
| 1–120 Hz 平均谱中 ≥8% 峰值的局部极大 | **只有 46.0 Hz** |
| 逐样本全局最强峰（1–120 Hz） | 240/293 落在 45–48 Hz（45–46: 57，46–47: 114，47–48: 60）；44–45: 1；48–49: 9；50–51: 9 |
| [45,47] Hz 窗内逐样本峰 | 中位 **46.00 Hz**，10–90 分位 [45.50, 47.00] |
| [20,26] Hz 窗内逐样本峰 | 中位 22.50 Hz，10–90 分位 [22.00, 23.50] —— **近似均匀 ⇒ 噪声底** |
| [20,26] Hz 能量占比 | 中位 **1.16%** |
| [48,52] Hz 能量占比 | 中位 1.72%，但 50.00 Hz 处幅度中位 **344.9**，仅为 46 Hz 峰（4679）的 **1/14** |
| [1,18] Hz 能量占比 | 中位 10.33% |

**所以"空转 ≈ 45.65% 额定转速 → 1×BPF = 22.82 Hz"不成立。**
该假设预言 1×BPF 在 22.82 Hz（通常应是该工况最强的 RSI 线），但那里完全无峰；而它顺带预言的 2×BPF = 45.65 Hz 恰好落在实测主峰 45.9 Hz 上。**这个百分比看起来是从观测到的 ~46 Hz 峰反推出来的，不是在 22.8 Hz 上测出来的。**（`check_spectral_bands.py:23,79` 与 `probe_harmonic_comb.py:21` 原本就把它写作"假设"。）

数据支持的读法是：空转的主导 45.9–46.4 Hz 是一个**工况漂移分量**。

- 空转类内逐样本 std **0.611 Hz**（若是转速/电网锁定线，应像 100.10 Hz 那样为 0.000）；
- 同一候选峰在各类间中位位置随工况移动：**46.39（空转）→ 46.88（高负荷）→ 47.36（中负荷）→ 47.85（抽水）→ 49.80（低负荷）** Hz；
- 46.39 / 5.5550 = **8.35 阶**，在额定转速下不是整数阶，因而不是转速锁定谐波；
- 它落在 **BPF_DRIFT [42,48]** 内，正是 B4 拆出来的那个节点。

**这构成对 B4 拆分的一次独立验证**：空转这个工况的主能量恰好是"漂移族"；而旧版把 [42,55] 合成一个带时，它与 50 Hz 尖线共用一个 `sigma`，物理上无法同时兼顾。

顺带两条对照事实：

1. 上述候选峰的中位位置**跨过了 48 Hz 边界**（空转 46.39 / 高负荷 46.88 / 中负荷 47.36 / 抽水 47.85 落在 BPF_DRIFT；低负荷 49.80 落在 BPF_LINE）。这意味着 `L_anchor`（BPF_LINE 49.995±1.5 → 窗口 [48.5, 51.5]）在空转、中负荷、高负荷、抽水下**根本不会被激活**（主峰在窗外），只有低负荷会真正触发它。设计上这由 presence 门控 + flat-bottom 正确处理，但评估锚的有效性时必须意识到它只在部分工况生效。
2. **100.10 Hz 是全数据集最稳的谱线**（低/中/高/抽水的逐样本 std = **0.000 Hz**），且在中负荷与高负荷是最强分量（0.00 dB）。理论 2×BPF 为 99.990 Hz，实测偏移 +0.108 Hz。因此在 0.5 Hz 分辨率下，"100.10 Hz 是机器的 2×BPF"与"100.10 Hz 是 50 Hz 的二次谐波"同样无法区分——**`L_anchor` 被满足并不构成"模型找到了转子 2×BPF"的证据**，这一点与 §6 问题 2 的结论一致。
---

## 15. 本轮改造：边界解耦 / 窗软化 / 均衡采样 / 原型重算 / 频率约束降级

§14 暴露了一个结构性问题：**48.0 Hz 这条硬边界同时压在两个峰之间和 bin 语义的两端**。本节记录本轮如何消除它，以及随之重算的所有量。所有数字均为实测。

### 15.1 问题：48.0 Hz 硬边界的三重后果

1. **bin 48.0 被双重计入。** 全仓库 7 处区域掩码都是两端皆闭的 `(freqs >= f_min) & (freqs <= f_max)`，`BPF_DRIFT[42,48]` 与 `BPF_LINE[48,53]` 在 48.0 处重叠一格。后果是 `Σ E_n > E_total`，而 `compute_V_obs` 的比例运算依赖 `Σ E_n = E_total`。
2. **空转的 `BPF_LINE` 带内出现“单峰劈裂”伪峰。** 实测该窗内 `E峰f = 48.000 Hz` —— 这不是 50 Hz 线，而是 46.4 Hz 漂移峰的右翼堆在**窗口左边缘**形成的伪极值。后果：空转的 `presence_e(BPF_LINE) = 0.317`，**全类最高**，与物理直觉完全相反（低负荷最低 0.082）。
3. **泄漏是单向的，硬边界却对称地切。** `σ_drift ∈ [1.75, 2.75]` bin（std 0.875–1.375 Hz），单个 46.9 Hz 源在 σ=2.75 时 **18.0% 的能量落在 48 Hz 以上**；而 `σ_line ∈ [0.625, 1.375]` bin（std 0.31–0.69 Hz）几乎不向左漏。结果是 `IF_hat[BPF_LINE]` 被系统性压低（偏置约 −1.3 Hz）。

### 15.2 方案：半开重划 + 共享内边单位分解过渡带

**边界**（只动内部共享边，外缘全不动）：

| 区域 | 原 | 新 |
|---|---|---|
| `BPF_DRIFT` | `[42.0, 48.0]` | `[42.0, 48.5)` |
| `BPF_LINE` | `[48.0, 53.0]` | `[48.5, 53.0)` |

`LOW_FREQ[2,18]` / `2xBPF[90,105]` 与间隙侧边界保持硬 0/1。选 48.5 的理由：它是 σ 分配的自然分割点 `(node_if_DRIFT + node_if_LINE)/2 ≈ 48.45 Hz`（`sast.py` 的 `i_star = dist.argmin(dim=1)`），把静态边界对齐到动态分配点，两者不再互相矛盾。

**权重函数**（`models/sast_nodes.region_weights`，单一真源）：

```python
SOFT_SPLIT_HZ = 0.5                       # 共享内部边的过渡带半宽 (Hz)
w_lower(f) = 1 - 0.5 * (1 - cos(pi * clamp((f - (b-h)) / (2h), 0, 1)))
w_upper(f) = 0.5 * (1 - cos(pi * clamp((f - (b-h)) / (2h), 0, 1)))
```

在共享边 `b = 48.5` 两侧用互补升余弦过渡（`[48.0, 49.0]`），使 **`Σ_n w_n(f) ≡ 1`**（单位分解）。

**为什么是单位分解而不是两侧各自软窗**：`node_energy` 与 `compute_V_obs` 的比例运算依赖 `Σ E_n = E_total`。若两侧各自升余弦，47.5–48.5 Hz 的能量会被两边同时打折，`R_LF+R_DRIFT+R_LINE+R_2x` 不再为 1，需额外重归一化；互补过渡带从构造上免掉这件事。

**为什么外缘不加过渡带**：抽水/中负荷的真实漂移峰实测达 `E峰f = 47.583–47.708 Hz`。若在 `BPF_DRIFT` 上沿加对称升余弦会削掉真实信号。

**7 处掩码副本收敛到一处**：`band_centroid_if`、numpy 提取器、GPU 提取器、`renyi_2d_loss`、`node_presence`、`deploy/sast_export_model.py`、`get_region_bounds`。过渡带位置取自**规范表** `PUMP_TURBINE_REGIONS` 而非调用方传入的 `regions`，否则只传单个区域的调用方（如 Rényi 逐区域遍历）会因 `range(N-1)` 为空而丢掉过渡带。

### 15.3 边界修复的实测效果（改前 / 改后）

`python scripts/analysis/probe_node_if.py --per-class 12`，真数据（`--legacy` 复现改前口径）：

| 检查项 | 改前 | 改后 |
|---|---|---|
| 空转 `BPF_LINE` 的 `E峰f` | **48.000 Hz**（劈裂伪峰） | **49.000 Hz** ✅ |
| 空转 `BPF_LINE` 的 `E质心` | 49.294 Hz | 49.807 Hz |
| 空转 `presence_e(BPF_LINE)` | **0.317**（全类最高） | 0.135（降 57%，但仍为全类最高，见下） |
| `BPF_LINE` 五类 `E质心` 跨度 | 49.294–50.033（0.739 Hz） | 49.807–50.298（0.491 Hz） |
| 五类 `abs(锚 − E峰f)` | 1.995 / 0.912 / 1.870 / 1.870 / 1.370 | 0.995 / 0.537 / 0.995 / 0.953 / 0.870 |
| `BPF_DRIFT` 五类 `E峰f` | 46.417 / 47.667 / 47.583 / 46.625 / 47.708 | **完全不变** ✅ |
| `presence_e(2xBPF)` | 0.097 / 0.319 / 0.338 / 0.343 / 0.335 | 0.074 / 0.238 / 0.252 / 0.256 / 0.250 |

要点：

- **`BPF_DRIFT` 侧逐位不变** —— 过渡带没有削掉任何真实漂移峰，验证了“外缘不加软化”的选择是对的（47.708 Hz 在过渡带内仍保留权重 1.0，因为过渡带是 `[48.0, 49.0]`，与 47.708 无交集）。
- **五类 `abs(锚 − E峰f)` 全部落入 ±1.5 Hz 容差** —— 改前空转/中负荷/高负荷都越界，改后全部在带内。这**加强**了 §15.5 的结论：锚不是“按类失效”，而是结构性失效。
- **`presence_e(BPF_LINE)` 空转仍是全类最高**（0.135）。这一项**没有完全达成**预期的“不再全类最高”。原因是诚实的：空转的 `BPF_LINE` 窗内没有独立的 50 Hz 线，但 46.4 Hz 漂移峰的右翼确实是真实存在的能量，过渡带只能把它压掉约一半。若把 `SOFT_SPLIT_HZ` 加宽到 1.0 Hz 可进一步压低，但代价是抽水 47.708 Hz 的真实峰权重从 1.0 掉到 0.973 —— 属于拿真实信号换指标，不做。

**权重分区自检**（`Σ_n w_n(f)`）：

| 检查 | 结果 |
|---|---|
| 覆盖集 `[2,18) ∪ [42,53) ∪ [90,105)` 上 `Σ_n w_n(f)` | `1.000000000`（逐 bin） |
| 间隙上 `Σ_n w_n(f)` | `0` |
| bin 48.0 的归属 | `BPF_DRIFT = 1.0 / BPF_LINE = 0.0`（原为两侧各 1，双计） |
| 过渡带 `[48.0, 49.0]` 两侧权重 | 48.5 → 0.5/0.5；49.0 → 0.0/1.0 |
| numpy / torch 两条路径 | 逐位一致 |

### 15.4 `bw_expected` 标定

`ridge_decay = exp(-(|f - if| / (2·bw_expected))² / 8)`，bw 偏大 → 远端 `ridge_factor ≈ 1` → 挤压参与因子失效。四个值全部按**改后**的实测带内 IF 标准差（五类均值）重新标定：

| 区域 | 原值 | 实测均值（改后） | 新值 |
|---|---|---|---|
| `LOW_FREQ` | 12.0 | 4.53 | **4.5** |
| `BPF_DRIFT` | 6.0 | 1.70 | **1.7** |
| `BPF_LINE` | 1.5 | 1.03 | **1.0** |
| `2xBPF` | 2.0 | 4.52 | **4.5** |

`ridge_factor` 实测剖面（典型 `node_if = [8.0, 46.1, 49.9, 99.5]`）：

| f (Hz) | 30 | 40 | 55 | 75 | 90 | 100 | 105 | 110 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| 改前 | 0.799 | 0.968 | 0.697 | 0.150 | 0.494 | 0.998 | 0.790 | 0.423 | 0.150 |
| 改后 | **0.150** | 0.669 | 0.444 | 0.396 | **0.870** | 1.000 | **0.954** | **0.844** | 0.150 |

- 30 Hz（无物理内容的间隙）从 0.80 掉到 floor 0.15 ✅，40 Hz 从 0.97 掉到 0.67；
- `2xBPF` 带内（90/105/110 Hz）从 0.49/0.79/0.42 升到 0.87/0.95/0.84 —— 改前 **2xBPF 自己的带边只拿到 46% 参与度**，这是原值 2.0 偏小造成的；
- ⚠ 副作用：75 Hz 从 0.150 升到 0.396（`2xBPF` 取 4.5 使 53–90 Hz 空带的参与度被抬起）。取实测均值的理由是标定方法可复现；若改取“分量自身宽度”（约 2.0）可消除该副作用但失去可复现口径。

### 15.5 方案 A：频率约束降级为诊断（`freq_constraints_active = False`）

`L_physics` / `L_anchor` / `L_ratio` **照旧计算**并写入 `losses_dict`，但不进入 `L_total` 的反向图。`total_sast_loss` 新增形参 `freq_constraints_active: bool = False`，`losses_dict` 新增 `w_physics` / `w_anchor` / `w_ratio` 三个**生效乘数**（降级时为 0.0），日志自证。`train_sast.py --freq_constraints_active` 一个开关即可复现消融。

**生效项 7 项**：`L_supcon`、`RE_2D`、`L_balance`、`L_var`、`L_consistency`、`L_lowfreq`、`L_range`。**报告项仍 10 项**。

理由（3 条）：

1. 重排落点 `omega_hat_int` 完全不可学 —— `ReassignerFunction.backward` 三条路径皆返回 `None`，只有 σ 有梯度。因此基于 `IF_hat` 的锚/比值约束无法校准瞬时频率，只能经 `w_i → σ_i` 反向操纵**测量窗口内的能量质心**，属伪物理约束。
2. 实测五类 `BPF_LINE` 带内质心仅跨 0.74 Hz（49.294/50.033/49.801/49.517/49.945），**窄于 ±1.5 Hz 容差**，锚值 49.995 天然落在该范围内 —— 对**任何**工况都不触发，是结构性失效而非按类失效。§15.3 显示改后全部五类 `abs(锚 − E峰f) < 1.5 Hz`，结论不变。
3. 保留诊断输出正是论文价值：`if_hat_{name}` 与 `abs(anchor − if_hat)` 每 epoch 照旧记录，可实证给出“锚在不被约束时也天然满足”。

`train_sast.py` 每 epoch 现在会打印一段 `[诊断]` 块（`anc` / `rat` 值与三个生效乘数 `wλ(p/a/r)`，以及逐节点 `if_hat` / `presence` / `abs(锚 − if_hat)`）。首次实跑（`--epochs 1 --max_samples 900`，降级态）的结果即为方案 A 的直接证据：

| 节点 | `if_hat` | `presence` | `abs(锚 − if_hat)` |
|---|---|---|---|
| LOW_FREQ | 10.093 Hz | 0.146 | —（无锚） |
| BPF_DRIFT | 45.990 Hz | 0.080 | —（无锚） |
| BPF_LINE | 49.535 Hz | 0.048 | **0.460 Hz** |
| 2xBPF | 99.409 Hz | 0.145 | **0.581 Hz** |

两条带锚的节点在**完全没有约束梯度**的情况下，1 个 epoch 后 `if_hat` 距锚仅 0.46 / 0.58 Hz，远在 ±1.5 Hz 容差之内 —— 即 `L_anchor` 在这份数据上不可能产生非零贡献。

### 15.6 P×K 类别均衡采样（P=2, K=5 → B=10）

**问题**：训练集各类 196 / 1786 / 587 / 3604 / 402（val_split=0.15），而训练循环是 `torch.randperm` 均匀切片。SupCon 需要每个类在 batch 内有正样本对，而 `torch.randperm + B=8` 下各类“batch 内 <2 个样本 ⇒ 无正样本对”的 batch 占比实测为：

| class | 空转 | 低负荷 | 中负荷 | 高负荷 | 抽水 |
|---|---|---|---|---|---|
| 池大小 | 196 | 1786 | 587 | 3604 | 402 |
| 旧 `randperm`/B=8 | **97.6%** | 32.8% | **84.8%** | 2.1% | **91.6%** |
| 新 P×K | **0.0%** | 0.0% | 0.0% | 0.0% | 0.0% |

**实现**：`sast_data.build_balanced_batch_indices(y, p, k, n_batches, rng)` 返回 `[n_batches, p·k_eff]` 的 `int64` 索引矩阵。类内**不放回**抽 p 个（放回会让同一样本自配成假正样本对）；类池 < p 时退化为放回；batch 内索引最后打乱。`train_sast.py` 每 epoch 用 `np.random.default_rng(seed + epoch)`，`n_batches = max(1, N_train // (p·k))`。

**代价与收益**：

| | 旧（B=8） | 新（P×K=2×5） |
|---|---|---|
| batch 大小 | 8 | 10 |
| batch 数 / epoch | 821 | 657 |
| 样本覆盖 / epoch | 6568 / 6575 | 6570 / 6575 |
| 各类每 epoch 出现次数 | 按池比例 | 各 1314 次 |
| 过采样倍数（空转/低/中/高/抽水） | 1× | **6.70× / 0.74× / 2.24× / 0.36× / 3.27×** |

K=5 恰等于类别数 ⇒ 每个 batch 必含全部 5 个类（`k_eff = min(k, 非空类数)` 的退化情形）；若 `K<5` 则每 batch 随机抽 K 个类。

**注意**：`sast_config.batch_size` 的默认值此前是 2，与 `train_sast.py` 的 CLI 默认 8 长期分歧；已统一为 8，并注明它只在 `--balanced_p 0` 时生效。`balanced_p < 2` 时启动打印 WARN（P=1 时每类在 batch 内仅 1 个样本，无正样本对，等于没修）。

### 15.7 工况原型重算（5×4 → 5×6）

**旧表的三个缺陷**（均以 `compute_V_obs` 为同一计算路径重算后根治）：

| 缺陷 | 证据 | 修正 |
|---|---|---|
| **归一化不一致** | 旧原型前三列之和 0.273/0.911/0.620/0.819/0.806，而在线的 `E_total = Σ 带能量` 使 `V_obs` 前三列恒和为 1.0。余弦相似度**不是**对“R 块 vs log 比值列”相对权重不变的，故 `alpha` 被系统性扭曲 | 重算时统一走 `compute_v_obs` |
| **`exp(log1p(·)) ≠ ·`** | `E_lin = torch.exp(node_energy)`，而 `node_energy = log1p(band_energy)` → `E_lin = 1 + band_energy`，弱带被 `+1` 地板抬高 | 改 `torch.expm1`（`log1p` 的精确逆） |
| **好耦合被丢掉** | 为对齐旧原型，把 `BPF_DRIFT`/`BPF_LINE` 重新合并为 `E_bpf`，恰好抹掉 B4 拆分最有判别力的特征 | 拆开，新增第 5 维 |

**新 V_obs（6 维）**：

```
V_obs = [ R_LF, R_DRIFT, R_LINE, R_2x,
          log10(E_DRIFT / E_LINE),              # 新增: 漂移/线 判别子
          log10((E_DRIFT + E_LINE) / E_2x) ]    # 保留: 原 4 维里的谐波比值
```

保留原比值是因为 `log10(E_bpf/E_2x)` 编码了 `BPF_LINE → 2xBPF` 的整数谐波边（`r_nom = 2.0`）。`PROTOTYPE_DIM` / `N_PROTOTYPES` 由 `STATIC_PROTOTYPES.shape` 派生；`obs_proj` 在 `forward` 中从未被调用（`alpha` 直接来自 `cosine`），故加宽无额外风险。

**新表**（`scripts/analysis/build_static_prototypes.py`，train 划分 N=6575，class 0 全量 196、其余各类抽 400，逐列取中位数后把 R 块重新归一化到 1）：

| cls | 工况 | R_LF | R_DRIFT | R_LINE | R_2x | log10(E_DRIFT/E_LINE) | log10((E_DRIFT+E_LINE)/E_2x) |
|---|---|---|---|---|---|---|---|
| 0 | 空转 | 0.138863 | 0.671624 | 0.112759 | 0.076755 | +0.738565 | +1.014024 |
| 1 | 低负荷 | 0.078291 | 0.414625 | 0.494852 | 0.012232 | −0.080094 | +1.868020 |
| 2 | 中负荷 | 0.098077 | 0.209527 | 0.118380 | 0.574016 | +0.181772 | −0.205221 |
| 3 | 高负荷 | 0.016501 | 0.093141 | 0.037490 | 0.852869 | +0.389718 | −0.778678 |
| 4 | 抽水 | 0.049762 | 0.455254 | 0.345705 | 0.149280 | +0.118111 | +0.746714 |

**三点自检**：

1. **R 块恒和为 1**（逐点实测 min 0.999999762 / max 1.000000238；逐列中位数之和为 0.935–0.975，故重归一化）——与任何在线样本落在同一流形上。
2. **新增的第 5 维随负荷单调**：低负荷 −0.080 < 中负荷 +0.182 < 高负荷 +0.390（负 = 线占优，正 = 漂移占优），空转 +0.739 最大（无转速锁定线，漂移占绝对主导）。这正是 B4 拆分想要的特征。
3. **R_2x 随负荷递增**：低负荷 0.012 → 中负荷 0.574 → 高负荷 0.853，与 §14.4 的 2×BPF 结论一致（空转时该分量近消失，presence 0.074 vs 带负荷 0.24–0.26）。

**根本方案与循环依赖**：脚本必须复用模型代码路径，但又不能实例化 `StaticPrototypeMatcher`（后者的 `PROTOTYPE_DIM` 依赖被重算的表）。解法是把 `compute_V_obs` 提为模块级纯函数，方法体退化为薄委托；脚本则在 `MSSTNodeExtractor` 的 numpy/GPU 输出上直接调用它，全程不实例化模型。

### 15.8 顺带修正

| # | 项 | 内容 |
|---|---|---|
| 1 | ONNX 导出侧掩码 | `deploy/sast_export_model.py:_extract_nodes_export` 改为 `import region_weights`（原为独立副本，边界一改即静默过期） |
| 2 | ONNX 导出侧 **IF 聚合口径** | 同文件原为**无权重中位数**，而训练侧已在 `942fe85`（A2）改为**能量加权均值**，导出副本未同步 → 导出模型的 `node_if` 与训练不一致（下游 edge_feats / gate / `i_star` 全部偏移）。已同步，并保留 `torch.where` 以替代布尔就地赋值（ONNX 不支持后者） |
| 3 | 清理已被证伪的 45.65% 假设 | `check_spectral_bands.py` 删 `F_ROT_IDLE` 及其两条梳齿候选、`probe_harmonic_comb.py` 删 `F_IDLE_ROT` 及两条参考线；两处改为陈述证伪结论。**同文件 `SAST_NODES` 仍是 3 节点旧边界 `('BPF', 42.0, 55.0)`，一并改为 4 节点新边界** |
| 4 | 旁路硬编码频带 | `scripts/plot/tfr_common.py` 的 `'harm':(42,105)` / `'low':(2,25)` 改为从 `PUMP_TURBINE_REGIONS` 导出；`analyze_band_compressibility.py` 的 42–48/48–53 注释更正为 48.5 |
| 5 | 文档 | 本节 + §8/§12 回填 |

**残留**：`scripts/doc/build_sast_docx.js:560` 仍写“46.4 Hz 与理论值 2×Z_r×f_rot,空转 = 45.6 Hz 吻合”。该句的数值巧合确实成立（2× 预测 45.645 vs 实测 45.898），但它承袭自已被证伪的 45.65% 前提（1× 预测 22.823 vs 最近实测峰 15.137，落空）。论文 docx 按计划在结果稳定后统一回填。

**导出侧一致性验证**（`_extract_nodes_export` vs `MSSTNodeExtractor.extract_gpu`）：

| 检查 | 结果 |
|---|---|
| `region_weights` 是否同源 | 两侧 `import` 同一函数对象；覆盖集 84 bin 上 `Σ_n w_n` 最大偏差 `0.00e+00` |
| 48.0 Hz 归属（导出轴） | `BPF_DRIFT = 1.0 / BPF_LINE = 0.0` |
| 构造谱（44.0–44.5 Hz 强线 @100 + 46–48 Hz 次强宽带 @1）上 `node_if[BPF_DRIFT]` | 导出侧返回 **44.251 Hz**（能量加权均值），无权重中位数应为 45.000 Hz → 口径已同步 |
| 6 条真信号端到端 | 四个节点的 `\|Δ if\|` / `\|Δ energy\|` / `\|Δ bw\|` **全部 0.000** |

（注：ONNX 写盘本身因 `my_work` 缺 `onnxscript` 而中断，见 §12 第 7 条；上述数值一致性检查绕开了写盘步骤。）

### 15.9 本轮未做

- `FreqEncoder` 的 `mean(dim=-1)` → `mean+std` 时间动态；
- 方案 B（可学习 Δf_i）；
- CUDA kernel 改动（`deploy/reassigner.cu` / `reassigner_opt.cu` 只在全谱两端 clamp，**不存在 48 Hz 处的通道截断**，截断效应只存在于测量窗口内，改这里是白费力气）。

### 15.10 影响与不兼容

- **旧 checkpoint 全部不兼容**：`PROTOTYPE_DIM` 4→6 改变 `obs_proj.weight` 形状（尽管是 dead code，仍是 `state_dict` 条目），且 `STATIC_PROTOTYPES` 值已变。必须重训。
- `band_centroid_if` / 提取器 / 损失侧的数值均有变化（过渡带加权），旧指标的绝对数值不可直接对照。

---

## 16. 自适应「阶数 / 次数」：把缺失的实验补上，再把次数做成可微

### 16.1 起因：偏离确实发生了，位置也定位到了

`models/sast.py` 的类 docstring 一直写着原设计：

```
-> PerBinOrderSelector -> N_star (推理用)
-> SparseGaussianReassigner(omegas[N_star], sigma_i) -> TFR_sast
训练时: 统一使用 omegas[-1] (最高阶), 仅 sigma_i 可微
推理时: 四参数全启用 (sigma_i + lambda_i + IF order + w_i)
```

但同一文件的 `SqueezeIterationController` 上方写着 `(formerly PerBinOrderSelector)`，
而 `PerBinOrderSelector` / `N_star` / `order_idx` **在整个仓库只存在于注释与论文里，
代码中从未存在**（已 grep 确认）。生产路径一直硬编码 `omegas[:, -1]`。也就是说：
**逐 bin 的 IF 层选择是在那次改名中被删掉的，docstring 与论文都没同步。**

同时 `papers/SAST_讲解文档.md` §5.1「MSST IF 迭代次数对不同分量的影响」是**空壳**——
§18.0「MSST lookup 迭代对所有 bin 无害，无需选层」这个断言**从未被实验验证过**。

### 16.2 预先澄清的两处术语误导

| 常见表述 | 代码事实 |
|---|---|
| `omegas [B, 3, F, T]`，对应 1/2/3 阶 SST | 实为 **`[B, 4, F, T]`**（`num = max(4, N_max)`）。`omegas[k]` 是**同一个一阶 IF 估计的固定点迭代层**：`omega_{k+1}[η,b] = omega_k[omega_k[η,b], b]` |
| 「高阶引入高阶相位导数，在稳态频带放大噪声」 | 这是 Colominas 论文针对**传统 SST** 的结论。本实现里没有任何二/三阶相位导数参与（真正的二阶 SST 是另一个函数 `sst2_stft`） |

### 16.3 补上 §5.1：层选择的实验判决（**结论是否定的**）

新增 `scripts/analysis/probe_if_layers.py`。真数据，5 类，`num=4`，σ ∈ {0.5,1.0,2.0}。

**层间差异**（层确实不同，但差异集中在窄带）：

| 层间 | 平均 \|Δ\| (Hz) | 改变 bin 占比 | 相对带宽 |
|---|---|---|---|
| ω₁−ω₀ | 0.74 | 67.7% | LOW_FREQ 4.6% / **BPF_LINE 16.4%** |
| ω₂−ω₁ | 0.58 | 42.3% | 逐样本 std 仅 0.006–0.008 Hz |
| ω₃−ω₂ | 0.35 | 18.2% | 频率分辨率 0.5 Hz；`σ_line` 仅 0.5–1.0 bin |

**质量**（跨 5 类平均，Rényi↓ / 集中度↑ / 锐度↑ 均为「越集中越好」）：

| 节点 | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| LOW_FREQ | 10.05 / 0.26 / 4.9 | 9.70 / 0.29 / 6.8 | 9.38 / 0.32 / 8.8 | **9.19 / 0.34 / 10.1** |
| BPF_DRIFT | 9.29 / 0.25 / 3.3 | 8.98 / 0.28 / 4.4 | 8.72 / 0.31 / 5.3 | **8.61 / 0.31 / 5.7** |
| BPF_LINE | 8.91 / 0.20 / 2.8 | 8.50 / 0.20 / 3.5 | 8.13 / 0.21 / 4.1 | **7.92 / 0.21 / 4.3** |
| 2xBPF | 9.10 / 0.70 / 10.9 | 8.93 / 0.76 / 12.7 | 8.88 / 0.78 / 13.1 | **8.86 / 0.79 / 13.3** |

**判决**：四个分量、三项指标，最优层**一致落在 k=3 = `omegas[-1]`**，即生产路径已在用的
那一层。事前写定的判据（「存在 k* 在窄带显著优于 k=num−1 且在宽带不劣化」）**不成立**。

因此 **不做**逐 bin 的 IF 层融合（β 软融合）：
1. 没有可择余地；
2. 更要紧的是，层间差异的效果方向与 σ / 挤压轮数**完全同向且单调**（越深越集中），
   选低层等价于「挤得更轻」——而 σ 已能**连续、可微**地表达这一点。用一个离散且
   不可微的层索引重复 σ 的功能是负收益。
3. ⚠ 软带上的单调下降**不能**读作「深层对软带更好」：软带设计上应保留展宽，
   「越集中」在那里不是优点。

### 16.4 自适应「次数」：可微软门控递推（本轮实施）

旧实现的三重问题：训练时根本不调用 `SqueezeIterationController`（固定 1 轮）→ 训推
不一致；`round()` 不可微 → 轮数拿不到梯度；`n_sqz_per_bin` 被 `clamp(1, …)` 卡在 ≥1 →
**推理永远不可能 0 轮**，与 §18.0 写的「λ=0 → LOW_FREQ 不挤」直接矛盾。

现改为（`models/sast.py: SparseGaussianReassigner.forward`）：

```
r   = clamp(w_i · M, 0, M)            期望轮数
g_m = clamp(r - (m-1), 0, 1)          满轮取 1, 末轮取小数, 之后恒 0
part = g_m · ridge_factor
S ← (1 - part)·S + part·Reassign(S)
```

训练与推理走**同一条**路径，`if not training` 分支已删除。

**为什么是「输出混合」而不是「对源谱加权」**：`ReassignerFunction.backward` 对第一个
输入恒返回 `None`。若写成 `S' = Reassign(S⊙g) + S⊙(1-g)`，autograd 只能看到
`∂S'/∂g = -S`，而真值是 `A_θS - S`（`A_θ` 为列随机重排矩阵）——误差恰好删掉「能量真聚到
目标 bin」那一项，在重排确实浓缩处会**翻号**。输出混合形式下 `∂S'/∂g = R - S`，
两者都是前向张量，直接偏导**精确**。

**⚠ 这是语义变更，不是纯梯度修正**：`R(S⊙g) ≠ g⊙R(S)`（源门控 vs 目标门控）。
实测量化（逐 bin 变化的 g）：σ=0.5 差 5.4% / σ=1.0 差 14.2% / σ=2.0 差 17.5%，
随 σ 增大而增大。**不能声称与旧实现等价。** 根治方案是给 `ReassignerFunction` 补
`tfr_weighted` 的 VJP（scatter 的伴随 gather），但需改三个 CUDA kernel，见 §16.7。

**门控形式的选择**：原提案用 `sigmoid(τ(w_i·M − m + 0.5))`，其 0.5 交叉点在
`w_i = (m−0.5)/M`，M=3 时第一轮阈值仅 `w_i > 0.167` → `w_i = 0.25` 时 `g₁ ≥ 0.5`，
**永远至少一整轮**，与「等效不到 1 次重排」的意图相反；且 τ 有退化最优。改用 clamp 版：
无参、处处单调、`w_i = 0` 时真正 0 轮。sigmoid 版保留为 `--gate_mode sigmoid` 消融项。

**验证**（`scripts/test/` 与临时校验脚本）：

| 检查 | 结果 |
|---|---|
| 门控行为 | `w_i`: 0.00→Σg=0.00 / 0.25→0.75 / 0.45→1.35 / 0.65→1.95 / 0.90→2.70 / 1.00→3.00 |
| 可微性（有限差分 vs autograd） | 相对误差 **5.7e-09** |
| 门控是独立活路径 | `sigma.detach()` 后 `\|dL/dw_i\| = 3.17e-03 ≠ 0` |
| 训推一致 | 两 flag 之差 / 同 flag 重跑之差 = **0.998**（比值 1.0 ⇒ 残差全部来自 kernel 的 atomicAdd 非确定性） |
| forward 中残余分支 | 已无 `if not training` |
| 整数锚点回归 | `w_i = k/M` 时与「硬 k 轮」最大差 **0.000e+00**（k=0,1,2,3） |

⚠ **可微性检查的一个陷阱**：用 `out.sum()` 做投影时导数恒为 0，因为重排**能量守恒**
（`ΣR = ΣS`），门控只改变分布不改变总量。必须用非均匀投影才能观测。同理，用常数 `g`
测语义差异也恒为 0，因为重排对 `S` 是线性的。两处都踩过，已写入脚本注释。

### 16.5 λ_supcon 1.0 → 0.2 与时频质量验证指标

实测当前占比：supcon 1.440 (62.3%) / RE_2D 0.798 (34.5%) / 其余 ≤0.05。降到 0.2 后
supcon ≈ 0.288 (~26%)，RE_2D 成为最大单项。

**代价**：`FreqEncoder` 的**唯一**梯度来源就是 `supcon_loss`，降 5 倍即梯度降 5 倍；
KNN 验证精度会下降——而 KNN 测的是**分类能力**，不是时频质量。

故新增 `sast_inference.evaluate_tf_quality(model, X, batch_size, device)`，报验证集上
逐节点的带内 Rényi-2D 熵 / 集中度 / 带内频率标准差，并给 hard / soft 分组均值，
与 §16.3 的探针同口径。训练日志现在 KNN 与时频质量并列打印。

顺带修：`L_var` 是活项（λ=0.5）却不在 epoch 日志行里，是日志对不上账面（残差 0.00067）
的原因，已补上 `var=` 字段。

### 16.6 本轮查出的其它问题

| # | 问题 | 处理 |
|---|---|---|
| 1 | 推理 `n_sqz_per_bin` 下界卡 1，永远不可能 0 轮，与 §18.0 矛盾 | 已由软门控取代 |
| 2 | 导出侧重算 `sigma_i` 漏了 `sigma_soft_cap=3.0`（顶到 15），训练侧封顶 3 | **已修**（§16.9）：把 `sigma_max` 由 15.0 改为 3.0，导出侧 `delta = sigma_max − sigma_min = 2.5` 与训练侧 `min(sigma_max, soft_cap) − sigma_min = 2.5` 自动一致 |
| 3 | `return_all` 是空操作（从不被读到） | 已在 docstring 标注为已废弃 |
| 4 | `lambda_balance` 配置 0.5 vs 函数签名默认 0.01，两处真值源 | **已修**（§16.9）：签名默认改为 0.5 |
| 5 | 「阶 / order」陈旧命名 + 描述不存在的 `PerBinOrderSelector` | 已修 `models/sast.py` 类 docstring、`papers/SAST_完整设计文档.md` §18.0/18.1、`papers/SAST_讲解文档.md` §5.1 |

### 16.7 待同步（本轮**未做**，明确登记）

- **`deploy/` 全链路**：`sast_export_model.py` 的 `_multi_round_reassign_export`、
  `export_sast_onnx.py` 与 `quantize_sast_modelopt.py` 的 `OUTPUT_NAMES`
  （`n_sqz_per_bin` 已被 `w_bin` 取代）、`sast_jetson_infer.py` 的 `_reassign_cuda`
  逐轮循环。**训练侧已改而导出侧未改，两边现在不一致**，导出模型不会复现训练行为。
- ~~**`ReassignerFunction` 对 `tfr_weighted` 的 VJP**~~ → **已降级为「不建议做」**。
  原理：补上 VJP 可把「输出混合」还原为「源门控」语义（二者实测差 σ=0.5 时 5.4% /
  σ=1.0 时 14.2% / σ=2.0 时 17.5%），代价是改三个 CUDA kernel 并重编译。但 §16.4.1
  已证明**门控路径梯度对最终收敛无实质影响**（`--gate_detach` 与 base 几乎逐位相同），
  即 σ 才是承重机制。既然改造不带来可测收益，就不值得动 kernel。
  **保留条件**：若将来让门控路径变成承重的（例如给它独立的监督），这个 VJP 才重新有意义。
- **导出侧 σ soft-cap 不一致**（§16.6 #2）与 **`lambda_balance` 双真值源**（#4）。
- **transpose 提升**（已实测但未做，优先级低）：`use_opt` 路径每次调用做 5 次
  `transpose(1,2).contiguous()`，其中 `omega_hat_int` 是 int64/153 MB。把 `sigma` /
  `omega_hat_int` 常驻 `[B,T,F]`、整条链只在进出各转一次，M=3 从 192 ms 降到 155 ms
  （−19% 重排时间，−12% 峰值显存），折合步时约 −3.6%。未做是因为需要引入第二份
  重排实现（`HoistedReassigner`），与本轮「消除重复实现」的方向相反。
- **M 的取值依据**（实测，B=10/F=1000/T=2000）。**孤立基准**：M=1 71 ms /
  M=2 128 ms / M=3 192 ms / M=4 256 ms，相对 M=1 增量 +50 / +109 / +170 ms。
  **端到端实测**：同一 `--max_samples 900` 配置，旧单轮 46 s/epoch → 现在 M=3
  为 **51 s/epoch（+11%）** —— 低于孤立基准外推的 +18%，因为一步里 MSST 等
  环节与之重叠。峰值显存 M=3 ≈ 1.22 GB（孤立基准）/ 6.5 GB（整模型实测，
  8 GB 卡；旧文档里的 70 GB 是旧纯 Python 循环的数字，与自定义 Function 的
  M 次链式调用无关）。
- **⚠ 显存已经贴近上限**：整模型实测占 6.5–7.5 GB / 8 GB。全尺寸的挤压诊断张量
  曾把峰值顶到 7477 MiB 并使训练近乎停滞 —— 因此 `SAST.forward` 只返回**节点级**
  有效轮数向量 `gate_rounds_node [N_phys]`，全尺寸的 `ridge_factor` / `w_bin`
  由 `model.emit_squeeze_diag`（默认 False）控制。后续若增大 M 或 batch，
  必须重新评估显存。

### 16.4.1 三方消融：门控到底是不是新信号？

`--epochs 2 --max_samples 900`，同一 seed，四个配置：

| 配置 | hard Rényi↓ | hard 集中度↑ | hard 带内std | Val acc | T (ep1) |
|---|---|---|---|---|---|
| base（ridge 开、两路都通） | 7.753 | 0.2506 | 0.839 | 0.963 | 0.9683 |
| `--train_ridge 0` | **7.634** | **0.2779** | **0.660** | **0.970** | 0.9409 |
| `--gate_detach 1`（只剩 σ 路径） | 7.754 | 0.2504 | 0.839 | 0.963 | 0.9686 |
| `--sigma_detach 1`（只剩门控路径） | 8.200 | 0.2431 | 0.912 | 0.985 | 0.9760 |

**（1）门控的行为是对的。** 训练后 `w_i` = LOW_FREQ 0.070 / BPF_DRIFT 0.495 /
BPF_LINE 0.999 / 2xBPF 1.000，对应有效轮数 `w_i·M` = **0.21 / 1.49 / 3.00 / 3.00**。
这正好实现设计意图：低频**不足一轮**（保留宽峰），窄锁定线**满 M 轮**（极致锐化）。
门控确实在按节点区分。

**（2）但门控的梯度贡献可以忽略。** `gate_detach`（只剩 σ 路径）与 base 的结果几乎
逐位相同（hard Rényi 7.754 vs 7.753、集中度 0.2504 vs 0.2506、T 0.9686 vs 0.9683）。
即**断掉门控路径后训练结果不变** —— 门控在当前损失配置下是 σ 路径的**重参数化**，
不是独立的监督来源。这与 Plan 评审的预判一致，本轮如实记录，**不宣称门控是新信号**。
（2 epoch / 765 样本的短跑，结论强度有限，但四个指标同时吻合到这种程度不是巧合。）

**（3）σ 是承重机制。** `sigma_detach`（只剩门控）在 hard 带上明显更差：Rényi 8.200
vs 7.753（更不集中）、带内 std 0.912 vs 0.839（更宽）。门控**不能**替代 σ。

**（4）⚠ `--train_ridge 0` 反而更好 —— 当前默认值不被数据支持。**
关掉 ridge 后 hard 带集中度显著提升（Rényi 7.634 vs 7.753、std 0.660 vs 0.839），
Val acc 也从 0.963 升到 0.970。注意这个开关**不是**「训练/推理分歧开关」——
`train_ridge=False` 会让训练与推理**都**用 `ridge=1.0`，两者仍然统一；它实际是
「要不要用 ridge 机制」的开关。
**未据此刻改默认值**（单 seed / 2 epoch / 765 样本，证据强度不足），但这是下一个
应优先复现的项：更长训练 + 多 seed 确认后，默认值大概率应改成 `False`。
机制上我**尚未解释**为什么施加 ridge 会降低集中度（一个可能：`ridge` 的脊线取
`node_if`，若它偏离真实峰值，被压到 floor 0.15 的恰恰是真实峰两侧最该一起搬走的
bin）；不做无证据的断言。

**（5）一个反直觉但重要的旁证**：`sigma_detach` 的 Val acc 最高（0.985）却在 hard 带
时频质量最差（Rényi 8.200）。这正是把 KNN 精度当作主指标的隐患 —— 它测的是工况
可分辨性，与时频集中度**可以背离**。这也是本轮补 `evaluate_tf_quality` 的直接理由。

### 16.4.2 ridge_factor 的机制：已量化验证，且是 BPF_LINE 专属效应

§16.4.1 的 (4) 只报了「`--train_ridge 0` 更好」这个现象，未解释机制。现已量化验证。

**机制**：在输出混合递推 `S ← (1-part)·S + part·Reassign(S)`（`part = g_m·ridge`）下，
一个源 bin 每轮保留 `(1-part)` 倍自身能量在**原位**。M 轮后的残余比例是

```
∏_m (1 − g_m(f)·ridge(f))
```

`ridge` 是**源 bin 到脊线的距离**的钟形衰减（floor 0.15）。而 MSST 的作用恰恰是把
**远离脊线的侧翼 bin** 的 ω̂ 指向脊线中心——也就是说，最该被搬走的 bin，正是
`ridge` 压得最低的那些。于是侧翼能量被 `(1-part)` 反复留在原地，谱线两侧的
STFT 泄漏裙边搬不干净。

**实测（逐 bin 的 `∏(1-part)`，真数据）：**

| 节点 | 带宽内 bin 数 | ridge 均值 | ridge 最小 | 残留均值 | **残留 >0.1 的 bin 占比** |
|---|---|---|---|---|---|
| **BPF_LINE** | 9 | **0.774** | **0.150**（floor） | **0.165** | **50.9%** |
| 2xBPF | 30 | 0.965 | 0.856 | 0.027 | 2.3% |
| BPF_DRIFT | 14 | 0.927 | 0.216 | 0.045 | 13.7% |
| LOW_FREQ | 32 | 0.958 | 0.590 | 0.027 | 3.9% |

**开/关 ridge 的直接对照（同一前向、同一输入）：**

| 节点 | 带内 std（ridge 开） | 带内 std（ridge 关） | 比值 |
|---|---|---|---|
| **BPF_LINE** | 1.0310 Hz | **0.6342 Hz** | **0.615** |
| 2xBPF | 1.4405 Hz | 1.3559 Hz | 0.941 |

**三点结论：**

1. **机制成立且可精确预测**。按 `(1−0.15)³ = 0.614` 估算的残留比例，与 BPF_LINE 实测的
   std 比值 **0.615** 吻合到 0.2%。这不是相关性，是可以算出来的。
2. **但它是 BPF_LINE 专属效应，不是两个硬带共有**。2xBPF 只变化 6%，因为它的
   `bw_expected = 4.5` 使 `ridge_dist = |Δf|/9` 增长很慢，带内 `ridge` 始终 ≥ 0.856，
   残留仅 2.7%。**「BPF_LINE 和 2xBPF 都受影响」的说法与数据不符。**
3. **⚠ 这个效应是被本轮的 `bw_expected` 标定放大的**。§15.4 把 BPF_LINE 的
   `bw_expected` 从 1.5 收到 1.0，使 `ridge_dist = |Δf|/2` 更陡。在 `|Δf| = 3 Hz` 处：
   - 旧值 1.5 → `ridge = exp(−1/8) = 0.882` → 3 轮残留 `(0.118)³ = 0.0016`
   - 新值 1.0 → `ridge = exp(−9/32) = 0.755` → 3 轮残留 `(0.245)³ = 0.0147`

   即**放大约 9 倍**。换言之该缺陷一直潜伏，是本轮的标定把它激活的。教训：对一个
   与架构不匹配的机制做参数标定，可能把潜在缺陷变成显性缺陷。

**架构层面的判断**：`ridge_factor` 源自 §18.2，那里的设想是「远离脊线的 bin 应跳过
挤压」。这个设想在**源门控**语义下是自洽的（不参与 = 源能量不动，那本来也是错的，
但至少不是被反复钉死）。但在**输出混合**语义下，`part < 1` 的含义变成「每轮把
`(1-part)` 的原始能量留在原地」，于是一个「跳过」的意图变成了一个**洗不掉的残留
基底**。二者不匹配，这才是根因 —— 所以正确的处置不是调 `ridge` 的参数，而是
**关掉它**（`part = g_m`）。

**残余不确定性**：以上全部基于 2 epoch / 765 样本的短跑与单样本前向。20 epoch 全量
对照正在进行（见 §16.4.1 的 (4)），结论以那次为准。

### 16.9 收口：三项小修与本轮的最终待办

**已修（本轮追加）**

| 项 | 改动 | 依据 |
|---|---|---|
| `sigma_max` 15.0 → **3.0** | `sast_config.py` / `SAST.__init__` / `SparseGaussianReassigner.__init__` / CLI 默认 | 实际 `sigma_i = 0.5 + (1−w_i)·2.5 ∈ [0.5, 3.0]`，15.0 从未被达到，却把重排半径算成 `K = ceil(3.0×15.0) = 45`。改成 3.0 后 `K = 9`。**附带自愈**了导出侧 σ soft-cap 不一致（§16.6 #2）：导出用 `delta = sigma_max − sigma_min = 2.5`，与训练侧 `min(sigma_max, soft_cap) − sigma_min = 2.5` 相同。 |
| `lambda_balance` 签名默认 0.01 → **0.5** | `models/sast_losses.py` | 消除与 `sast_config.py` 的双真值源（§16.6 #4） |
| VJP 改造 | **降级为「不建议做」** | §16.4.1 证明门控路径梯度对收敛无实质影响，改造无收益（§16.7） |

**`K` 45 → 9 的实际收益（需按 kernel 分别说，不能一概而论 5×）**

| 路径 | 循环范围 | 收益 |
|---|---|---|
| `reassigner.cu`（base）与 Python fallback | 满 ±K，无动态截断 | **≈5×** |
| `reassigner_opt.cu`（训练实际使用） | `K_eff = min(ceil(6σ), K)` | **仅 σ > 1.5 时被截断**（12/15/18 → 9），最大 2×；σ ≤ 1.5 不变 |

⚠ 而 `σ > 1.5` 只出现在 **LOW_FREQ（σ≈2.8）与 BPF_DRIFT（σ≈1.8）** 两个**软带**；
hard 带（BPF_LINE / 2xBPF）训练后 `w_i ≈ 1.0` → `σ = 0.50` → `K_eff = 3`，远低于 9，
**完全不受影响**。截断代价是 σ=3 时核被截到 ±3σ，丢失 **0.27%** 高斯质量（归一化
后能量仍守恒），落在本就该宽的软带上，可接受。

**待办（按优先级）**

1. **20 epoch 全量 `train_ridge` 对照**（进行中）。若 `--train_ridge 0` 的优势保持，
   则把默认值改为 `False`，并考虑直接删除 `SqueezeIterationController` 的
   `ridge_factor` 计算（省掉 `[B, N_phys, F, T]` 距离矩阵的显存与算力）。
   依据见 §16.4.2：ridge 与输出混合架构根本不匹配，不是参数没调好。
2. **`deploy/` 全链路同步**（等主干超参锁定后一次性做）：`sast_export_model.py`
   的 `_multi_round_reassign_export`、`export_sast_onnx.py` 与 `quantize_sast_modelopt.py`
   的 `OUTPUT_NAMES`（`n_sqz_per_bin` → `w_bin`）、`sast_jetson_infer.py` 的
   `_reassign_cuda`；并装 `onnxscript` 完成导出验证。**注意训练侧已改而导出侧未改，
   两边现在不一致。**
3. **transpose 提升**（低优先级，已实测：M=3 从 192 ms → 155 ms，−12% 峰值显存）。
   未做是因为要引入第二份重排实现。

### 16.10 Phase 1 判决：「自适应次数」不能成为承重机制（附原理）

§16.4.1 发现 `--gate_detach` 与 base 几乎逐位相同，即门控不是承重机制。为决定是否值得
给它独立的自由度（把轮数与 σ 解耦），先做了**一次便宜的证伪实验**，结论是**不值得**。

**判据（事前写定）**：在 `(σ, R)` 平面上扫网格取 hard 带时频质量最优；再沿 w_i 约束曲线
`σ = 0.5 + (1−w_i)·2.5`、`R = w_i·M` 扫取最优。若网格最优明显优于曲线最优，说明当前
架构表达不出那个方向，解耦有空间；否则没有。

**结果（`BPF_LINE`，15 条真样本，5 类各 3）：**

```
(σ, R) 网格 — 带内 Rényi↓ / 集中度↑ / 带内 std↓
  σ\R        R=1              R=2              R=3              R=4
  0.50   7.499/0.219/0.37  7.205/0.242/0.31  7.172/0.244/0.32  7.172/…（=R=3）
  1.00   7.884/0.205/0.41  7.795/0.211/0.38  7.771/0.212/0.39  7.771/…
  3.00   8.572/0.192/0.78  8.506/0.175/0.78  8.387/0.177/0.77  8.387/…

网格最优 (7.1719)  ==  曲线最优 (7.1719)      差距 0.00%
```

**原理（这才是可写进论文的部分）：** Rényi 熵在 R 上**单调递减**、在 σ 上**单调递减**
（σ 越小越集中），所以 (σ, R) 平面的最优**必然是角点** (σ→极小, R→极大)。而 w_i 约束
曲线恰好也是从 (σ 大, R 小) 单调走到 (σ 小, R 大) 的——**两者是同一条对角线**。于是
不受约束的最优与受约束的最优重合，解耦**按构造不可能带来收益**。

推广到四个节点，结论一致且更完整：

| 节点 | 期望的挤压强度 | 期望 (σ, R) 方向 |
|---|---|---|
| BPF_LINE / 2xBPF（硬带） | 最大 | σ→min **且** R→max |
| LOW_FREQ / BPF_DRIFT（软带） | 最小 | σ→max **且** R→min |

**两类节点的两个轴永远同向**（要挤就两轴一起加，要松就两轴一起减）。因此「多轮 + 宽核」
或「一轮 + 窄核」这两种需要解耦才能表达的组合，**在这份数据上没有任何分量想要**。

**处置：不做 Phase 2（给轮数独立自由度）。** 门控保留在现有形态——它是一次真实的
工程改进（把不可微的整数轮数变成连续可微、并消除了训推不一致），但**不是**新增的
监督信号，论文里**不得**把它表述为独立的机制贡献。

**残余说明**：本实验只测了「带内集中度」这一类指标。若将来引入一个**非单调**的目标
（例如要求某分量既集中又保留可测带宽，即需要「挤得狠但核不能窄」），两个轴才可能
真正解耦。当前损失集里没有这样的项。

### 16.11 为什么「重估 IF」这条唯一有希望的路也走不通

承 §16.10：轮数唯一可能具备、而 σ 无法表达的能力，是**逐轮重估 IF**
（§18.0 旧稿写的「每轮重估 omega_5, SNR 逐轮提升」）。我在 §15.9 曾以「不可微」为由
否掉它 —— 那个理由**不充分**（落点索引本来就没有梯度，重估不会损失任何现存的梯度
路径）。真正的阻塞在别处：

```
重排搬的是幅值:  tfr_mag = tfr_stft.abs()          (models/sast.py:1063)
                 scatter_add(target, w_k * tfr_weighted)   (:734)
IF 估计需要相位:  omega = diff(unwrap(angle(tfr))) * N / (2π)
```

第 1 轮之后 `S` 已经是一个**非负实场**，相位信息不存在，**无从重估 IF**。
要让轮间重估成立，必须把重排从「搬运幅值」改成「搬运复系数」并重新导出相位 ——
那要改动重排器的值语义、能量守恒构造、以及 `reassigner.cu` / `reassigner_opt.cu`
两个 kernel。那不是「1–2 天」的量级，而且是另一套架构、另一个题目。

**因此 (b)「让门控承重」有两条独立的死角：**
1. **单调冗余**（§16.10）：两个轴永远同向，解耦按构造无收益；
2. **唯一的差异能力不可实现**（本节）：轮数唯一能做的 σ 做不到的事，需要相位，
   而相位在第 1 轮就丢了。

**处置：停止 (b)，回到「冻结架构」。** 门控保留现有形态 —— 它是一次真实的工程改进
（整数轮数 → 连续可微；消除训推不一致；顺带修掉 0 轮不可达的 bug），但论文中
**不得**表述为独立的机制贡献。§16.10 / §16.11 这两条论证本身是可写进论文的
**负结果**：它把「为什么这里的自适应次数是冗余的」讲清楚了，而不是含糊带过。

#### 16.10.1 四节点复核（含该实验自身的一个偏向，必须说明）

§16.10 只测了 `BPF_LINE`，其余三个节点是按损失集结构**论证**的。补测四节点：

```
节点              网格最优(σ,R)      Rényi     曲线最优 w_i     Rényi      差距     判定
LOW_FREQ          (0.50, 3)       8.4076       1.000        8.4076    -0.00%    无空间
BPF_DRIFT         (0.50, 4)       8.0390       1.000        8.0390     0.00%    无空间
BPF_LINE          (0.50, 3)       7.1719       1.000        7.1719    -0.00%    无空间
2xBPF             (0.50, 3)       8.2324       1.000        8.2324    -0.00%    无空间
```

**⚠ 该实验的偏向**：所用指标只测**带内集中度**，对四个节点都给出同一答案——挤得越狠
越好（最优全在 σ=0.5、R=max 的角点）。对两个**软带**这是**错的目标**：它们该保持
展宽（`SOFT_F_TYPES` 被 `renyi_2d_loss` 跳过，且 `lowfreq_sharpness_loss` 正是惩罚
锐化）。所以这张表**不能**用来主张「软带也应该猛挤」。

**但这个偏向不推翻结论，反而把原理说清楚了**：w_i 约束曲线覆盖的正是从
「最大挤压」`(σ=0.5, R=3)` 到「最小挤压」`(σ=3.0, R=0)` 的**整条对角线**。因此
无论某个节点想要极端集中还是极端宽松，最优点都落在**曲线的端点**上：

- **只要目标函数在 (σ, R) 上是单调的，其最优就落在曲线上**，解耦不可能有收益；
- 解耦只有在目标**非单调**时才有意义（例如「必须集中，但带宽不得低于 X」这种
  需要「挤得狠 + 核不能窄」的组合）。

**当前损失集里没有任何非单调项** —— 硬带只要集中（RE_2D 最小化），软带只要不集中
（RE_2D 跳过 + lowfreq 惩罚锐化），都是单调的。这就是门控无法成为承重机制的根本原因。

**因此 (b) 正式关闭。** 若将来引入非单调目标，「解耦轮数与 σ」可以重新评估——本节的
实验脚本与判据可直接复用。
