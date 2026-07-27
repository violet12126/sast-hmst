# WSST2 实现审查 & SAST 改进方案

> 日期: 2026-07-23

---

## 第一部分：WSST2 Python 实现审查

### 1.1 文件位置

- **Python 实现**: [models/tfr.py](models/tfr.py) — 函数 `wsst2()`, 行 146-334
- **MATLAB 参考**: [WSST2-master/Wsst2_new.m](WSST2-master/Wsst2_new.m) — Pham & Meignen (2015)
- **验证脚本**: [plot_compare.py](plot_compare.py) — ssqueezepy WSST vs Our WSST vs Our WSST2

### 1.2 算法流程（MATLAB ↔ Python 逐行对照）

```
信号输入
  │
  ├─ 尺度网格
  │   MATLAB L31:  noct = log2(n)-1;  na = noct*nv+1
  │   Python L208: noct = np.log2(n); na = floor(noct*nv+1)   ← +1 倍频程
  │
  ├─ 对称填充到 2 的幂
  │   MATLAB L40:  [N, sx, n1] = mypad(s)
  │   Python L213: N, sx, n1 = _mypad(s)                      ✅ 一致
  │
  ├─ FFT + 负频率截断
  │   MATLAB L52:  xh = fft(sx)
  │   Python L216: xh = np.fft.fft(sx); xh[N//2+1:] = 0      ← 解析信号构造
  │
  ├─ 逐尺度循环 (MATLAB L76-95 / Python L229-261)
  │   for each scale a:
  │     ├─ 小波频域响应 ψ̂(a·ξ), ψ̂'(a·ξ)
  │     │   MATLAB: [psih, dfilt, ~] = filt(a)
  │     │   Python: _cmor() / _morlet() / _morse(a*xi)        ✅ 一致
  │     │
  │     ├─ norm2psi = ||ψ̂||                                   ✅ 一致
  │     │
  │     ├─ 5 个频域 CWT 矩变换 (ifft(conj(·)·xh)):
  │     │   Wtmp   = V^ψ                                       ✅ 一致
  │     │   Wnu    = V^{ξ·ψ}                                   ✅ 一致
  │     │   Wnunu  = V^{ξ²·ψ}                                  ✅ 一致
  │     │   Wp     = V^{ψ'}                                    ✅ 一致
  │     │   Wnup   = V^{ξ·ψ'}                                  ✅ 一致
  │     │
  │     ├─ 一阶 IF:   ω̂₁ = 1/a · Wnu / Wtmp         (L91)     ✅ 一致
  │     ├─ 群延迟:     τ = a·Wp / (2iπ·Wtmp)         (L92)     ✅ 一致
  │     └─ chirp 率:  q̂ = 2iπ/a²·(Wnunu·Wtmp-Wnu²)/(den)     ✅ 一致
  │                   den = Wtmp²+Wnup·Wtmp-Wp·Wnu  (L93)
  │
  ├─ 二阶 IF 计算 — 关键步骤
  │   MATLAB L97-100:
  │     %phipp(abs(real(tau)*n)<1)=0;    ← 被注释 (正则化未启用)
  │     omega2 = real(omega - phipp.*tau);  ← 复数运算后取实, 保留 Im 交叉项
  │     omega = real(omega);
  │     tau = real(tau);
  │
  │   Python L263-277:
  │     omega = np.real(omega)            ← 先取实
  │     tau = np.real(tau)                ← 先取实
  │     phipp[abs(tau)*n < 1] = 0        ← 启用 τ 正则化 (抑制竖条纹)
  │     omega2 = omega - np.real(phipp*tau) ← 无 Im 交叉项
  │
  │   差异分析:
  │     MATLAB: real(ω - q·τ) = Re(ω) - Re(q)·Re(τ) + Im(q)·Im(τ)
  │     Python: Re(ω) - Re(q)·Re(τ)
  │     → Python 消除了 Im(q)·Im(τ) 虚噪声交叉项，物理上更正确
  │
  ├─ 去填充 (L103-108 / L278-283)                            ✅ 一致
  │
  ├─ 阈值处理 (L116-121 / L285-291)                           ✅ 一致
  │   omega<0 → NaN;  |WT|<gamma → NaN
  │
  ├─ 挤压 (L127-148 / L293-319)
  │   自适应阈值: 2·γ·√2·norm2psi[ai] / √(2n)                 ✅ 一致
  │   WSST2: k = round(1 + nv·log₂(omega2))                   ✅ 一致
  │   WSST:  k = round(1 + nv·log₂(omega))                    ✅ 一致
  │
  └─ 归一化: WSST = WSST / nv · log(2)                        ✅ 一致
```

### 1.3 符号约定验证

**CWT 域二阶 IF 为何用 `-` 号:**

```
STFT 域 (sstn.m L124):  ω₂ = ω₁ + q̂·τ̂     (+ 号正确)
CWT  域 (Wsst2_new.m):  ω₂ = ω₁ - q̂·τ     (- 号正确)

原因: CWT 域 τ = b - t_inst (群延迟 = 分析时间 - 瞬时时间)
      STFT 域 τ̂ = t_inst - b (群延迟 = 瞬时时间 - 分析时间)
      两者符号相反 → IF 修正项符号也相反。
```

**实测验证:** Rényi entropy α=3

| 方法 | Rényi | 参考 |
|------|-------|------|
| ssqueezepy WSST (默认 morlet) | 11.77 | 基线 |
| Our CWT-WSST (Morse, N=1) | 11.31 | -0.46 vs 基线 |
| Our CWT-WSST2 (Morse, N=2) | **11.21** | Δ=-0.10 vs N=1, -0.56 vs 基线 |

WSST2 对调频信号的脊线收敛能力优于 WSST1，符合理论预期。

### 1.4 小波基实现

| 小波 | MATLAB | Python | 状态 |
|------|--------|--------|------|
| Complex Morlet `'cmorFb-Fc'` | `cmor.m` | `_cmor(Fb, Fc, xi)` | ✅ 一致 |
| Morse `'morse'` (amor) | `gmor.m` (β,γ) | `_morse(xi, beta=20, gamma=3)` | ✅ 一致 |
| Morlet `'morl'` | — | `_morlet(xi, omega0=6.0)` | ✅ 新增 |

Morse 小波频域公式:
```
ψ̂_{β,γ}(ξ) = 2·(e·γ/β)^{β/γ} · ξ^β · exp(-ξ^γ)   for ξ > 0
```
默认 β=20, γ=3 (MATLAB `'amor'` 默认值)。

### 1.5 与 MATLAB 的 4 处有意差异

| # | 差异 | MATLAB | Python | 理由 |
|---|------|--------|--------|------|
| 1 | `real()` 顺序 | 先算 ω₂ 后取实 | 先取实后算 ω₂ | 消除 Im(q)·Im(τ) 虚噪声交叉项 |
| 2 | τ 正则化 | 被注释 (L97) | 启用 | 抑制波谷处竖条纹 |
| 3 | 负频率截断 | 无 | `xh[N//2+1:]=0` | 构造解析信号，避免高频混叠 |
| 4 | 倍频程范围 | `noct=log2(n)-1` | `noct=log2(n)` | +1 octave 覆盖完整 500Hz |

---

## 第二部分：SAST 改进 WSST2 方案

### 2.1 目标

将 SAST 的**结构感知自适应压缩**机制引入 WSST2，替换固定阈值 + 硬最近邻挤压，
实现结构感知的二阶同步压缩变换 (SAST-WSST2)。

### 2.2 核心思路

```
当前 WSST2 (硬挤压):                   SAST-WSST2 (自适应软挤压):
                                      
  对每个 (a,b):                         对每个 (a,b):
    if |WT| > 固定阈值:                   C_i = GAT(ridge_features)  ← 学习
      k = round(1+nv·log₂(ω₂))            σ_i = σ_min + (1-C_i)·Δσ   ← 确定映射
      WSST2[k,b] += WT[a,b]               WSST2[:,b] += WT[a,b] · G(f; ω₂, σ_i)
                                      
  问题: 噪声和信号同等对待               优点: 可信分量激进压缩，噪声保守平滑
```

### 2.3 架构设计

```
信号 s(t)
  │
  ├─ wsst2 前半段 (保留, 不修改)
  │   ├─ CWT: WT[na, n] (复数)
  │   ├─ 一阶 IF: omega[na, n]
  │   ├─ 二阶 IF: omega2[na, n]
  │   ├─ chirp 率: phipp[na, n]
  │   └─ 频率轴: freqs[na] (对数间隔, Hz)
  │
  ├─ 脊线提取 (从 |WSST1| 或 |WT| 幅度)
  │   └─ K 条匿名脊线: ridge_freq[K, n], ridge_energy[K, n]
  │
  ├─ SAST 图推理 (每帧)
  │   ├─ 匿名全连接图 + 观测边特征
  │   ├─ PhysicsPrototypeMemory: 原型匹配 → gate, C_prior
  │   └─ EdgeConditionedGAT: → C_i[K] ∈ (0,1]
  │
  └─ 自适应 CWT 域挤压
      σ_i = σ_min + (1-C_i) · (σ_max - σ_min)
      对每个 (a,b):
        WSST2_enhanced += WT[a,b] · exp(-(log₂(ω₂) - log₂(freqs))² / (2σ_i²))
```

### 2.4 关键设计决策

#### 决策 1: 文件组织

| 文件 | 当前 | 修改 |
|------|------|------|
| `models/tfr.py` | `wsst2()` — 完整 CWT+SST | 新增 `sast_wsst2()` 函数 |
| `models/sast.py` | `SAST` 类 — STFT 域 | 新增 `SAST_WSST2` 类 (轻量) |
| `plot_compare.py` | 3 面板对比 | 新增 SAST-WSST2 面板 |

**原则:** 不修改已验证的 `wsst2()`，不破坏现有 `SAST` 类。

#### 决策 2: 频率网格处理

WSST2 使用**对数频率轴** `freqs[na]` (32 点/倍频程)。

SAST 的关键组件与频率网格的关系:
- `BlindRidgeExtractor`: 接受 `freqs` 参数，对网格类型**无依赖** ✅
- `PhysicsPrototypeMemory`: 接受频率 Hz 值，对网格**无依赖** ✅
- `EdgeConditionedGAT`: 只消费节点/边特征，对网格**无依赖** ✅
- `AdaptiveSqueeze`: **强依赖线性网格** — 用 bin 索引做 Gaussian 模糊 ❌

方案: 重写 AdaptiveSqueeze 为 CWT 域版本。

#### 决策 3: CWT 域自适应挤压

原 SAST 的 AdaptiveSqueeze `_gaussian_blur_along_freq`:
- 在**线性 bin 索引**上做 1D conv1d
- 预计算 20 个离散 sigma level 的卷积核
- 每个 TF bin 按 `level_idx` 选择最近的离散 level

CWT 域自适应挤压的区别:
- 频率轴是**对数间隔**: `f_k = f₀ · 2^{k/nv}`
- 挤压目标由 IF `omega2[ai, b]` (归一化频率) 决定
- 用 `log₂` 坐标做 Gaussian kernel 更自然:
  ```
  k_center = nv·log₂(omega2[ai,b])
  weight(k) = exp(-(k - k_center)² / (2·σ_i²))
  ```
- `σ_i` 单位为**倍频程内的 bin 数** (nv=32 → 1 octave = 32 bins)

#### 决策 4: NumPy vs PyTorch

- `wsst2()` 是纯 NumPy — 保留
- SAST-WSST2 的**推理路径**（已训练模型）可用 NumPy 实现，无需 PyTorch
- SAST-WSST2 的**训练路径**需要 PyTorch（梯度通过 C_i 回传）

方案: 新增函数 `sast_wsst2_numpy()` 用于推理 + 对比，参数直接指定 C_i 或使用启发式 C_i。

### 2.5 实现步骤

#### Step 1: CWT 域脊线提取 (`models/tfr.py`)

新增函数 `extract_cwt_ridges(mag, freqs, K=6)`:
```python
def extract_cwt_ridges(mag, freqs, K=6, min_dist_bins=3):
    """
    从 CWT/TFR 幅度谱提取每帧的 top-K 脊线。
    
    Args:
        mag:   [na, n] 幅度谱 (numpy)
        freqs: [na] 频率轴 (Hz, 对数间隔)
        K:     脊线数量
    Returns:
        ridge_freq:   [K, n] 脊线频率 (Hz)
        ridge_energy: [K, n] 脊线能量 (log)
        ridge_bw:     [K, n] 粗略带宽 (Hz)
    """
```

逻辑与 `BlindRidgeExtractor` 的 `_find_local_maxima` + `topk` 相同，
但不含 persistence 跟踪（简化版），或包含帧间贪心匹配。

#### Step 2: C_i 令牌生成 (`models/tfr.py` 或 `models/sast.py`)

新增函数 `compute_compressibility_tokens(ridge_freq, ridge_energy, prototypes)`:

两种模式:
- **模式 A (启发式, 无需训练)**: 基于 ridge_persistence + 频率稳定性 + 原型匹配 直接计算 C_i
- **模式 B (PyTorch, 可训练)**: 使用 PPM + GAT 完整流水线

先实现**模式 A**，可立即验证 SAST-WSST2 效果。模式 B 后续加入。

模式 A 的 C_i 公式:
```
C_i = α · persistence  +  β · (1 - r_std)  +  γ · gate
```
其中 gate 来自 PPM 频率距离门控（当前 PPM 无需训练即可工作）。

#### Step 3: CWT 域自适应挤压 (`models/tfr.py`)

修改 `wsst2()` 的挤压循环，新增参数 `C_i`:

```python
def sast_wsst2(s, fs, gamma=0.01, mywav='morse', nv=32,
               K=6, sigma_min=0.5, sigma_max=8.0):
    """
    SAST-WSST2: Structure-Aware Second-Order WSST.
    
    与 wsst2() 相同的前处理 (CWT + IF 估计)，
    但使用自适应高斯核挤压替代硬最近邻。
    
    Args:
        (同 wsst2)
        K:         匿名脊线数
        sigma_min: 最小挤压带宽 (log2 bin 单位)
        sigma_max: 最大挤压带宽 (log2 bin 单位)
    
    Returns:
        dict: 同 wsst2(), WSST2 替换为自适应版本
    """
```

挤压循环核心变化:

```python
# 原: 硬最近邻
# k2 = int(round(1 + nv * log2(omega2[ai, b])))
# WSST2[k2-1, b] += WT[ai, b]

# 新: 高斯软核, 带宽由 C_i 控制
k_center = nv * np.log2(omega2[ai, b])
sigma_k = sigma_i[i_star, b]  # i_star: 该 (a,b) 归属的脊线
for k in range(max(0, int(k_center-3*sigma_k)), min(na, int(k_center+3*sigma_k+1))):
    weight = np.exp(-0.5 * ((k - k_center) / sigma_k)**2)
    WSST2_enhanced[k, b] += WT[ai, b] * weight
```

#### Step 4: 对比脚本更新 (`plot_compare.py`)

新增第 4 面板: SAST-WSST2，与原 WSST2 对比 Rényi entropy。

### 2.6 预期效果

| 指标 | 当前 WSST2 | SAST-WSST2 (预期) |
|------|-----------|-------------------|
| 挤压方式 | 硬最近邻 (离散 bin) | 高斯软核 (连续) |
| 阈值策略 | 固定 (γ·norm2psi) | 自适应 (C_i 驱动) |
| 噪声分量 | 同等挤压 | 保守平滑 (高 σ) |
| 信号分量 | 同等挤压 | 激进压缩 (低 σ) |
| 脊线连续性 | 依赖 IF 精度 | IF + 帧间一致性 |
| 物理先验 | 无 | PPM 原型匹配 |

预期 Rényi entropy 进一步降低（更集中），尤其对噪声环境和多分量信号。

### 2.7 风险与缓解

| 风险 | 缓解 |
|------|------|
| 对数网格上 Gaussian kernel 覆盖 bin 数随频率变化 | kernel 宽度用 `log₂` 单位，bin 数恒定 (nv bins/octave) |
| 脊线提取在低 SNR 下失败 | 先用 WSST1 粗挤压提高 SNR 再提取脊线 |
| 计算量增大 (na×n 双重循环 + kernel 循环) | 只对 kernel 范围内的 k 循环 (3σ 截断) |
| 模式 A 的启发式 C_i 不够好 | 保留模式 B (PyTorch GAT) 作为升级路径 |

---

## 附录：审查结论

**WSST2 Python 实现 (`wsst2()`) 忠实复刻了 MATLAB `Wsst2_new.m`。** 核心算法
(CWT 矩变换、IF 估计、挤压公式、归一化) 全部一致。4 处有意差异均为改进性质，
不影响正确性，且已通过信号测试验证（Rényi entropy 改善）。

**可以进入 SAST 改进阶段。**
