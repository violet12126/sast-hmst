# 结构感知同步压缩变换 (Structure-aware Synchrosqueezing Transform, SAST)

## —— 物理频率结构图引导的信任分配与自适应能量重分配

---


## 概述

SAST 将时频分析拆分为三个独立职责，每个职责由专门的组件承担：

```
┌─────────────────────────────────────────────────────────────────────┐
│  HMST:     回答 "频率在哪里？"                                        │
│            高阶 IF 估计 → ω̂(t, η)。职责结束，IF 原封不动。              │
├─────────────────────────────────────────────────────────────────────┤
│  Graph:    回答 "这个频率位置值得多激进的能量重分配？"                    │
│            GAT 在图上游走，根据物理边一致性输出重分配策略 σ_sq。          │
├─────────────────────────────────────────────────────────────────────┤
│  Squeeze:  回答 "基于这个策略，能量具体怎么重新分配？"                    │
│            高斯软核，带宽 σ_sq 决定将邻域能量多激进地集中到 IF 位置。     │
└─────────────────────────────────────────────────────────────────────┘
```

**核心区别**：GAT 不输出"信不信这个 IF"——它输出的是"**重分配策略**"。IF 对不对不是 GAT 该管的事（那是 HMST 的职责）。GAT 只决定一个东西：把能量挪到 IF 位置时，下手多重。

- 物理图自洽（多条边的测量互相印证）→ **激进策略**：窄核，把能量狠狠集中到 IF 位置
- 物理图矛盾（各路径互相打架）→ **保守策略**：宽核，能量留在原区域，不急着挤

HMST 说频率在 X，Graph 说"对 X 下手轻点/重点"，Squeeze 执行。**网络输出的不是"对 IF 的信任"，而是"对重分配激进程度的决策"。**

---

## 动机：为什么重分配需要"策略"？

HMST 的硬挤压基于一个隐含假设：**把能量全部集中到 IF 位置总是对的。** 这个假设在理想信号中成立，但在水泵水轮机信号中，有两类频率分量的行为截然不同。

### 第一类：该硬挤的分量 —— 整数谐波（BPF, GPF, 2×BPF…）

这类分量的 IF 由机械结构唯一决定：BPF = 9×fr，GPF = 20×fr。叶片数、导叶数是固定的，所以倍频关系是严格的物理定律。

更重要的是，**TF 能量展宽是纯粹的"测量模糊"，不是真实的物理展宽。** 能量之所以散开，是因为 STFT 的 Heisenberg 不确定性、噪声、以及有限窗长。把能量集中回 IF 位置，是在**消除测量伪影，恢复信号的真实形态。**

HMST 的高阶 IF 估计对这类分量已经非常准确——因为在整数倍频处，多阶导数展开不会碰到分数阶的奇点问题。硬挤 = 回归物理真相。

### 第二类：该软挤的分量 —— 分数阶滑差分量（RSI）

RSI 来自尾水管涡带，不是机械齿轮啮合。它的频率不是任何转速的整数倍，而且在瞬态过程中**比值本身在漂移**（滑差）。

关键区别在于：**能量展宽不全是测量模糊——有一部分是真实的物理展宽。**

| 展宽来源 | 整数谐波 | RSI |
|---------|---------|-----|
| STFT 窗函数模糊 | ✓（应消除） | ✓（应消除） |
| 噪声导致 IF 估计抖动 | ✓（应消除） | ✓（应消除） |
| **真实的频率漂移（滑差）** | ✗ 不存在 | **✓ 存在——这是诊断信号** |

对于 RSI，如果把能量全部挤成一条线，等于**把滑差信号从 TFR 中删除了。** 而滑差恰是涡带稳定性的核心指标——滑差幅度突增意味着涡带正在失稳，是压力脉动加剧的早期预警。

### 两种分量在同一信号中并存

水泵水轮机同时包含两类分量。HMST 没有区分——硬挤所有东西。结果：

- 稳态工况：RSI 被过度挤压 → 滑差信号丢失 → TFR 看起来漂亮但诊断信息受损
- 瞬态工况：RSI 能量弥散 + HMST 硬挤 → 能量被挤向多个不稳定位置 → TFR 混乱

**SAST 要解决的问题不是"HMST 的 IF 不准"，而是"HMST 的重分配没有区分两类分量"。**

### 区分标准：图自洽

在没有真值的情况下，怎么知道一个分量该硬挤还是软挤？

答案在物理图上。整数谐波（如 BPF）有多条独立路径交叉验证——fr→BPF（9×）、BPF→GPF（比值 ≈ 9/20）、BPF→2×BPF（2×）。如果所有路径的测量值互相印证 → 这个分量的 IF 环境是干净的 → 硬挤是安全的。

RSI 只有一条主要路径（RSI→fr，比值约 0.43），没有其他分量能独立验证它的 IF。而且这条路径的比值在时间窗口内天然波动（$\sigma_r$ 大）→ 图在此处无法给出强一致性 → 硬挤不安全 → 软挤保留信息。

**硬挤/软挤的决策不是基于"哪个分量更重要"，而是基于"图对这个分量的 IF 环境给出了多强的交叉验证"。**

---


## Contribution 1: Physics-guided Frequency Graph

### 1.1 设计原则：静态拓扑 + 动态边特征

**图结构由物理知识预设且固定。网络学的是"如何利用这张图"而非"这张图长什么样"。**

图的拓扑在训练和推理中完全不变——换设备时换图模板即可，网络架构不变。运行时只更新边特征值（从 HMST/TFR 测量值实例化）。

### 1.2 节点：有物理身份的频率实体

以水泵水轮机为例：

```
V = {
    fr:       转频,         标称 5.56 Hz,    类型 ROTATION
    RSI_low:  尾水管涡带,   标称 1.7~2.8 Hz,  类型 VORTEX_ROPE
    RSI_turb: 湍流脉动,     标称 5.6~11.1 Hz, 类型 TURBULENCE
    BPF:      叶片通过频率, 标称 50.0 Hz,     类型 BLADE_PASS
    2×BPF:    二阶叶片谐波, 标称 100.0 Hz,    类型 BLADE_HARMONIC
    GPF:      导叶通过频率, 标称 111.1 Hz,    类型 GUIDE_VANE
    3×BPF:    三阶叶片谐波, 标称 150.0 Hz,    类型 BLADE_HARMONIC
}
```

**节点特征**（每个时间步 $t$）：

| 特征 | 来源 | 性质 |
|------|------|------|
| $IF\_raw_i(t)$ | HMST 高阶 IF + 软频率定位 | 动态 |
| $energy_i(t)$ | STFT 幅值在搜索范围内的加权和 | 动态 |
| $f\_type\_embed_i$ | 频率类型可学习嵌入 | 静态 |

**IF_raw 提取**：HMST 输出每个 TF bin 的高阶 IF $\hat{\omega}(t, \eta)$。对每个物理节点，在其标称频率的搜索范围内做能量加权定位：

$$
IF\_raw_i(t) = \frac{\sum_{\eta \in [f_{nom,i}-\Delta f,\; f_{nom,i}+\Delta f]} |V_x(t,\eta)| \cdot \hat{\omega}(t,\eta)}{\sum |V_x(t,\eta)| + \epsilon}
$$

搜索范围 $\Delta f$ 由设备转速波动决定（如 ±5% 标称频率）。

### 1.3 边：有类型的物理关系

```
EdgeType = {
    INTEGER_HARMONIC:   严格整数倍 (BPF = 9×fr, GPF = 20×fr)
    FRACTIONAL_HARMONIC: 非整数倍 (RSI ≈ 0.43×fr)，比值可漂移
    PHASE_LOCKED:       同源锁相 (BPF ↔ GPF, 都锁在转轴上)
    SLIP_COUPLED:       滑差耦合 (RSI ↔ fr, 瞬态时锁定松动)
}
```

**边特征向量**（运行时实例化，**不含相位差**）：

```
e_{ij}(t) = [
    r_ij,       # 实测瞬时比值 IF_i(t) / IF_j(t)
    σ_r,        # 比值稳定性 (局部时间窗口上的 std)
    Corr_E,     # 能量包络相关系数
    d_f,        # 归一化频率距离 |f_i - f_j| / max(f_i, f_j)
]
```

> **为什么删除 $\Delta\phi$**：不同频率的相位转动速度不同（10 Hz 和 50 Hz 差 5 倍），直接相减导致严重的相位卷绕（wrapping），数值上毫无意义。四个特征足够 GAT 学习物理拓扑关系。

### 1.4 跨设备泛化：换图模板不换网络

| 设备 | 节点模板 | 边模板 |
|------|---------|--------|
| 水泵水轮机 | {fr, RSI_low, RSI_turb, BPF, 2×BPF, 3×BPF, GPF} | INT / FRAC / PHASE / SLIP |
| 滚动轴承 | {fr, BPFI, BPFO, BSF, FTF, 2×BPFI} | INT / PHASE / SLIP |
| 齿轮箱 | {fr, GMF, 2×GMF, sidebands} | INT / PHASE / MOD |

相同的 GAT 架构，不同的物理图模板。"**结构感知**"= 网络感知的是频率间的约束结构，而非特定频率数值。

---

## Contribution 2: Graph Attention for Trust Allocation

### 2.1 核心思路：输出重分配策略，而非 IF 修正

HMST 已经回答了"频率在哪里"。同步压缩真正还有自由度的地方是 **"这个位置附近，把能量向它集中的动作应该多激进"**——也就是重分配策略。

传统思路（Δω 预测）混淆了职责：

```
GAT → Δω → IF_corrected → squeeze
      ↑
      网络在替 HMST 重做 IF 估计
      风险：Δω 错 → 能量挤到完全错误的频率 → 虚假分量
```

SAST 严格分离职责：

```
HMST: "频率在 X" (职责结束，不可修改)
GAT:  "对 X 下手轻点还是重点" → σ_sq (重分配策略)
       ↑
      网络管的是"能量重分配的动作幅度"，不是"频率位置"
      风险：σ_sq 错 → 下手太轻(TFR 模糊)或太重(丢失展宽信息)
             → 但频率不会错位 → 不会产生虚假分量
```

**HMST 的职责在 IF 估计完成后就结束了。GAT 不碰 IF，只决定重分配策略。**

### 2.2 Edge-Conditioned Graph Attention

标准 GAT 只在节点特征上计算注意力——不区分边类型，整数谐波和滑差耦合被同等对待。

SAST 采用 **Edge-Conditioned GAT**：将边特征注入注意力计算，使 GAT 能根据物理关系类型、比值稳定性、能量相关性自动调节信任度：

```
第 l 层 Edge-Conditioned GAT:

  α_{ij}^{(l)} = softmax_j( LeakyReLU( a^T [W_q h_i^{(l-1)} || W_k h_j^{(l-1)} || W_e e_{ij}] ) )

  h_i^{(l)} = ReLU( Σ_j α_{ij}^{(l)} · W_v h_j^{(l-1)} )
```

**边特征注入的效果**（GAT 自动学到，无需手动编码规则）：

- `σ_r`（比值波动大）→ $\alpha$ 自动降低 → 不可信的边被弱化
- `Corr_E`（能量相关性高）→ $\alpha$ 自动升高 → 同源分量被强化
- `d_f`（频率距离近）→ $\alpha$ 可升可降（GAT 根据工况学习）
- `slip_index`（由 σ_r 和 r_ij 计算）→ $\alpha$ 自动降低 → 滑差边松绑

### 2.3 双输出头：图一致性 → 重分配策略

GAT 在图上游走，根据物理边的自洽程度输出两个东西——都是关于"重分配该怎么下手"的决策依据：

```
Shared Feature (GAT L 层输出 h_i^{(L)})
      │
      ├── A_ij = α_{ij}^{(L)}    ← "这条物理边自洽吗？" (图一致性得分)
      │                             高 = 多条路径交叉验证通过
      │                             低 = 路径间互相矛盾
      │
      └── MLP_σ → σ_i(t)         ← "对这个频率分量的重分配应该多激进？"
                                    窄 = 激进，宽 = 保守
```

**$A_{ij}$ 和 $\sigma_i$ 的关系不是"先有信任再有带宽"的流水线，而是从图一致性直接到重分配策略的端到端映射。**

物理图上发生了什么：

- 稳态工况，BPF 节点：fr→BPF (9×) 的比值稳定 ≈ 9.0，BPF→GPF 的能量包络同涨同落 → 多条边自洽 → **GAT 读到一致性 → 输出激进策略 → $\sigma$ 窄 → 能量被狠狠集中到 IF 位置**

- 瞬态工况，RSI 节点：RSI→fr 比值在 0.38~0.48 漂移，$Corr\_E$ 低 → 边在"打架" → **GAT 读到矛盾 → 输出保守策略 → $\sigma$ 宽 → 能量留在原区域，不硬挤**

$A_{ij}$ 的附加价值——诊断信号：

| 观测 | 物理含义 | 诊断意义 |
|------|---------|---------|
| BPF↔fr 的 $A$ 骤降 | 9 倍谐波关系被破坏 | 叶轮损伤 |
| RSI↔fr 的 $A$ 突升 | 涡带与转轴进入锁频 | 压力脉动加剧预警 |
| GPF↔BPF 的 $A$ 持续偏高 | 同源锁相稳定 | 正常运行 |

**$A_{ij}$ 不是"IF 的信任度"，而是"物理边的自洽度"。一条边自洽说明两端的频率按照物理预期的关系在走——这不是对 HMST 的打分，而是对物理图当前状态的诊断。**

### 2.4 时间维度处理

每个时间步 $t$ 的图实例**独立通过 GAT**（共享权重），不引入 RNN/Transformer 的时间递归。时间一致性由两个机制保证：

1. **边特征窗口化**：$\sigma_r$ 和 $Corr\_E$ 在局部窗口（如 ±5 帧）上计算，天然携带时序上下文
2. **$\mathcal{L}_{smooth}$ 正则**：显式惩罚 $\sigma_i(t)$ 在时间轴上的剧烈跳变

不用 GRU/LSTM 的理由：时间递归引入梯度消失风险，且使 GAT 层间信息传递与时间建模耦合，难以诊断问题。

---

## Contribution 3: Adaptive Kernel Synchrosqueezing

### 3.1 核心思路

传统 HMST：$\delta(\omega - \hat{\omega})$，硬 Dirac 函数。没有"策略"的概念——所有频率分量一视同仁，能量全部挤到 IF 位置，迭代次数固定。

SAST：**挤压的激进程度由 GAT 输出的重分配策略条件化。** HMST 负责说"频率在 $\hat{\omega}$"，GAT 负责说"往 $\hat{\omega}$ 挤的时候下手轻还是重"，Squeeze 负责执行。

- 图自洽（多条边交叉验证通过）→ **激进策略**：窄核，能量被狠狠集中 → 锐利 TFR
- 图矛盾（各路径互相打架）→ **保守策略**：宽核，能量留在原区域 → **保留物理展宽**
- **滑差本身是诊断信号，把它挤掉等于消灭诊断依据**

### 3.2 Strategy-Conditioned Soft Kernel

$$
SAST(t, \omega) = \int V_x(t, \eta) \cdot K(\omega - \hat{\omega}^{HMST}(t, \eta);\; \sigma_{sq}(t, \eta)) \; d\eta
$$

其中 $K(z; \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{z^2}{2\sigma^2})$。

**重分配策略从 per-node 映射到 per-TF-bin**（两步）：

**Step 1**：GAT 输出 per-node 重分配激进程度：

$$
\sigma_i(t) = \sigma_{min} + (\sigma_{max} - \sigma_{min}) \cdot sigmoid(MLP_\sigma(h_i^{(L)}(t)))
$$

$\sigma_{min} \approx 1$ bin（激进重分配——图自洽，狠狠挤），$\sigma_{max} \approx 10\text{–}15$ bin（保守重分配——图矛盾，保留展宽）。

**Step 2**：每个 TF bin 分配到最近物理节点，继承其重分配策略：

$$
i^*(t, \eta) = \arg\min_i |\eta - IF\_raw_i(t)|
$$
$$
\sigma_{sq}(t, \eta) = \sigma_{i^*}(t)
$$

> **注意**：挤压用的是 HMST 原始 $\hat{\omega}^{HMST}$。HMST 负责"频率在哪"，GAT 负责"往那挤的时候下手多重"。职责不交叉。

### 3.3 自适应迭代门控

不再固定迭代次数 $M$：

$$
TFR^{(m)} = TFR^{(m-1)} + gate^{(m)} \cdot (Squeeze(TFR^{(m-1)}) - TFR^{(m-1)})
$$

$$
gate^{(m)}(t, \omega) = \sigma(MLP_{iter}([energy(t,\omega),\; \sigma_{sq}(t,\omega),\; m]))
$$

直觉：能量已集中的 TF bin 少挤（$gate \approx 0$），能量分散的多挤（$gate \approx 1$）。RSI 分量因 $\sigma_{sq}$ 宽而 $gate$ 持续偏低 → 保留展宽。

---

## 损失函数设计

$$
\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{entropy} + \lambda_2 \mathcal{L}_{physics} + \lambda_3 \mathcal{L}_{smooth}
$$

### 4.1 $\mathcal{L}_{task}$：下游任务损失

使用 TFDCL 框架现有的跨模态对比学习损失。SAST TFR 经 `FreqGlobalEncoder` 得到频域表示 $z_{freq}$，时域信号经 `TimeEncoder` 得到 $z_{time}$，计算 `complematch_cross_modal_ce_loss`：

$$
\mathcal{L}_{task} = -\frac{1}{N}\sum_i \log \frac{\exp(sim(z_{freq}^i, z_{time}^i) / \tau)}{\sum_j \exp(sim(z_{freq}^i, z_{time}^j) / \tau)}
$$

梯度路径：$\mathcal{L}_{task} \rightarrow z_{freq} \rightarrow FreqGlobalEncoder \rightarrow TFR \rightarrow$ squeeze kernel $\rightarrow \sigma_{sq} \rightarrow$ GAT。

### 4.2 $\mathcal{L}_{entropy}$：时频聚集性损失

Rényi $\alpha$-熵衡量 TFR 的能量集中程度。$\alpha=3$ 是 TFR 文献中的常用选择（对弱分量更敏感）：

$$
\mathcal{L}_{entropy} = \frac{1}{1-\alpha} \log \iint \left(\frac{TFR(t,\omega)}{\iint TFR}\right)^\alpha d\omega dt, \quad \alpha = 3
$$

Rényi 熵越低 → TFR 越集中 → 推动 GAT 输出合理的 $\sigma_{sq}$（不能全是最宽的核）。

**注意**：$\mathcal{L}_{entropy}$ 和 $\mathcal{L}_{task}$ 形成自然平衡——太窄的核（$\sigma \rightarrow 0$）使 TFR 锐利但可能丢失信息（损害 $\mathcal{L}_{task}$），太宽的核使 TFR 模糊（损害 $\mathcal{L}_{entropy}$）。GAT 必须在两者之间学习最优 $\sigma_{sq}$。

### 4.3 $\mathcal{L}_{physics}$：物理图约束损失

约束 GAT 的注意力权重 $A_{ij}$ 与可测量的物理一致性对齐。按边类型采用差异化的约束形式：

$$
\mathcal{L}_{physics} = \frac{1}{|E|} \sum_{(i,j) \in E} w(type_{ij}) \cdot A_{ij} \cdot \ell_{ij}
$$

| 边类型 | $\ell_{ij}$（物理一致性惩罚） | 含义 |
|--------|---------------------------|------|
| INTEGER_HARMONIC | $\|r_{ij} - r_{nom}\|_2^2$ | 比值必须等于机械决定的整数 |
| FRACTIONAL_HARMONIC | $\text{Var}_t(r_{ij})$ | 比值应时间稳定（允许偏离标称） |
| PHASE_LOCKED | $1 - Corr\_E$ | 同源锁相应有高能量相关性 |
| SLIP_COUPLED | $\text{slip\_index} = \frac{\|r_{ij} - r_{nom}\|}{r_{nom}}$ | 滑差大时天然高惩罚，自动松绑 |

$$
w(type) = \begin{cases} 1.0 & \text{INTEGER\_HARMONIC} \\ 0.5 & \text{PHASE\_LOCKED} \\ 0.3 & \text{FRACTIONAL\_HARMONIC} \\ 0.2 & \text{SLIP\_COUPLED} \end{cases}
$$

**关键设计**：$A_{ij}$ 出来自 GAT 最后一层的 softmax 注意力。$\mathcal{L}_{physics}$ 惩罚的是"给了高注意力但物理不一致"的边——推动 GAT 学会利用边特征中的 $\sigma_r$、$Corr\_E$ 等信息正确分配注意力。**注意**：约束的是 $A_{ij}$（注意力权重）而非 IF 比值（IF 来自 HMST 不变）。

### 4.4 $\mathcal{L}_{smooth}$：时间平滑正则

惩罚 $\sigma_i(t)$ 在时间轴上的剧烈跳变，避免 GAT 输出逐帧抖动的挤压带宽：

$$
\mathcal{L}_{smooth} = \frac{1}{|V|} \sum_i \frac{1}{T-1} \sum_t \|\sigma_i(t) - \sigma_i(t-1)\|_2^2
$$

同时对 $A_{ij}$ 施加轻量平滑（物理关系不应瞬时跳变）：

$$
\mathcal{L}_{smooth}^A = \frac{1}{|E|} \sum_{(i,j)} \frac{1}{T-1} \sum_t \|A_{ij}(t) - A_{ij}(t-1)\|_2^2
$$

---

## 端到端流程

```
Raw Signal x(t) [B, T, C]
      │
      ├──→ TimeEncoder ──────────────────────────────→ z_time ─────┐
      │                                                             │
      │                                                             ▼
      │                                                        L_task (contrastive)
      │                                                             ↑
      │                                                             │
      └──→ STFT → V_x(t,η), ω̂_HMST(t,η)     [HMST: 固定, 不参与梯度]  │
            │         │                                             │
            │         ├──→ IF_raw_i(t), energy_i(t)  [软频率定位]     │
            │         │         │                                   │
            │         │         ▼                                   │
            │         │  ┌──────────────────────────┐               │
            │         │  │ C1: Physical FSG          │               │
            │         │  │  V = {fr, RSI, BPF, …}    │               │
            │         │  │  E = 类型化物理边          │               │
            │         │  │  e_ij = [r, σ_r, Corr, d] │               │
            │         │  └────────────┬─────────────┘               │
            │         │               │                             │
            │         │               ▼                             │
            │         │  ┌──────────────────────────┐               │
            │         │  │ C2: Edge-Cond. GAT (L 层) │               │
            │         │  │  α_ij = attn(h_i,h_j,e_ij)│              │
            │         │  │  → A_ij (诊断信号)         │               │
            │         │  │  → σ_i (per-node 带宽)    │               │
            │         │  └────────────┬─────────────┘               │
            │         │               │                             │
            │         │               ▼                             │
            │         │  σ_sq(t,η) ← nearest(η, IF_raw_i)           │
            │         │               │                             │
            │         └───────────────┤                             │
            │                         ▼                             │
            │                ┌──────────────────────────┐           │
            │                │ C3: Adaptive Squeeze      │           │
            │                │  K(z; σ_sq) 软核重分配     │           │
            │                │  + 自适应迭代门控          │           │
            │                └────────────┬─────────────┘           │
            │                             │                         │
            │                             ▼                         │
            │                       SAST TFR(t,ω)                    │
            │                             │                         │
            │                             ▼                         │
            │                     FreqGlobalEncoder                  │
            │                             │                         │
            │                             ▼                         │
            │                          z_freq ──────────────────────┘
            │
            └──→ [L_entropy 直接计算在 TFR 上]
                 [L_physics 作用于 A_ij]
                 [L_smooth 作用于 σ_i + A_ij]
```

---

## 可微性分析

| 模块 | 可微？ | 处理方式 |
|------|--------|---------|
| STFT + HMST IF 估计 | ⚠️ 部分不可微 | IF_raw 作为固定特征输入 GNN，不参与梯度 |
| 边特征计算 | ✅（IF_raw 固定） | IF_raw 固定 → 边特征固定 → 无需梯度 |
| Edge-Conditioned GAT | ✅ 完全可微 | linear, softmax, LeakyReLU 均为标准操作 |
| $\sigma_i$ MLP | ✅ 完全可微 | 标准 MLP + sigmoid |
| 自适应挤压 | ⚠️ scatter index 不可微 | index 用 `detach` 切断；value 可微（梯度经高斯核流向 $\sigma_{sq}$） |
| $\mathcal{L}_{task}$ | ✅ 完全可微 | 经 FreqGlobalEncoder 反向传播 |
| $\mathcal{L}_{entropy}$ | ✅ 完全可微 | Rényi 熵对 TFR 可微 |
| $\mathcal{L}_{physics}$ | ✅ 完全可微 | 只依赖 $A_{ij}$ 和 HMST 固定 IF |
| $\mathcal{L}_{smooth}$ | ✅ 完全可微 | 只依赖 $\sigma_i$ 和 $A_{ij}$ |

**关键**：整个训练管线端到端可微。挤压的 scatter index 被 detach（切断从最终 TFR 像素位置到 IF 值的梯度），但 IF 本身来自 HMST 不变，无需梯度。

---

## 参数增量

| 模块 | 参数量 |
|------|--------|
| 节点特征投影 (IF + energy + type_embed → d_h) | ~8K |
| 边特征投影 (4D → d_e) | ~2K |
| 2 层 Edge-Conditioned GAT (d_h=128, K=4 heads) | ~35K |
| MLP_σ (d_h → d_h/2 → 1) | ~10K |
| 自适应挤压 MLP_iter | ~6K |
| 频率类型嵌入 | ~2K |
| **总计** | **~63K** |

相比 TFDCL (~5M) 约 1.3%，可忽略。

---

## 消融实验

| 编号 | 配置 | 验证目标 |
|------|------|---------|
| B0 | 原始 HMST + 固定挤压 | 基准线 |
| B1 | B0 + Physics FSG + GAT + Adaptive Squeeze (完整 SAST) | **整体收益** |
| B2 | B1 去掉 $\mathcal{L}_{entropy}$ | 时频聚集性损失的必要性 |
| B3 | B1 去掉 $\mathcal{L}_{physics}$ | 物理图约束的必要性 |
| B4 | B1 去掉 $\mathcal{L}_{smooth}$ | 时间平滑正则的必要性 |
| B5 | Edge-Conditioned GAT vs 标准 GAT (无边特征) | 边特征注入注意力是否必要 |
| B6 | Physics FSG vs 数据驱动图 (脊线→全连接→稀疏化) | **图作为先验 vs 图作为输出** |
| B7 | 不同 GAT 层数 L (1, 2, 3) | GAT 深度敏感性 |
| B8 | 不同注意力头数 K (1, 2, 4, 8) | 多头注意力必要性 |
| B9 | 固定 $\sigma_{sq}$ vs GAT 预测 $\sigma_{sq}$ | 自适应带宽 vs 全局带宽 |
| B10 | 跨设备迁移 (水泵→轴承，换图模板) | **结构感知泛化能力** |
| B11 | $\Delta\omega$ 预测 vs 信任分配 ($A_{ij} + \sigma$) | **两种范式的直接对比** |

> B11 是最关键的哲学对比实验——验证"信任分配"是否优于"IF 修正"。

---

## 总结

**SAST = HMST（频率在哪）+ Physical FSG + GAT（重分配策略）+ Adaptive Squeeze（执行策略）**

1. **HMST 回答"频率在哪里"**（C1 的输入）：高阶 IF 估计给出 $\hat{\omega}(t, \eta)$。职责到此结束，IF 原封不动，不被任何模块修改。

2. **Graph + GAT 回答"重分配应该多激进"**（C2）：Edge-Conditioned GAT 在物理图上根据边的自洽程度——$r_{ij}$ 是否吻合 $r_{nom}$、$\sigma_r$ 是否小、$Corr\_E$ 是否高——输出 $A_{ij}$（图一致性得分）和 $\sigma_i$（重分配激进程度）。输出的是**策略**，不是对 IF 的修正。

3. **Squeeze 回答"基于策略如何执行"**（C3）：$\sigma_{sq}$ 从 per-node 映射到 per-TF-bin。图自洽 → 激进 → 窄核狠狠挤；图矛盾 → 保守 → 宽核保留展宽。迭代门控自适应决定停止点。

4. **损失端到端**：$\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{entropy} + \lambda_2 \mathcal{L}_{physics} + \lambda_3 \mathcal{L}_{smooth}$。下游任务损失驱动 GAT 学会"什么样的重分配策略对诊断有利"，辅助损失提供正则化。整个管线端到端可微。

**三个角色，三种职责，一条链路：HMST 定位 → Graph 决策 → Squeeze 执行。网络不修正物理，只制定重分配策略。**
