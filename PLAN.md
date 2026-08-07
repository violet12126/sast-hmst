# 实现推理自动挤压次数 (λ) + w_i 平滑 + 低频锐化 + 能量/脊线修复

> **状态: ✅ 全部完成 (2026-08-07)** | 分支: `feat/multi-round-sqz-temporal-smooth-lowfreq-sharp`

## 关键发现
`infer_sast` 未传 `training=False` → 推理自适应分支从未执行(死代码)。
w_i 每帧独立算无时序耦合 → 跳变 → σ 跳 → TFR 竖线。

---

## A. 推理自动挤压次数 λ ✅

### A1. `infer_sast` 传 `training=False`
- [sast_utils.py:278](sast_utils.py#L278): `model(x, return_all=True, training=False)`
- 仅可视化路径; KNN 分类(predict_class 等)仍 `training=True` 避免 z_freq 偏移

### A2. `SparseGaussianReassigner.forward` 加多轮 λ ✅
- 增参 `n_sqz_per_bin`, `N_max`; `__init__` 加 `N_max` 属性
- 推理迭代挤压: 每轮 `part = src_mask * ridge_factor`, 挤走的重排、未挤留原位
- 修复: `tfr_cur = tfr_mag`(不减能量) — ridge_factor 只控参与比例, 不门控初始能量
- `ridge_floor=0.15` 保证远端谐波至少轻度参与

### A3. `SAST.forward` 推理分支 ✅
- 复用 `i_star`: `n_sqz_per_bin = lambda_per_bin.clamp(0, n_sqz_max).long()`
- SAST.__init__ 加 `n_sqz_max`; SastConfig 加 `n_sqz_max`

---

## B. w_i 时序平滑 ✅

**TemporalSmoother** (1D depthwise conv, 固定高斯 base + 可学残差):
- `base_logit`: 固定高斯核 (buffer, 不可学), 保证基础平滑永不被破坏
- `residual`: 可学残差 (Parameter, 初值0), L2 正则约束 → 允许微调不偏离
- 最终权重 = softmax(base_logit + residual), 恒正 sum=1
- `reg_loss()` 罚 residual 偏离 0, 加入 total loss
- 旧 ckpt strict=False 加载: 缺 residual→0, 等效固定高斯

---

## C. 低频锐化 ✅

`lowfreq_sharpness_loss`:
- LOW_FREQ 频段 Rényi 熵 × tonality(CV) 加权
- 只在低频简谐时鼓励集中, 宽带噪声保留展宽
- 接入 total_sast_loss, `lambda_lowfreq=0.05`

---

## D. sqz_controller 重构 ✅

根因: 旧版用 energy-weighted ridge position → 谐波(100/200Hz)距主瓣太远 → ridge_factor=0 → 不参与挤压.

修复:
- 脊线 = node IF (GAT 已跟踪, 无需 energy-weighted)
- 加宽距离衰减: sigma=2×bw_expected, Gaussian exp(-dist²/8)
- ridge_floor=0.15: 所有 bin 至少 15% 参与
- 去除 `energy_ratio` 乘子 (避免能量弱的 bin 被进一步压低)

---

## E. 训练稳定性修复

### E1. softmax 防爆炸
- 旧: `weight/weight.sum()` → sum→0 时归一化放大 → w_i 突破(0,1) → loss 爆炸
- 新: `F.softmax(logit, dim=2)` → 恒正, sum=1, w_i 始终在(0,1)

### E2. 固定高斯 base
- 旧: 纯可学 softmax 权重 → 学成 delta → 绕过平滑 → w_i 跳变
- 新: 固定高斯 base + 可学残差 + L2 → 平滑不可被绕过

---

## F. 绘图更新
- 默认 freq-max=500Hz (全频段), 不再限制 200Hz

---

## 改动文件
1. [models/sast.py](models/sast.py): TemporalSmoother(固定base+残差+L2), reassigner多轮λ+能量修复, sqz_controller重构, SAST.__init__加n_sqz_max
2. [models/sast_losses.py](models/sast_losses.py): lowfreq_sharpness_loss + 接入 total_sast_loss
3. [sast_utils.py](sast_utils.py): infer_sast training=False, load_checkpoint strict=False, SastConfig 加 lambda_lowfreq/smoother_kernel/n_sqz_max/resume
4. [train_sast.py](train_sast.py): --lambda_lowfreq --smoother_kernel --n_sqz_max --resume CLI, smoother_reg 加入总loss, 日志加 lf/sr
5. [scripts/plot/plot_sast_vs_msst.py](scripts/plot/plot_sast_vs_msst.py): default freq-max=500

## 验证
- ✅ smoke test: lowfreq_sharpness_loss, TemporalSmoother, SAST forward 两侧
- ✅ 推理 n_sqz_per_bin 自适应 per-bin
- ✅ 旧 ckpt 加载 (strict=False, wi_smoother 缺 key → 高斯初值)
- ✅ 训练稳定 (w_m∈[0.68,0.81], smo<0.015, bal<0.07)
- ✅ 100/200Hz ridge_factor 不再为 0 (ridge_floor=0.15)
- ✅ 能量不丢失 (SAST/STFT=1.0 在 100/200Hz)
- ✅ plot 0-500Hz 全频段

## Commits
```
4be4eea fix(sast): TemporalSmoother 用 softmax 替代 weight/sum 归一化, 防训练爆炸
ed99b44 fix(sast): 训练日志增加 lowfreq_sharp (lf) 输出
d9b690d fix(sast): TemporalSmoother 改为固定高斯 + 可学残差 + L2正则
c5b4a60 fix(sast): 推理重排不再用 ridge_factor 乘初始 TFR 能量
(待提交) fix(sast): sqz_controller 用 node IF + 加宽衰减 + ridge_floor
```
