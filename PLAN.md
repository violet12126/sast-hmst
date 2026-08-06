# 实现推理自动挤压次数 (λ) + w_i 平滑(方法1+2) + 低频锐化(方法5)

> **状态: ✅ 全部完成 (2026-08-06)** | commit: (待提交)

## 关键发现
`infer_sast` ([sast_utils.py:278](sast_utils.py#L278)) 调 `model(x, return_all=True)` **未传 `training=False`** → forward 默认 `training=True` → [sast.py:1021](models/sast.py#L1021) 的推理自适应分支(λ/ridge_factor)**从未执行**(死代码)。且 w_i 每帧独立算无时序耦合 → 跳变 → σ 跳 → TFR 竖线。

---

## A. 推理自动挤压次数 λ (request 1) ✅

### A1. `infer_sast` 传 `training=False` ✅
- [sast_utils.py:278](sast_utils.py#L278): `model(x, return_all=True, training=False)`
- 仅可视化路径改; KNN 分类路径(predict_class 等)仍 `training=True` 避免 z_freq 偏移

### A2. `SparseGaussianReassigner.forward` 加多轮 λ ✅
- [sast.py:770](models/sast.py#L770) 增参 `n_sqz_per_bin: Optional[Tensor]=None`, `N_max: int=4`
- `__init__` 加 `N_max` 属性
- 非 None(推理)时迭代挤压:
  ```
  tfr_cur = tfr_mag * ridge_factor
  for r in 1..N_max:
      src = (n_sqz_per_bin >= r).float()
      if src.max()==0: break
      moved = ReassignerFunction.apply(tfr_cur*src, sigma, omega_hat_int, K, F)
      tfr_cur = moved + tfr_cur*(1-src)
  ```
  固定 `omega_final=omegas[:,-1]`; LOW_FREQ(λ=0)完全不挤=保留原貌
- None(训练): 原单 pass 不变

### A3. `SAST.forward` 推理分支算 per-bin 轮数 ✅
- [sast.py:1083](models/sast.py#L1083): 复用 `i_star`
  `lambda_per_bin = lambda_sqz[B_idx_f, i_star, T_idx_f]`
  `n_sqz_per_bin = round(lambda_per_bin * ridge_factor).clamp(0, n_sqz_max).long()`
  传给 reassigner; 存入 result 供诊断
- SAST.__init__ 加 `n_sqz_max: int = 4` 参数

---

## B. w_i 时序平滑 (方法1+2 合并) ✅

**机制**: 新增 `TemporalSmoother` 模块 — 1D depthwise conv 沿时间维, **高斯初始化(归一化, 和=1)**, 可学习。接在 GAT w_i 输出之后、算 σ/λ 之前, **训练+推理都走**。
- **方法1(立刻见效)**: 旧 checkpoint 加载后 conv 是高斯初值 → 等效推理时固定平滑 → 竖线立即消失, 无需重训
- **方法2(根治)**: 可学习, 重训后学到最优时序耦合
- `load_checkpoint` 改 `strict=False` 并打印 missing keys → 旧 checkpoint 缺 conv 权重不报错(用高斯初值)
- flag `use_temporal_smoother`(默认 True), kernel_size 可配(默认 15, σ=3)
- 训练路径也走 → 无 train/infer 不一致; 旧模型继续训练时 conv 自适应

## C. 低频锐化 (方法5) ✅

新增 `lowfreq_sharpness_loss` ([sast_losses.py](models/sast_losses.py)):
- 取 HYDRAULIC(LOW_FREQ) 频段, 算 **tonality = 1/(1+CV(逐帧能量))** (稳态简谐→高, 宽带噪声→低)
- Rényi 熵 × tonality 加权 → **只在低频是简谐时鼓励集中, 宽带仍保留**(不破坏设计"LOW_FREQ 保留展宽")
- 加入 `total_sast_loss`, `lambda_lowfreq`(默认 0.05); 诊断进 losses_dict
- train_sast.py 传 `--lambda_lowfreq` 参数

---

## 改动文件
1. [models/sast.py](models/sast.py): ✅ TemporalSmoother 模块 + 接入 forward; reassigner 多轮 λ; forward 推理分支算 n_sqz_per_bin; SAST.__init__ 加 n_sqz_max
2. [models/sast_losses.py](models/sast_losses.py): ✅ lowfreq_sharpness_loss + 接入 total_sast_loss
3. [sast_utils.py](sast_utils.py): ✅ infer_sast 传 training=False; load_checkpoint strict=False; SastConfig 加 lambda_lowfreq
4. [train_sast.py](train_sast.py): ✅ 传 lambda_lowfreq

## 验证 ✅
- ✅ smoke test: lowfreq_sharpness_loss (harmonic<noise), total_sast_loss (lowfreq_sharp in dict)
- ✅ SAST forward 两侧 (training=True/False) 正常; TemporalSmoother var 0.99→0.19
- ✅ 推理时 n_sqz_per_bin unique=[0,1,2] (λ 自适应 per-bin)
- ✅ 旧 checkpoint 加载成功 (strict=False, wi_smoother.weight 高斯初值)
- ✅ plot_sast_vs_msst 三个样本 (19/20/28) 正常推理, SAST Rényi < MSST Rényi
