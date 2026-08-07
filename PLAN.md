# w_i 跳变根治方案: 频段内时序一致性

> **状态: ✅ 已实施 (2026-08-07)** | 分支: `fix/w-i-balance-variance`

## 核心思路

**一旦模型确定了频率脊线, 该频段内的挤压宽度 w_i 就应该近似恒定。**

- BPF 频段: 所有帧用差不多的 w_i (如 0.8)
- 2×BPF 频段: 所有帧用差不多的 w_i (如 0.6, 可与 BPF 不同)
- LOW_FREQ 频段: 所有帧用差不多的 w_i (如 0.2)
- **不同频段之间可以有差异, 但同一频段内部不能跳变**

直接用 loss 惩罚 w_i 沿时间的方差, 而不是靠卷积层间接平滑。

---

## 实施内容

### ✅ 新增: `w_consistency_loss`
```python
# models/sast_losses.py
def w_consistency_loss(w_i):
    """惩罚各物理节点内 w_i 沿时间的方差."""
    per_node_var = w_i.var(dim=2)    # [B, N_phys]
    return per_node_var.mean()
```
- 常数 w_i → loss=0
- 跳变 w_i → loss 大
- 与 w_variance_loss 互补: 一个管内聚(时间), 一个管分化(节点间)

### ❌ 移除: TemporalSmoother
- 删除整个 `TemporalSmoother` 类 (~50行)
- 删除 `SAST.__init__` 的 `use_temporal_smoother` + `smoother_kernel` 参数
- 删除 `SAST.forward` 的 `wi_smoother` 调用
- 删除 `train_sast.py` 的 `smoother_reg` 计算和 `--smoother_kernel` CLI

### ❌ 移除: temporal_smoothness_loss
- 被 `w_consistency_loss` 替代 (全局方差 > 相邻帧差分)
- 删除 `lambda_smooth` 所有引用

### ✅ 保留
| 组件 | 原因 |
|------|------|
| balance_loss | 防 w_i 饱和到 0 或 1 |
| w_variance_loss | 鼓励节点间 w_i 分化 (与新 loss 互补) |
| lowfreq_sharpness_loss | 管 TFR 浓度, 与 w_i 一致性无关 |
| sqz_controller (node IF + ridge_floor=0.15) | 管频率轴覆盖 (100/200Hz 脊线) |
| 能量修复 (tfr_cur = tfr_mag) | 管能量不丢失 |
| SupCon / entropy / physics | 核心训练目标 |

---

## 改动文件
1. [models/sast_losses.py](models/sast_losses.py): 新增 `w_consistency_loss`, 删除 `temporal_smoothness_loss`, 更新 `total_sast_loss` 签名
2. [models/sast.py](models/sast.py): 删除 `TemporalSmoother` 类, 删除 `wi_smoother` 创建和调用
3. [sast_utils.py](sast_utils.py): SastConfig 删除 `smoother_kernel`/`lambda_smooth`, 新增 `lambda_consistency`
4. [train_sast.py](train_sast.py): CLI 删除 `--smoother_kernel`/`--lambda_smooth`, 新增 `--lambda_consistency`, 删除 `smoother_reg`, 更新日志

## 训练命令
```bash
python train_sast.py --epochs 50 --device cuda \
  --lambda_consistency 0.3 --lambda_var 0.5 \
  --lambda_balance 0.5 --lambda_lowfreq 0.05
```

## 验证
- ✅ constant w_i → w_consistency_loss = 0.0
- ✅ alternating w_i → w_consistency_loss = 0.207
- ✅ SAST import OK (TemporalSmoother 移除后)
- ✅ 所有 loss 分量 smoke test 通过
