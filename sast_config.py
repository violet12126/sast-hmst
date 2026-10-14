"""
SAST Configuration & Reproducibility
====================================
SastConfig dataclass + set_seed.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class SastConfig:
    """Unified SAST configuration - single source of truth for all hyperparams."""

    # ── Signal ──
    fs: int = 1000
    max_len: int = 2000

    # ── Model architecture ──
    d_h: int = 96
    n_heads: int = 4
    n_layers: int = 2
    sigma_min: float = 0.5
    sigma_max: float = 15.0
    msst_num: int = 4
    d_cond: int = 32
    f_type_embed_dim: int = 16
    ppn_temperature: float = 0.08
    prototype_temperature: float = 0.1
    dropout: float = 0.1

    # ── Training ──
    epochs: int = 20
    # batch_size 仅在 balanced_p == 0 (关闭均衡采样) 时生效。
    # 旧默认值 2 与 train_sast.py 的 CLI 默认 (8) 长期分歧。SupCon 需要每个类
    # 在 batch 内有 >=2 个样本才有正样本对, 实测 (train 划分, val_split=0.15,
    # 池 196/1786/587/3604/402) 该事件缺席的 batch 占比:
    #   B=2 : 空转 99.9%, 抽水 99.6%   -> 稀疏类基本拿不到梯度
    #   B=8 : 空转 97.6%, 中负荷 84.8%, 抽水 91.6%, 低负荷 32.8%, 高负荷 2.1%
    # 现值 8 与 CLI 对齐; 均衡采样开启时实际 batch 大小 = balanced_p·balanced_k。
    batch_size: int = 8
    # ── 类别均衡采样 (P×K) ──
    # 每 batch 抽 balanced_k 个类 × 每类 balanced_p 个样本, 类内不放回。
    # P=2/K=5 → B=10: 每个出现的类都必有 >=1 正样本对 (实测 0.0% 缺席);
    # 各类每 epoch 出现次数拉平为 n_batches·P = 1314 次
    # (过采样倍数 空转 6.70× / 低负荷 0.74× / 中负荷 2.24× / 高负荷 0.36× / 抽水 3.27×);
    # 每 epoch batch 数 657 (旧 B=8 为 821), 样本覆盖 6570/6575, 吞吐基本不变。
    # balanced_p < 2 会使每类在 batch 内只有 1 个样本 → 无正样本对, 等于没修。
    balanced_p: int = 2
    balanced_k: int = 5
    lr: float = 0.001
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42

    # ── Loss weights ──
    lambda_supcon: float = 1.0
    lambda_entropy: float = 0.1
    lambda_physics: float = 0.5
    lambda_balance: float = 0.5
    lambda_var: float = 0.5
    lambda_consistency: float = 0.3
    lambda_lowfreq: float = 0.05
    lambda_range: float = 0.3
    lambda_anchor: float = 0.2      # 绝对频率锚 (BPF_LINE 49.995±1.5, 2xBPF 99.990±1.5)
    lambda_ratio: float = 0.2       # 谐波比值一致性 (作用在可微 if_hat 上, A2)
    ratio_tol: float = 0.06         # 比值允许的相对偏差 (flat-bottom 半宽)
    presence_frac: float = 0.1      # 分量存在度: 计入"峰"的带内 bin 比例
    w_low_min: float = 0.05         # 分量缺席时 w_range/balance 下界松弛到该值
    # ── 方案 A: 频率约束 (伪物理约束) 降级为诊断 ──
    # False (默认): L_physics / L_anchor / L_ratio 不进入 L_total, 仅写入
    # losses_dict 作诊断。理由见 sast_losses.total_sast_loss 的说明与 docs §14.2:
    # 重排落点不可学, 这三项只能经 σ 操纵测量窗内的能量质心; 且实测五类
    # BPF_LINE 质心仅跨 0.74 Hz < ±1.5 Hz 容差, 锚对任何工况都不触发。
    # True 可一键复现消融。
    freq_constraints_active: bool = False

    # ── Data ──
    data_path: str = '5_dataset.npz'
    val_split: float = 0.15
    max_samples: Optional[int] = None

    # ── New modules ──
    n_sqz_max: int = 4
    resume: Optional[str] = None  # checkpoint path for fine-tuning

    # ── Output ──
    save_dir: str = 'sast_checkpoints'
    viz_every: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SastConfig':
        """Create config from dict (e.g. checkpoint args), ignoring unknown keys."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
