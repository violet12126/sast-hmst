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
    batch_size: int = 2
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
