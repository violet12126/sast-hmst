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
    # sigma_max = 3.0 (= sigma_soft_cap). 实际 sigma_i = 0.5 + (1-w_i)*2.5 ∈ [0.5,3.0],
    # 故 15.0 从未被达到; 而 K = ceil(kernel_radius(3.0)*sigma_max) 被算成 45,
    # 白跑 5 倍循环 (base kernel 与 Python fallback 用满 ±K; opt kernel 的
    # K_eff=min(ceil(6σ),K) 只在 σ>1.5 时被新 K=9 截断)。
    # 附带修掉 deploy 侧 sigma soft-cap 不一致: 导出用 delta = sigma_max - sigma_min,
    # sigma_max=3.0 时 = 2.5, 与训练侧 min(sigma_max, soft_cap)-sigma_min 相同。
    # 代价: σ=3 时核被截到 ±3σ, 丢 0.27% 高斯质量 (重归一化)。**现只影响
    # LOW_FREQ**: 它是唯一的软带, w∈[0.10,0.45] -> σ∈[1.875,2.75] > 1.5 会被截。
    # 硬带 σ = 0.5 + (1-w)*2.5; BPF_DRIFT 现为硬带 w∈[0.60,0.85] -> σ∈[0.875,1.5],
    # K_eff=ceil(6σ) 最大正好 9 = K, 不触发截断; BPF_LINE/2xBPF 的 σ≤0.875 更不受影响。
    sigma_max: float = 3.0
    # ── MSST 迭代次数 (IF 固定点 lookup 层数) ──
    # 4 -> 1: 4 次迭代会把 BPF_LINE 带内 bin 的 IF 从 49.6-50.2 Hz 一路拖到
    # 45.8-48.3 Hz (掉进 BPF_DRIFT 的 46-48 Hz 主瓣), 迭代越深越偏 —— 即
    # 「IF 精化」在本数据上对锁频线是**有害**的 (实测 iter0 中心 47.1-49.1 vs
    # iter3 45.8-48.3, 五类全中; 带内落点越界比例 5.5%/14% → 10%/25%)。
    # num=1 即用 STFT 的一阶 IF 估计 (标准 SST 口径), 与基线 MSST 图一致。
    msst_num: int = 1
    # ── MSST 窗长 (样本点) ──
    # 512 -> 1024。46.9 Hz 漂移分量与 50 Hz 锁频线只差 3.1 Hz, 而 512 窗的
    # 主瓣半宽约 2 Hz —— 两者在 [48.5,53) 里根本分不开, 该带 bin 的 IF 被
    # 漂移峰右翼拽低。实测 (class0, hl/num=1, node_if 众数):
    #   hl= 320 → 47.95 Hz (961/2000 帧钉在 48 Hz 附近, 即被拽到底)
    #   hl= 512 → 48.88 Hz
    #   hl=1024 → 49.54 Hz, 全帧落在 [48.53, 51.33], 越界 0%
    # 代价: 有效时间分辨率 0.32 s → 1.02 s (信号长 2 s)。取 1024 是因为
    # "46.9 Hz 起伏线 与 50 Hz 直线能不能分开"是硬判据, 时间分辨率是软的。
    # 显式给出该字段后, scripts/plot/plot_msst_tfr.py 直接读它 —— 基线与推理
    # 不再是两套窗长 (旧版一边 320 一边 512, 同一信号相关只有 0.58)。
    msst_hlength: Optional[int] = 1024
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
    # λ_supcon 1.0 -> 0.2: 不做分类的前提下, SupCon 只作为「让 cond_ctx 保持
    # 工况区分度」的辅助正则, 优化重心回到时频分布本身。
    # 实测加权贡献 (1.0 时): supcon 1.440 (62.3%) / RE_2D 0.798 (34.5%) / 其余 ≤0.05。
    # 降到 0.2 后 supcon ≈ 0.288 (~26%), RE_2D 成为最大单项。
    # ⚠ 代价: FreqEncoder 的**唯一**梯度来源就是 supcon_loss, 降 5 倍即梯度降 5 倍;
    #   故 sast_inference 的 KNN 验证精度 (只测分类能力) 会下降 —— 这是预期内的,
    #   时频质量改由 evaluate_tf_quality 观测。
    lambda_supcon: float = 0.2
    lambda_entropy: float = 0.1
    # ── 软带相对熵降护栏 (L_soft_ent) ──
    # 只允许增强谱比原始谱少 delta_H nats 的 Rényi-2D 熵。**作用域随
    # SOFT_F_TYPES 收窄**: 现在只有 LOW_FREQ/HYDRAULIC 一条软带 —— BPF_DRIFT
    # 已被重新判为窄带调频脊线 (单帧本征宽度 0.686 Hz, 与 BPF_LINE 0.584 /
    # 2xBPF 0.650 同量级) 归入硬带, 不再受本项保护。
    # delta_H 由合成对照标定: 软带 FWHM 2-3 Hz -> σ_int≈1.0 Hz -> 0.55 nats
    # (区间 0.36-0.79 对应 σ_int 0.85-1.27)。实测底噪跨 500 倍熵降变化 <0.6%,
    # 即共模抵消成立, 固定 delta_H 可用。
    # 保留 0.5: LOW_FREQ 是真正的宽带湍流云团 (2.789 Hz, 6 倍于锁频带), 需要
    # 一个显式的"别挤"回拉力, 与 lambda_w_center 二次项同向。
    lambda_soft_ent: float = 0.5
    delta_H: float = 0.55
    lambda_physics: float = 0.5
    lambda_balance: float = 0.5
    lambda_var: float = 0.5
    # ── w_i 逐帧平稳性 (0.3 -> 1.0) ──
    # 实测 (results/w_curve/, e15 ckpt): 帧级 |Δw| 只有 0.009 (无毛刺), 但
    # **样本内极差** ptp 可达 BPF_LINE 0.86 / 2xBPF 0.58, 时间尺度 0.4-1.0 s
    # —— 0.1 s 的 avg_pool 结构低通压不住这个尺度。故把权重提回 1.0, 并在
    # w_consistency_loss 里加 250 ms 粗尺度差分项 (见 sast_losses)。
    lambda_consistency: float = 1.0
    lambda_lowfreq: float = 0.05
    # ── w_range 逐帧范围 (0.3 -> 1.0) + 平方 hinge ──
    # 平底 hinge 在界内梯度为 0、界外梯度恒定, 会在边界上来回"打铃"; 平方 hinge
    # 在边界处梯度连续衰减到 0, 配合 λ=1.0 把越过 w_range 的帧按越界幅度平方惩罚。
    lambda_range: float = 1.0
    # ── 先验中心二次惩罚 (L_w_center) ──
    # 拉 w_i 回逐节点先验中心 C_prior: mean((w_i - c_n)^2)。配合有界 logit 后,
    # w_range/balance 的平底 hinge 在范围内梯度为 0, RE_2D 的溢出串扰可把软带 w
    # 一路顶到 w_range_high (实测 BPF_DRIFT 顶死 0.50)。二次项提供连续回拉力,
    # 使平衡点落在中心附近的小偏离处而非墙上。
    # 与 w_range 的 presence 机制一致: 分量缺席时中心下沉到 w_low_min, 避免把
    # 2xBPF 在空转强行拉回 0.90 (配合 renyi_2d_loss 的 presence 加权)。
    # 权重较小: 只需盖过 RE_2D 串扰 (实测 ~1e-6 量级), 不盖过 supcon/RE_2D 的
    # 真信号, 保留 BPF_DRIFT 的跨类/跨样本方差。
    lambda_w_center: float = 0.05
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
    # ── 自适应挤压轮数 (可微软门控) ──
    # n_sqz_max 现为软门控的 M (最大轮数): r = clamp(w_i*M, 0, M),
    # g_m = clamp(r - (m-1), 0, 1)。取代旧的整数 n_sqz_per_bin 掩码
    # (那条路径训练不启用、round() 不可微、且 clamp(1,..) 使 0 轮不可达)。
    # M 由 3 降为 2: 实测 (docs §16.4.4) R=1->2 有实质收益 (BPF_LINE Rényi
    # 7.1199->7.0178), 而 R=2->3->4 完全进入平台期 (7.0178/7.0122/7.0128)。
    # 取 M=2 既在拐点上, 又把单步重排 192ms 降到 128ms (省 33%)。
    # 实测代价 (B=10,F=1000,T=2000): 重排本体 M=1 71ms -> M=3 192ms (+109ms);
    # 端到端 46s -> 51s per epoch (=+11%, 与 MSST 等环节重叠)。
    # ⚠ 整模型显存实测已占 6.5-7.5 GB / 8 GB, 加大 M 或 batch 前需重估。
    n_sqz_max: int = 2
    # 门控形状: 'clamp' (推荐, 无参且能真正到 0 轮) | 'sigmoid'
    gate_mode: str = 'clamp'
    # 是否施加 ridge 空间参与因子。**默认 False** —— 见 docs §16.4.2 / §16.4.1。
    # WHY: 在输出混合递推下, part<1 意味着每轮把 (1-part) 的原始能量留在原位,
    #   而 ridge 恰恰把"最该被搬走的侧翼 bin"压到 floor 0.15, 于是残余
    #   ∏(1-g_m·ridge) 洗不掉。实测 (20 epoch 全量, 同 seed):
    #     ridge=1: hard Rényi 7.853 / hard 带内 std 0.873
    #     ridge=0: hard Rényi 7.721 / hard 带内 std 0.659   <- 两个轴都更好
    #   Val acc 两组都是 1.000 (已饱和, 无损失)。ridge 与架构不匹配, 不是参数问题。
    # 保留该开关是为了论文里的消融可复现; 设为 True 可复现旧行为。
    train_ridge: bool = False
    # 是否把 FreqEncoder 的输入限制在 4 个物理带内。默认 True。
    # WHY: 带外占 92% 的 bin 且携带工况信息 (带外 1-NN 0.800 vs chance 0.200),
    #   编码器可据此把 SupCon 刷到 0 而不依赖 GAT —— 实测 sc 塌到 0.000、
    #   Val acc 饱和在 1.000。屏蔽后梯度才必须经过物理节点的重排。
    freq_encoder_masked: bool = True
    # 消融: 断开门控路径 / 断开 sigma 路径 (回答门控是否只是 w_range 的重参数化)
    gate_detach: bool = False
    sigma_detach: bool = False
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
