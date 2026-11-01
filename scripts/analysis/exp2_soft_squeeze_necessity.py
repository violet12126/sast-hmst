"""实验 2 —— 软挤压的必要性 (纯推理对照, 不重训).

7 策略 (表 5-1 / 表 5-2 的消融):
  B1      : tfr_msst           传统单点 Dirac 硬挤压 (模型前向已产出)
  B2      : w ≡ 1.0            全频段统一硬挤 (GAT 输出钩子直接置 1)
  B3      : w ≡ C_prior        静态物理先验 (3 节点: [0.20, 0.90, 0.95]) (GAT 钩子)
  B4      : w = 0.06 + presence·(C_prior-0.06)   单带解析存在度公式
  B5a     : 去图 + 去 cond_ctx  屏蔽全部 10 条边 (h_new=0) + PPM cond_mod 置零
  B5b     : 留 cond_ctx 去物理边  只保留 CONDITION (OP→物理节点) 3 条边
  Proposed: SAST 完整异构图

四类指标 (无需分类器):
  [1]  软带包络保真度: 软带 (LOW_FREQ) 内, 与「移动平均去噪后的原始 STFT」
       的 2D 余弦相似度 (越高越保真)
  [1b] 软带撕裂度: 带内 top-1 bin 能量占比 (越高越被挤成细线)
  [1c] 硬带聚焦: BPF + 2xBPF 的带内 Rényi-2D 熵 (α=2, 越低越集中)
       + 带内能量加权频率 std (Hz, 越低越窄)  —— 口径与
       sast_inference.evaluate_tf_quality 完全一致
  [2]  软带独立可分性: 软带时间平均谱 (全谱形状) / 统计矩 (宽度隔离)
       的 5 工况 leave-one-out 1-NN 精度 + Fisher 判别比
  [3]  空转 2xBPF 伪峰抑制率 PMR (n=38)

所有干预都在**本脚本内**用前向钩子 / 层内掩码完成, 不改 models/ 源码;
B5a / B5b 为 **zero-shot 掩码** (不重训), 与 Proposed 共用同一 checkpoint。

用法:
  python scripts/analysis/exp2_soft_squeeze_necessity.py \
      --ckpt sast_checkpoints/retrain_e10/sast_v3_e010.pt --tag retrained_ablation
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(r'D:\aifile\SAST')
sys.path.insert(0, str(REPO))

from models.sast_nodes import PUMP_TURBINE_REGIONS, region_weights
from models.sast_graph import CONDITION_EDGE_INDICES, N_EDGES
from models.sast_losses import node_presence
from models.sast import scatter_softmax
from sast_config import SastConfig
from sast_data import load_dataset
from sast_model import create_model

sys.stdout.reconfigure(encoding='utf-8')

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']
# 软/硬带分组从单一真源导出 (SOFT_F_TYPES), 不再硬编码节点名。
# ⚠ 历史可比性: results/exp2/*.json 是 4 节点时期 (软带 LOW_FREQ+BPF_DRIFT,
#   硬带 BPF_LINE+2xBPF) 的产物。B4 撤回后软带只剩 LOW_FREQ、硬带变成
#   BPF+2xBPF, 数字**不可与前一批直接对比**, 必须同 tag 重跑后再比。
from models.sast_nodes import SOFT_F_TYPES as _SOFT_F_TYPES   # noqa: E402
SOFT_NAMES = [r.name for r in PUMP_TURBINE_REGIONS
              if r.f_type in _SOFT_F_TYPES]
HARD_NAMES = [r.name for r in PUMP_TURBINE_REGIONS
              if r.f_type not in _SOFT_F_TYPES]
STRATEGIES = ['B1', 'B2', 'B3', 'B4', 'B5a', 'B5b', 'Proposed']
NODE_NAMES = [r.name for r in PUMP_TURBINE_REGIONS]
C_PRIOR = torch.tensor([r.C_prior for r in PUMP_TURBINE_REGIONS])
FS = 1000.0


# ───────────────────────────────────────────────────────────────
# 指标 (与原脚本一致)
# ───────────────────────────────────────────────────────────────
def band_mask(freqs, f_min, f_max):
    return (freqs >= f_min) & (freqs < f_max)


def moving_avg_1d(x, w):
    """沿最后一维 (时间) 的移动平均, 边界用边缘值填充。x: [..., T] numpy。"""
    pad = w // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, w - 1 - pad)], mode='edge')
    c = np.concatenate(
        [np.zeros(xp.shape[:-1] + (1,), dtype=xp.dtype),
         np.cumsum(xp, axis=-1)], axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def band_cos_sim(tfr, ref, freqs, f_min, f_max):
    """带内 2D 余弦相似度 (逐样本)。tfr/ref: [B,F,T]。"""
    m = band_mask(freqs, f_min, f_max)
    a = tfr[:, m, :].reshape(tfr.shape[0], -1)
    b = ref[:, m, :].reshape(ref.shape[0], -1)
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return (a * b).sum(axis=1) / (na * nb)


def band_top1_frac(tfr, freqs, f_min, f_max):
    """带内 top-1 bin 能量占比 (逐帧平均, 越高越被「撕成细线」)。"""
    m = band_mask(freqs, f_min, f_max)
    band = tfr[:, m, :]                 # [B, Fn, T]
    e = band.sum(axis=1) + 1e-12        # [B, T] 每帧带内总能量
    top = band.max(axis=1)              # [B, T] 每帧最大 bin
    return (top / e).mean(axis=1)       # [B] 每样本平均 top-1 占比


def band_renyi2d_and_std(tfr, freqs, w):
    """带内 Rényi-2D 熵 + 带内能量加权频率 std。

    与 sast_inference.evaluate_tf_quality / models.sast_losses.renyi_2d_loss
    完全同口径: band = tfr * region_weights (单位分解过渡带), 归一化成 (F,T)
    联合分布后 α=2; std 为逐帧频率加权标准差再对时间取均值。
    tfr: [B,F,T] numpy; w: [F] 区域权重。
    """
    m = w > 0.0
    band = tfr[:, m, :] * w[m].view(1, -1, 1)   # torch
    fb = freqs[m]
    eps = 1e-8
    p = band.clamp(min=eps)
    p = p / p.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    h = torch.log((p ** 2).sum(dim=(1, 2)).clamp(min=eps)) / (1.0 - 2)
    wsum = band.sum(dim=1).clamp(min=eps)                     # [B, T]
    mu = (fb.view(1, -1, 1) * band).sum(dim=1) / wsum          # [B, T]
    var = ((fb.view(1, -1, 1) - mu.unsqueeze(1)) ** 2 * band
           ).sum(dim=1) / wsum
    std = torch.sqrt(var.clamp(min=eps)).mean(dim=1)           # [B]
    return h, std


def soft_feature(tfr, freqs):
    """软带独立特征: LOW_FREQ + BPF_DRIFT 的时间平均谱分布 (各归一化后拼接)。"""
    feats = []
    for r in PUMP_TURBINE_REGIONS:
        if r.name not in SOFT_NAMES:
            continue
        m = band_mask(freqs, r.f_min, r.f_max)
        prof = tfr[:, m, :].mean(axis=2)            # [B, Fn] 时间平均谱
        prof = prof / (prof.sum(axis=1, keepdims=True) + 1e-12)
        feats.append(prof)
    return np.concatenate(feats, axis=1)


def soft_moment_feature(tfr, freqs):
    """软带统计矩特征 —— 只测带宽/位置, 隔离「宽度 vs 位置」之争。

    各软带 4 维: [总能量 log1p, 谱质心 Hz, 谱标准差 Hz, 1D Rényi-2 熵]。
    计算 1-NN/Fisher 前须标准化。
    """
    feats = []
    for r in PUMP_TURBINE_REGIONS:
        if r.name not in SOFT_NAMES:
            continue
        m = band_mask(freqs, r.f_min, r.f_max)
        prof = tfr[:, m, :].mean(axis=2)         # [B, Fn] 时间平均谱
        e = prof.sum(axis=1) + 1e-12             # [B] 带内总能量
        p = prof / e[:, None]                    # 归一化谱
        f = freqs[m].astype(np.float64)          # [Fn]
        centroid = (p * f[None, :]).sum(axis=1)                    # [B] Hz
        std = np.sqrt((p * (f[None, :] - centroid[:, None]) ** 2).sum(axis=1))
        ren_1d = -np.log((p ** 2).sum(axis=1) + 1e-12)             # [B] nats
        feats.append(np.stack([np.log1p(e), centroid, std, ren_1d], axis=1))
    return np.concatenate(feats, axis=1)         # [B, 8]


def one_nn_accuracy(feats, labels):
    """leave-one-out 1-NN (最近的其他样本, 余弦相似度)。"""
    X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T                       # [N, N]
    n = S.shape[0]
    S[np.arange(n), np.arange(n)] = -np.inf   # 排除自身
    nn = S.argmax(axis=1)
    return float((labels[nn] == labels).mean())


def fisher_ratio(feats, labels):
    """类间/类内散度迹之比 (越大越可分)。"""
    mu = feats.mean(axis=0)
    Sb = np.zeros((feats.shape[1], feats.shape[1]))
    Sw = np.zeros_like(Sb)
    for c in np.unique(labels):
        xc = feats[labels == c]
        muc = xc.mean(axis=0)
        d = muc - mu
        Sb += len(xc) * np.outer(d, d)
        Sw += (xc - muc).T @ (xc - muc)
    return float(np.trace(Sb) / (np.trace(Sw) + 1e-12))


def pmr_2xbp(tfr, freqs):
    """[90,105) Hz 带内时间平均谱的 Peak-to-Median Ratio, 逐样本 [B]。"""
    m = band_mask(freqs, 90.0, 105.0)
    prof = tfr[:, m, :].mean(axis=2)
    return prof.max(axis=1) / (np.median(prof, axis=1) + 1e-12)


# ───────────────────────────────────────────────────────────────
# 干预: 推理侧钩子 / 边掩码 (不改 models/ 源码)
# ───────────────────────────────────────────────────────────────
def _masked_forward(layer, state):
    """EdgeConditionedGATLayer.forward 的可掩码副本。

    keep=None   -> 走原实现
    keep=[M]    -> 被掩码边在 scatter_softmax 前置 -inf (不参与归一化),
                   消息权重再乘 keep, 等价于「图上没有这条边」
    zero_all    -> 直接返回 h_new=0 (B5a: 所有边都不传消息)
    """
    def fwd(h, edge_feats, edge_src, edge_dst):
        if state['keep'] is None:
            return state['orig'](h, edge_feats, edge_src, edge_dst)
        B, N, _ = h.shape
        M = edge_feats.shape[1]
        if state['zero_all']:
            attn = torch.zeros(B, M, layer.n_heads, device=h.device, dtype=h.dtype)
            return torch.zeros_like(h), attn

        Q = layer.W_q(h).view(B, N, layer.n_heads, layer.d_head)
        K = layer.W_k(h).view(B, N, layer.n_heads, layer.d_head)
        V = layer.W_v(h).view(B, N, layer.n_heads, layer.d_head)
        E = layer.W_e(edge_feats).view(B, M, layer.n_heads, layer.d_head)

        cat_feat = torch.cat([Q[:, edge_dst], K[:, edge_src], E], dim=-1)
        attn_logits = (cat_feat * layer.attn_a.view(1, 1, layer.n_heads, -1)
                       ).sum(dim=-1)
        attn_scores = F.leaky_relu(attn_logits, negative_slope=0.2)
        keep = state['keep'].to(h.device).view(1, M, 1)
        neg = torch.finfo(attn_scores.dtype).min / 2
        attn_scores = attn_scores.masked_fill(keep == 0, neg)
        attn_weights = scatter_softmax(attn_scores, edge_dst, N)
        attn_weights = attn_weights * keep          # 掩码边权重严格为 0

        msg = attn_weights.unsqueeze(-1) * V[:, edge_src]
        h_new = torch.zeros(B, N, layer.n_heads, layer.d_head, device=h.device)
        dst_exp = edge_dst.view(1, M, 1, 1).expand(B, -1, layer.n_heads, layer.d_head)
        h_new = h_new.scatter_add(1, dst_exp, msg).reshape(B, N, layer.d_h)

        h_out = layer.dropout(F.relu(layer.out_proj(h_new)))
        return h_out, attn_weights
    return fwd


def install_interventions(model):
    """注册 PPM/GAT 输出钩子 + 每层 GAT 的掩码副本, 返回可变状态。"""
    st = {
        'gat': {'mode': None, 'const': None, 'presence': None, 'cprior': None},
        'ppm': {'zero_cond_mod': False},
        'edge': [],
    }

    def ppm_hook(mod, inp, out):
        h_enh, cprior, ge, gn, cs = out
        st['gat']['cprior'] = cprior.detach()
        if st['ppm']['zero_cond_mod']:
            # B5a: cond_mod 置零 == C_prior 退回 sigmoid(C_prior_logit)
            raw = torch.sigmoid(mod.C_prior_logit).view(1, -1).expand_as(cprior)
            return (h_enh, raw, ge, gn, cs)
        return out

    def gat_hook(mod, inp, out):
        w, A = out
        mode = st['gat']['mode']
        if mode is None:
            return out
        if mode == 'ones':                       # B2
            w2 = torch.ones_like(w)
        elif mode == 'const':                    # B3
            w2 = st['gat']['const'].to(w.device).view(1, -1).expand_as(w)
        elif mode == 'presence':                 # B4
            cp = st['gat']['cprior']
            p = st['gat']['presence'].to(w.device)
            assert p.shape == w.shape, (p.shape, w.shape)
            w2 = 0.06 + p * (cp - 0.06)
        else:
            raise ValueError(mode)
        return (w2, A)

    model.ppm.register_forward_hook(ppm_hook)
    model.gat.register_forward_hook(gat_hook)

    for layer in model.gat.layers:
        state = {'keep': None, 'zero_all': False, 'orig': layer.forward}
        st['edge'].append(state)
        layer.forward = _masked_forward(layer, state)
    return st


def set_intervention(st, mode=None, keep=None, zero_all=False, zero_cond_mod=False,
                     const=None, presence=None):
    st['gat']['mode'] = mode
    st['gat']['const'] = const
    st['gat']['presence'] = presence
    st['ppm']['zero_cond_mod'] = zero_cond_mod
    for s in st['edge']:
        s['keep'] = keep
        s['zero_all'] = zero_all


# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str,
                    default='sast_checkpoints/retrain_e10/sast_v3_e010.pt')
    ap.add_argument('--tag', type=str, default='retrained_ablation')
    ap.add_argument('--n', type=int, default=None, help='cap val samples (None=all)')
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--strategies', type=str, default=','.join(STRATEGIES),
                    help='逗号分隔, 可选 ' + ','.join(STRATEGIES))
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
    for s in strategies:
        assert s in STRATEGIES, s
    want = set(strategies)
    # 任何策略都需要一次干净前向 (raw/msst/freqs + B4 的 presence)
    need_base = True

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = SastConfig.from_dict(ck['args'])

    model = create_model(cfg, dev)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    st = install_interventions(model)

    ds = load_dataset(str(REPO / cfg.data_path), max_samples=None,
                      max_len=cfg.max_len, val_split=cfg.val_split, seed=cfg.seed)
    X, y = ds['val_X'], ds['val_y']
    if args.n is not None:
        idx = np.random.default_rng(0).choice(len(X), size=min(args.n, len(X)),
                                              replace=False)
        X, y = X[idx], y[idx]
    N = len(X)
    print(f'val: {N} 样本 | 类分布 {np.bincount(y, minlength=5).tolist()} '
          f'(空转={int((y == 0).sum())})')
    print(f'节点顺序: {NODE_NAMES}')
    print(f'策略: {strategies}')

    # 一次热身前向拿 freqs / T_msst
    with torch.no_grad():
        warm = model(torch.from_numpy(X[:2]).float().to(dev), training=False)
    freqs_t = warm['freqs']
    freqs = freqs_t.cpu().numpy()
    T_msst = warm['tfr_raw'].shape[-1]
    band_w = {r.name: region_weights(freqs_t, [r])[0].cpu()
              for r in PUMP_TURBINE_REGIONS}

    # 掩码定义
    cond_keep = torch.zeros(N_EDGES)
    cond_keep[CONDITION_EDGE_INDICES] = 1.0
    zero_keep = torch.zeros(N_EDGES)

    # 累积器
    m1 = {s: {'cos': {b: 0.0 for b in SOFT_NAMES},
              'top1': {b: 0.0 for b in SOFT_NAMES},
              'renyi': {b: 0.0 for b in HARD_NAMES},
              'std': {b: 0.0 for b in HARD_NAMES}} for s in strategies}
    m3 = {s: {'pmr_sum': 0.0, 'n': 0} for s in strategies}
    feats = {s: [] for s in strategies}
    feats_mom = {s: [] for s in strategies}
    wst = {s: {'sum': np.zeros(len(NODE_NAMES)), 'n': 0,
               'cls_sum': np.zeros((5, len(NODE_NAMES))),
               'cls_n': np.zeros(5)} for s in strategies}
    raw_ref = {'cos': {b: 0.0 for b in SOFT_NAMES}}

    def rec_w(name, w, yb):
        if w is None:
            return
        wm = w.mean(axis=2)                     # [B, N] 时间平均
        d = wst[name]
        d['sum'] += wm.sum(axis=0)
        d['n'] += len(wm)
        for c in np.unique(yb):
            sel = yb == c
            d['cls_sum'][c] += wm[sel].sum(axis=0)
            d['cls_n'][c] += int(sel.sum())

    def add_metrics(name, t, yb, raw_den):
        for b in SOFT_NAMES:
            r = next(rr for rr in PUMP_TURBINE_REGIONS if rr.name == b)
            m1[name]['cos'][b] += float(
                band_cos_sim(t, raw_den, freqs, r.f_min, r.f_max).sum())
            m1[name]['top1'][b] += float(
                band_top1_frac(t, freqs, r.f_min, r.f_max).sum())
        tt = torch.from_numpy(t)
        for b in HARD_NAMES:
            h, s = band_renyi2d_and_std(tt, freqs_t.cpu(), band_w[b])
            m1[name]['renyi'][b] += float(h.sum())
            m1[name]['std'][b] += float(s.sum())
        feats[name].append(soft_feature(t, freqs))
        feats_mom[name].append(soft_moment_feature(t, freqs))
        idle = yb == 0
        if idle.any():
            p = pmr_2xbp(t[idle], freqs)
            m3[name]['pmr_sum'] += float(p.sum())
            m3[name]['n'] += int(idle.sum())

    plot_idx = []
    for c in [0, 3]:
        ci = np.where(y == c)[0]
        if len(ci):
            plot_idx.append(int(ci[0]))
    plot_cache = {}
    plot_keys = [k for k in ['B1', 'B2', 'B3', 'B4', 'B5a', 'B5b', 'Proposed']
                 if k in want]

    for start in range(0, N, args.batch):
        end = min(start + args.batch, N)
        batch_x = torch.from_numpy(X[start:end]).float().to(dev)
        yb = y[start:end]
        B = end - start
        tb = {}

        with torch.no_grad():
            # ── 干净前向: raw / msst / Proposed ──
            set_intervention(st)
            res = model(batch_x, training=False)
            raw = res['tfr_raw'].cpu().numpy()
            msst = res['tfr_msst'].cpu().numpy()
            enh = res['tfr_enhanced'].cpu().numpy()
            tb['B1'] = msst
            tb['Proposed'] = enh
            rec_w('B1', None, yb)
            rec_w('Proposed', res['w_i'].cpu().numpy(), yb)

            raw_den = moving_avg_1d(raw, 5)

            # ── B2: w ≡ 1.0 ──
            if 'B2' in want:
                set_intervention(st, mode='ones')
                r2 = model(batch_x, training=False)
                tb['B2'] = r2['tfr_enhanced'].cpu().numpy()
                rec_w('B2', r2['w_i'].cpu().numpy(), yb)

            # ── B3: w ≡ C_prior ──
            if 'B3' in want:
                set_intervention(st, mode='const', const=C_PRIOR)
                r3 = model(batch_x, training=False)
                tb['B3'] = r3['tfr_enhanced'].cpu().numpy()
                rec_w('B3', r3['w_i'].cpu().numpy(), yb)

            # ── B4: w = 0.06 + presence·(C_prior - 0.06) ──
            if 'B4' in want:
                pres = node_presence(torch.from_numpy(raw).to(dev), freqs_t,
                                     frac=0.1)                 # [B,N,T]
                pres_bt = pres.permute(0, 2, 1).reshape(B * T_msst, -1)
                set_intervention(st, mode='presence', presence=pres_bt)
                r4 = model(batch_x, training=False)
                tb['B4'] = r4['tfr_enhanced'].cpu().numpy()
                rec_w('B4', r4['w_i'].cpu().numpy(), yb)

            # ── B5b: 只留 CONDITION (OP→物理节点) 4 条边 ──
            if 'B5b' in want:
                set_intervention(st, keep=cond_keep)
                r5b = model(batch_x, training=False)
                tb['B5b'] = r5b['tfr_enhanced'].cpu().numpy()
                rec_w('B5b', r5b['w_i'].cpu().numpy(), yb)

            # ── B5a: 全部 11 条边屏蔽 + PPM cond_mod 置零 ──
            if 'B5a' in want:
                set_intervention(st, keep=zero_keep, zero_all=True,
                                 zero_cond_mod=True)
                r5a = model(batch_x, training=False)
                tb['B5a'] = r5a['tfr_enhanced'].cpu().numpy()
                rec_w('B5a', r5a['w_i'].cpu().numpy(), yb)

            set_intervention(st)          # 复位

            # ── 指标 ──
            for s in strategies:
                add_metrics(s, tb[s], yb, raw_den)
            for b in SOFT_NAMES:
                r = next(rr for rr in PUMP_TURBINE_REGIONS if rr.name == b)
                raw_ref['cos'][b] += float(
                    band_cos_sim(raw_den, raw_den, freqs, r.f_min, r.f_max).sum())

            for pi in plot_idx:
                if start <= pi < end and pi not in plot_cache:
                    j = pi - start
                    plot_cache[pi] = {'raw': raw[j], 'B1': msst[j],
                                      'Proposed': enh[j]}
                    for k in plot_keys:
                        if k not in ('B1', 'Proposed'):
                            plot_cache[pi][k] = tb[k][j]

        if (start // args.batch) % 20 == 0:
            print(f'  {end}/{N}')

    # ── 汇总 ──
    out = {'ckpt': args.ckpt, 'N': N,
           'class_dist': np.bincount(y, minlength=5).tolist(),
           'nodes': NODE_NAMES,
           'edge_retrain': 'zero-shot 掩码 (不重训)'}

    def avg(d, b):
        return round(d[b] / N, 4)

    out['metric1_cos_sim'] = {s: {b: avg(m1[s]['cos'], b) for b in SOFT_NAMES}
                              for s in strategies}
    out['metric1_top1_frac'] = {s: {b: avg(m1[s]['top1'], b) for b in SOFT_NAMES}
                                for s in strategies}
    out['metric1_raw_self_cos'] = {b: avg(raw_ref['cos'], b) for b in SOFT_NAMES}

    # [1c] 硬带: 逐带 + 两带均值
    out['metric1c_hard_renyi'] = {
        s: {b: avg(m1[s]['renyi'], b) for b in HARD_NAMES} for s in strategies}
    out['metric1c_hard_std_hz'] = {
        s: {b: avg(m1[s]['std'], b) for b in HARD_NAMES} for s in strategies}
    for s in strategies:
        out['metric1c_hard_renyi'][s]['mean'] = round(
            np.mean([out['metric1c_hard_renyi'][s][b] for b in HARD_NAMES]), 4)
        out['metric1c_hard_std_hz'][s]['mean'] = round(
            np.mean([out['metric1c_hard_std_hz'][s][b] for b in HARD_NAMES]), 4)

    out['metric2'] = {}
    out['metric2_moments'] = {}
    for s in strategies:
        F_ = np.concatenate(feats[s], axis=0)
        out['metric2'][s] = {'1nn_acc': round(one_nn_accuracy(F_, y), 4),
                             'fisher': round(fisher_ratio(F_, y), 4)}
        Fm = np.concatenate(feats_mom[s], axis=0)
        Fm = (Fm - Fm.mean(axis=0)) / (Fm.std(axis=0) + 1e-12)
        out['metric2_moments'][s] = {
            '1nn_acc': round(one_nn_accuracy(Fm, y), 4),
            'fisher': round(fisher_ratio(Fm, y), 4)}

    out['metric3_idle_pmr'] = {s: round(m3[s]['pmr_sum'] / max(1, m3[s]['n']), 4)
                               for s in strategies}
    out['metric3_idle_n'] = m3[strategies[0]]['n']

    # w_i 统计 (表 5-1)
    out['w_stats'] = {}
    for s in strategies:
        d = wst[s]
        if d['n'] == 0:
            out['w_stats'][s] = None
            continue
        means = d['sum'] / d['n']
        cls_means = np.array([d['cls_sum'][c] / d['cls_n'][c]
                              if d['cls_n'][c] else np.full(len(NODE_NAMES), np.nan)
                              for c in range(5)])
        out['w_stats'][s] = {
            'node_mean': {NODE_NAMES[i]: round(float(means[i]), 4)
                          for i in range(len(NODE_NAMES))},
            'class_mean': {CLASS_NAMES[c]: {
                NODE_NAMES[i]: (round(float(cls_means[c, i]), 4)
                                if np.isfinite(cls_means[c, i]) else None)
                for i in range(len(NODE_NAMES))} for c in range(5)},
            'delta_w_BPF': round(float(
                np.nanmax(cls_means[:, NODE_NAMES.index('BPF')]) -
                np.nanmin(cls_means[:, NODE_NAMES.index('BPF')])), 4),
            'hard_mean': round(float(np.mean([means[NODE_NAMES.index(b)]
                                              for b in HARD_NAMES])), 4),
        }

    # ── 打印 ──
    print('\n' + '=' * 84)
    print('实验 2 消融读数 (N=%d)' % N)
    print('=' * 84)
    print('\n[1] 软带包络保真度 cos (越高越保真):')
    for s in strategies:
        v = out['metric1_cos_sim'][s]
        print(f'    {s:<9s} ' + '  '.join(f'{b}={v[b]:.4f}' for b in SOFT_NAMES))
    print('\n[1b] 软带 top-1 bin 占比 (撕裂度, 越低越好):')
    for s in strategies:
        v = out['metric1_top1_frac'][s]
        print(f'    {s:<9s} ' + '  '.join(f'{b}={v[b]:.4f}' for b in SOFT_NAMES))
    print('\n[1c] 硬带 Rényi-2D 熵 / 带内 std Hz (越低越聚焦):')
    for s in strategies:
        v = out['metric1c_hard_renyi'][s]
        sd = out['metric1c_hard_std_hz'][s]
        print(f'    {s:<9s} ' + '  '.join(
            f'{b}={v[b]:.3f}/{sd[b]:.3f}' for b in HARD_NAMES) +
            f'  | mean={v["mean"]:.3f}/{sd["mean"]:.3f}')
    print('\n[2a] 软带 Fisher (全谱形状 / 统计矩):')
    for s in strategies:
        print(f'    {s:<9s} {out["metric2"][s]["fisher"]:.4f} / '
              f'{out["metric2_moments"][s]["fisher"]:.4f}')
    print('\n[2b] 软带 1-NN (统计矩):')
    for s in strategies:
        print(f'    {s:<9s} {out["metric2_moments"][s]["1nn_acc"]:.4f}')
    print('\n[3] 空转 2xBPF PMR (越低越好, n=%d):' % out['metric3_idle_n'])
    for s in strategies:
        print(f'    {s:<9s} {out["metric3_idle_pmr"][s]:.4f}')
    print('\n[表 5-1] w_i 统计:')
    for s in strategies:
        w = out['w_stats'][s]
        if w is None:
            print(f'    {s:<9s} (无 w, 等效硬挤)')
            continue
        print(f'    {s:<9s} Δw(BPF)={w["delta_w_BPF"]:.3f} '
              f'hard_mean={w["hard_mean"]:.3f}  '
              + ' '.join(f'{k}={v:.3f}' for k, v in w['node_mean'].items()))

    # ── 存 json ──
    outdir = REPO / 'results' / 'exp2'
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / f'exp2_{args.tag}.json'
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已存: {jpath}')

    # ── 出图 ──
    if not args.no_plot and plot_cache:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        BANDS = [('LOW_FREQ', 2.0, 18.0),
                 ('BPF family', 42.0, 53.0),
                 ('2xBPF', 90.0, 105.0)]
        rows = [('raw', 'Raw STFT')] + [(k, k) for k in plot_keys]

        def _img(tfr, freqs, f_min, f_max):
            m = (freqs >= f_min) & (freqs < f_max)
            band = np.log1p(tfr[m, :])
            lo, hi = np.percentile(band, [2, 98])
            if hi - lo < 1e-6:
                hi = lo + 1e-6
            return band, lo, hi

        for pi, cache in plot_cache.items():
            cls = y[pi]
            fig, axes = plt.subplots(len(rows), len(BANDS),
                                     figsize=(len(BANDS) * 4.4, len(rows) * 2.4),
                                     sharex=True, squeeze=False)
            for ri, (key, sname) in enumerate(rows):
                for bi, (bname, fmin, fmax) in enumerate(BANDS):
                    ax = axes[ri, bi]
                    band, lo, hi = _img(cache[key], freqs, fmin, fmax)
                    ax.imshow(band, aspect='auto', origin='lower',
                              extent=[0, cache[key].shape[1] / FS, fmin, fmax],
                              cmap='magma', norm=Normalize(lo, hi))
                    if ri == 0:
                        ax.set_title(bname, fontsize=11)
                    if bi == 0:
                        ax.set_ylabel(f'{sname}\nHz', fontsize=9)
                    if ri == len(rows) - 1:
                        ax.set_xlabel('t (s)', fontsize=9)
            fig.suptitle(f'Exp2 消融 — {CLASS_NAMES[cls]} (val#{pi})', fontsize=12)
            fig.tight_layout()
            p = outdir / f'exp2_{args.tag}_class{cls}_id{pi}.png'
            fig.savefig(p, dpi=110)
            plt.close(fig)
            print(f'已出图: {p}')

    print('完成.')


if __name__ == '__main__':
    main()
