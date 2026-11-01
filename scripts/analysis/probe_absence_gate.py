"""缺席门控的单元核对 (只读, 不训练).

核对三件事:
  (1) absence_gate 的逐节点行为: 只有 2xBPF 启用; e=0.084 -> a=1, e=0.275 -> a=0,
      BPF / LOW_FREQ 恒为 0。
  (2) 加载工况 (a=0) 时 w_range / balance 与"不传 absence"逐位相同 —— 不误伤加载。
  (3) 空转 2xBPF (a=1) 时 0.973 被上界 0.20 惩罚, 而 0.10 不被惩罚。

用法: conda run -n my_work python scripts/analysis/probe_absence_gate.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.sast_losses import (absence_gate, balance_loss, renyi_2d_loss,   # noqa: E402
                               w_range_loss)
from models.sast_nodes import PUMP_TURBINE_REGIONS                           # noqa: E402

REG = PUMP_TURBINE_REGIONS
N = len(REG)
NAMES = [r.name for r in REG]


def section(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def main():
    torch.manual_seed(0)
    T = 50

    # ── (1) absence_gate ──
    section('(1) absence_gate: 逐节点门限')
    print(f'节点门限 lo/hi: '
          + '  '.join(f'{r.name}=({r.p_absent_lo:.2f},{r.p_absent_hi:.2f})'
                      for r in REG))
    probes = {
        '空转 (LOW 0.195 / BPF 0.211 / 2x 0.084)': [0.195, 0.211, 0.084],
        '低负荷 (0.148 / 0.156 / 0.275)': [0.148, 0.156, 0.275],
        '高负荷 (0.147 / 0.126 / 0.311)': [0.147, 0.126, 0.311],
    }
    for tag, vals in probes.items():
        p = torch.tensor(vals).view(1, N, 1).repeat(1, 1, T)
        a = absence_gate(p, REG)[0, :, 0]
        print(f'  {tag:42s} -> a = ['
              + ' '.join(f'{x:.3f}' for x in a.tolist()) + ']')
    ok1 = True

    p_idle = torch.tensor([0.195, 0.211, 0.084]).view(1, N, 1).repeat(1, 1, T)
    a_idle = absence_gate(p_idle, REG)[0, :, 0]
    p_load = torch.tensor([0.148, 0.156, 0.275]).view(1, N, 1).repeat(1, 1, T)
    a_load = absence_gate(p_load, REG)[0, :, 0]
    checks = [
        ('空转 2xBPF a == 1', float(a_idle[2]) == 1.0),
        ('加载 2xBPF a == 0', float(a_load[2]) == 0.0),
        ('BPF 恒 a == 0 (空转)', float(a_idle[1]) == 0.0),
        ('BPF 恒 a == 0 (加载)', float(a_load[1]) == 0.0),
        ('LOW_FREQ 恒 a == 0', float(a_idle[0]) == 0.0 and float(a_load[0]) == 0.0),
    ]
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
        ok1 &= ok

    # ── 构造 w_i ──
    B = 1
    w_low = torch.tensor([r.w_range_low for r in REG])
    w_high = torch.tensor([r.w_range_high for r in REG])

    def w_tensor(vals):
        return torch.tensor(vals).view(1, N, 1).repeat(B, 1, T)

    # ── (2) 加载工况不误伤 ──
    section('(2) 加载工况 (a=0): 与不传 absence 逐位相同')
    w_loaded = w_tensor([0.106, 0.980, 0.980])
    p_loaded = p_load
    a_loaded = absence_gate(p_loaded, REG)
    lr_ref = w_range_loss(w_loaded, w_low, w_high, presence=p_loaded, w_low_min=0.05)
    lr_new = w_range_loss(w_loaded, w_low, w_high, presence=p_loaded,
                          absence=a_loaded, w_low_min=0.05, w_high_min=0.20)
    lb_ref = balance_loss(w_loaded, w_min=w_low, w_max=w_high,
                          presence=p_loaded, w_min_floor=0.05)
    lb_new = balance_loss(w_loaded, w_min=w_low, w_max=w_high,
                          presence=p_loaded, absence=a_loaded,
                          w_min_floor=0.05, w_max_floor=0.20)
    print(f'  L_range   旧 = {lr_ref.item():.6f}   新 = {lr_new.item():.6f}')
    print(f'  L_balance 旧 = {lb_ref.item():.6f}   新 = {lb_new.item():.6f}')
    ok2 = abs(lr_ref.item() - lr_new.item()) < 1e-9 and \
        abs(lb_ref.item() - lb_new.item()) < 1e-9
    print(f'  [{"PASS" if ok2 else "FAIL"}] 加载工况零差异')

    # ── (3) 空转 2xBPF 收拢 ──
    section('(3) 空转 2xBPF (a=1): 上界收拢到 0.20')
    p_idle_full = p_idle
    a_idle_full = absence_gate(p_idle_full, REG)
    for w3 in (0.973, 0.10):
        w_t = w_tensor([0.048, 0.980, w3])
        lr_old = w_range_loss(w_t, w_low, w_high, presence=p_idle_full,
                              w_low_min=0.05)
        lr_new = w_range_loss(w_t, w_low, w_high, presence=p_idle_full,
                              absence=a_idle_full, w_low_min=0.05,
                              w_high_min=0.20)
        lb_new = balance_loss(w_t, w_min=w_low, w_max=w_high,
                              presence=p_idle_full, absence=a_idle_full,
                              w_min_floor=0.05, w_max_floor=0.20)
        print(f'  w_3={w3:.3f}:  L_range 旧={lr_old.item():.4f} '
              f'新={lr_new.item():.4f}   L_balance 新={lb_new.item():.4f}')
    ok3 = True
    w_hi = w_tensor([0.048, 0.980, 0.973])
    lr_hi = w_range_loss(w_hi, w_low, w_high, presence=p_idle_full,
                         absence=a_idle_full, w_low_min=0.05, w_high_min=0.20)
    w_lo = w_tensor([0.048, 0.980, 0.100])
    lr_lo = w_range_loss(w_lo, w_low, w_high, presence=p_idle_full,
                         absence=a_idle_full, w_low_min=0.05, w_high_min=0.20)
    ok3 &= lr_hi.item() > 0.10 and abs(lr_lo.item()) < 0.30
    print(f'  [{"PASS" if lr_hi.item() > 0.10 else "FAIL"}] '
          f'w_3=0.973 被惩罚 (L={lr_hi.item():.4f})')
    print(f'  [{"PASS" if lr_lo.item() < 0.30 else "FAIL"}] '
          f'w_3=0.100 基本不罚 (L={lr_lo.item():.4f})')

    # ── (4) RE_2D: 空转时 2xBPF 不产生推力 ──
    # 判据: 只改动 2xBPF 带内的 tfr, 若 a_2x=1 则该带被排除, 损失**不变**;
    #       若 a_2x=0 则该带参与, 损失**改变**。
    section('(4) RE_2D: a_2x=1 时 2xBPF 带被排除 (改动该带不影响损失)')
    Fb = 1000
    freqs = torch.linspace(0.5, 500.0, Fb)
    tfr = torch.rand(B, Fb, T) + 0.1
    in_2x = (freqs >= 90.0) & (freqs < 105.0)
    tfr_b = tfr.clone()
    tfr_b[:, in_2x, :] = (torch.rand(B, int(in_2x.sum()), T) + 0.1) * 6.0

    a_zero = torch.zeros_like(a_idle_full)
    d_zero = (renyi_2d_loss(tfr_b, freqs, REG, absence=a_zero)
              - renyi_2d_loss(tfr, freqs, REG, absence=a_zero)).item()
    d_idle = (renyi_2d_loss(tfr_b, freqs, REG, absence=a_idle_full)
              - renyi_2d_loss(tfr, freqs, REG, absence=a_idle_full)).item()
    print(f'  a_2x=0 (无门控) 时改 2xBPF 带: ΔL = {d_zero:+.6f}')
    print(f'  a_2x=1 (空转)   时改 2xBPF 带: ΔL = {d_idle:+.6f}')
    ok4 = abs(d_zero) > 1e-3 and abs(d_idle) < 1e-9
    print(f'  [{"PASS" if abs(d_zero) > 1e-3 else "FAIL"}] 无门控时该带参与')
    print(f'  [{"PASS" if abs(d_idle) < 1e-9 else "FAIL"}] 门控时该带被完全排除')

    print('\n' + '=' * 72)
    allok = ok1 and ok2 and ok3 and ok4
    print(f'总判定: {"全部通过" if allok else "存在失败项"}')
    print('=' * 72)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())