"""
重排 M 轮的真实成本基准 (B=10, F=1000, T=2000, 真实训练形状)
=============================================================
动机: 之前引用的 0.349 ms/call 是在 [1, 501, 256] (12.8 万元素) 上测的
(docs/nsys_interview_qa.md), 而真实形状是 2e7 元素 —— **大 156 倍**。
且 use_opt 路径每次调用额外做 5 次 transpose(1,2).contiguous(), 其中
omega_hat_int 是 int64 / 153 MB。所以 M 轮的真实代价必须实测, 不能外推。

对比两种 Python 侧编排 (都含 backward, 才可比):
  A. naive   —— 生产现状: 每轮各自调 ReassignerFunction, 每次内部 5 次 transpose
  B. hoisted —— sigma / omega 常驻 [B,T,F], 整条链只在进出各转一次。
                **纯 Python 侧改动, 不碰 kernel 源码**, 只需一个转置版
                autograd Function (见 HoistedReassigner)。

判据: 若 M=3 相对 M=1 的增量占真实步时 (>5%, 约 0.6 s/step) 不可忽略,
      则应先做 hoist 优化再上 M 轮。

用法:
  python scripts/test/bench_reassigner_mrounds.py --batch 10 --f 1000 --t 2000
"""
import argparse
import sys
import time
from pathlib import Path

import torch

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from models.sast import ReassignerFunction                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

_HAS_OPT = False
try:
    from deploy import reassigner_opt as _opt
    _HAS_OPT = True
except Exception as e:                                          # pragma: no cover
    print(f'[warn] reassigner_opt 不可用: {e}')


class HoistedReassigner(torch.autograd.Function):
    """[B,T,F] 布局的重排 (与 ReassignerFunction 数值同源, 但省掉每轮 5 次
    transpose)。前向/反向直接调 opt kernel, 不经过 [B,F,T] 布局。"""

    @staticmethod
    def forward(ctx, tfr_bTf, sigma_bTf, omega_bTf, K):
        B, T, F = tfr_bTf.shape
        out = torch.zeros(B, T, F, device=tfr_bTf.device, dtype=torch.float32)
        _opt.forward(tfr_bTf, sigma_bTf, omega_bTf, out, K)
        ctx.save_for_backward(sigma_bTf, tfr_bTf, omega_bTf)
        ctx.K = K
        return out

    @staticmethod
    def backward(ctx, grad_out):
        sigma, tfr, omega = ctx.saved_tensors
        B, T, F = sigma.shape
        gs = torch.zeros(B, T, F, device=sigma.device, dtype=torch.float32)
        _opt.backward(grad_out.contiguous(), sigma, tfr, omega, gs, ctx.K)
        return None, gs, None, None


def make_inputs(B, F, T, dev):
    torch.manual_seed(0)
    tfr = torch.rand(B, F, T, device=dev).requires_grad_(True)
    # sigma 必须 requires_grad: 它是 w_i -> sigma_i 的唯一梯度入口
    sigma = (torch.rand(B, F, T, device=dev) * 2.0 + 0.5).requires_grad_(True)
    omega = torch.randint(0, F, (B, F, T), device=dev).long()
    return tfr, sigma, omega


def run_naive(tfr0, sigma, omega, M, K, F):
    """A: 生产现状的编排 (每轮各自 transpose)。"""
    S = tfr0
    for _ in range(M):
        S = ReassignerFunction.apply(S, sigma, omega, K, F)
    S.sum().backward()
    return S


def run_hoisted(tfr0, sigma, omega, M, K, F):
    """B: 常驻 [B,T,F]。"""
    sg = sigma.transpose(1, 2).contiguous()
    oh = omega.transpose(1, 2).contiguous()
    S = tfr0.transpose(1, 2).contiguous()
    for _ in range(M):
        S = HoistedReassigner.apply(S, sg, oh, K)
    S.sum().backward()
    return S


def bench(fn, *a, reps=2):
    best_t, peak = float('inf'), 0.0
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn(*a)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best_t = min(best_t, time.perf_counter() - t0)
        peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 20
                   if torch.cuda.is_available() else 0.0)
    return best_t * 1000.0, peak


def main():
    ap = argparse.ArgumentParser(description='重排 M 轮真实成本基准')
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--f', type=int, default=1000)
    ap.add_argument('--t', type=int, default=2000)
    ap.add_argument('--reps', type=int, default=2)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, F, T = args.batch, args.f, args.t
    K = 45          # ceil(kernel_radius(3.0) * sigma_max(15.0)), 与生产一致

    print(f'设备 {dev};  形状 B={B}, F={F}, T={T}  '
          f'({B * F * T / 1e6:.1f}e6 元素);  K={K}')
    print(f'reassigner_opt 可用: {_HAS_OPT};  重复 {args.reps} 次取最优\n')

    mb = B * F * T * 4 / 2 ** 20
    print(f'单张 [B,F,T] float32 = {mb:.0f} MB;  omega_hat_int(int64) = {2 * mb:.0f} MB')
    print(f'naive 每轮额外 transpose 流量 ≈ '
          f'{(mb * 2 + mb * 2 + 2 * mb * 2 + mb + mb * 2) / 1024:.2f} GB\n')

    print(f'{"M":>2s} {"A naive(ms)":>12s} {"B hoist(ms)":>12s} '
          f'{"A 峰值(MB)":>11s} {"B 峰值(MB)":>11s} {"省时":>7s} {"省显存":>8s}')
    base_a = base_b = None
    for M in (1, 2, 3, 4):
        tfr, sigma, omega = make_inputs(B, F, T, dev)
        ta, pa = bench(run_naive, tfr, sigma, omega, M, K, F, reps=args.reps)
        if _HAS_OPT and dev == 'cuda':
            tfr, sigma, omega = make_inputs(B, F, T, dev)
            tb, pb = bench(run_hoisted, tfr, sigma, omega, M, K, F,
                           reps=args.reps)
            st, sm = f'{(1 - tb / ta):.0%}', f'{(1 - pb / pa):.0%}'
        else:
            tb, pb, st, sm = float('nan'), float('nan'), '—', '—'
        if M == 1:
            base_a, base_b = ta, tb
        print(f'{M:>2d} {ta:>12.1f} {tb:>12.1f} {pa:>11.0f} {pb:>11.0f} '
              f'{st:>7s} {sm:>8s}')

    print(f'\n相对 M=1 的增量:')
    for M in (2, 3, 4):
        tfr, sigma, omega = make_inputs(B, F, T, dev)
        ta, _ = bench(run_naive, tfr, sigma, omega, M, K, F, reps=args.reps)
        print(f'  M={M}: naive +{ta - base_a:7.1f} ms '
              f'({(ta - base_a) / base_a:+.0%} vs M=1)')
    if _HAS_OPT and dev == 'cuda':
        print(f'  (hoisted 基准 M=1 = {base_b:.1f} ms)')

    print('\n参照: 真实训练一步 (B=10, T=2000, P×K=2×5) 实测约 0.6 s/step。')
    print('判据: M=3 的增量若 > 30 ms (≈5% 步时), 先做 hoist 再上 M 轮。')


if __name__ == '__main__':
    main()
