"""
modelopt PTQ 量化 + QDQ ONNX 导出 (SAST 网络部分)
=====================================================

WHY: SAST 完整 forward 含 91 次循环的高斯重排 + 自定义 CUDA kernel.
     若整体导出 ONNX, Python 循环被 unroll 成 7000+ 节点巨图 → OOM.
     因此**只量化/导出网络部分** (MSST→节点提取→PPM→GAT→sigma/ridge),
     重排保留 CUDA kernel (deploy/reassigner.cu), 在 Jetson 上单次 launch.

导出内容 (forward_network):
    signal [1, T] → tfr_mag, sigma, omega_hat_int, ridge_factor, n_sqz_per_bin
    (5 个重排 kernel 所需中间量, 供 deploy/reassigner.cu 消费)

用法 (需 modelopt_env 环境, 有 GPU):
    python quantize_sast_modelopt.py --checkpoint ../sast_checkpoints/sast_v3_e050.pt \
        --data ../5_dataset.npz --output sast_net_qdq.onnx --samples 100

输出:
    sast_net_qdq.onnx         QDQ 量化 ONNX (网络部分)
    精度报告 (FP32 vs INT8 中间量对比)

下一步 (Jetson 上):
    python build_trt_sast.py --onnx sast_net_qdq.onnx --output sast_net.engine
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_PROJ_ROOT, _DEPLOY_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from sast_export_model import ExportableSAST


class SastNetWrapper(torch.nn.Module):
    """网络部分包装: forward_network → 5 个重排输入 (标量输出便于量化度量)."""

    def __init__(self, exportable: ExportableSAST):
        super().__init__()
        self.exportable = exportable

    def forward(self, x: torch.Tensor):
        d = self.exportable.forward_network(x)
        return (d['tfr_mag'], d['sigma'], d['omega_hat_int'],
                d['ridge_factor'], d['n_sqz_per_bin'])


def load_dataset(data_path: str, max_len: int = 2000,
                 max_samples: int = None):
    """加载 5_dataset.npz 前几个样本作为校准数据."""
    data = np.load(data_path, allow_pickle=True)
    X = data['train_X']
    if X.ndim == 3:
        X = X[:, :, 0]
    X = X[:, :max_len].astype(np.float32)
    if max_samples:
        X = X[:max_samples]
    return X


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main():
    p = argparse.ArgumentParser(description='modelopt PTQ 量化 SAST 网络部分')
    p.add_argument('--checkpoint', default='sast_checkpoints/sast_v3_e050.pt',
                   help='SAST checkpoint (.pt)')
    p.add_argument('--data', default='5_dataset.npz')
    p.add_argument('--output', default='sast_net_qdq.onnx', help='QDQ ONNX 输出')
    p.add_argument('--samples', type=int, default=64, help='校准样本数')
    p.add_argument('--algorithm', default='max',
                   choices=['max', 'mse', 'smoothquant'],
                   help='校准算法: max=极值, mse=最小化误差, smoothquant=平滑激活离群值')
    p.add_argument('--opset', type=int, default=18)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"})')

    # ── 加载模型 + 组装导出包装 ──
    print(f'\nLoading SAST checkpoint: {args.checkpoint}')
    exportable = ExportableSAST.from_checkpoint(args.checkpoint, device)
    wrapper = SastNetWrapper(exportable).eval().to(device)

    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f'Network part params: {n_params:,}')

    # ── 校准数据 ──
    print(f'Loading calibration data: {args.data}')
    calib_X = load_dataset(args.data, max_samples=args.samples)
    print(f'  calib: {calib_X.shape} (samples x T)')

    # ── FP32 基线: 用前 N 个样本的中间量做参考 ──
    print('\n[基线] FP32 网络部分前向...')
    ref_outs = []
    with torch.no_grad():
        for i in range(min(16, len(calib_X))):
            x = torch.from_numpy(calib_X[i:i + 1]).to(device)
            outs = wrapper(x)
            ref_outs.append([o.detach().clone() for o in outs])

    # ── modelopt PTQ ──
    from modelopt.torch.quantization import quantize, INT8_DEFAULT_CFG

    def forward_loop(model_q):
        for i in range(len(calib_X)):
            x = torch.from_numpy(calib_X[i:i + 1]).to(device)
            model_q(x)

    quant_cfg = {**INT8_DEFAULT_CFG, 'algorithm': args.algorithm}
    print(f'\nmodelopt PTQ 量化 (calib={args.samples}, algorithm={args.algorithm})...')
    t0 = time.time()
    model_q = quantize(wrapper, quant_cfg, forward_loop=forward_loop)
    print(f'  量化完成 ({time.time()-t0:.1f}s)')

    # ── INT8 精度对比 (中间量 max abs diff) ──
    print('\n[验证] INT8 量化后中间量误差 (vs FP32):')
    names = ['tfr_mag', 'sigma', 'omega_hat_int', 'ridge_factor', 'n_sqz_per_bin']
    model_q.eval()
    with torch.no_grad():
        for i in range(min(8, len(calib_X))):
            x = torch.from_numpy(calib_X[i:i + 1]).to(device)
            outs_q = model_q(x)
            print(f'  sample {i}:')
            for j, (name, oq) in enumerate(zip(names, outs_q)):
                err = _max_abs_diff(ref_outs[i][j], oq)
                print(f'    {name:<16s} maxdiff={err:.6f}')

    # ── 导出 QDQ ONNX ──
    print(f'\n导出 QDQ ONNX: {args.output}')
    dummy = torch.from_numpy(calib_X[0:1]).to(device)
    torch.onnx.export(
        model_q, dummy, args.output,
        input_names=['signal'],
        output_names=names,
        opset_version=args.opset,
        do_constant_folding=False,   # 保留 QDQ 节点
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'完成: {args.output} ({size_mb:.1f} MB)')

    import onnx
    m = onnx.load(args.output)
    ops = sorted(set(n.op_type for n in m.graph.node))
    qdq_ops = [o for o in ops if 'Quantize' in o or 'Dequantize' in o]
    print(f'  ONNX ops: {len(ops)} 种, 含 QDQ: {len(qdq_ops)} 种 {qdq_ops}')
    print(f'\n下一步 (Jetson): python build_trt_sast.py --onnx {args.output} --output sast_net.engine')


if __name__ == '__main__':
    main()
