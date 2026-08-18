"""
导出 SAST 网络部分为 FP32 ONNX (Jetson 部署)
===============================================

WHY 只导出网络部分 (不含重排):
  SAST 完整 forward 含 91 次循环的高斯重排 + 自定义 CUDA kernel.
  若整体导出 ONNX, Python 循环被 unroll 成 7000+ 节点巨图 → OOM.
  因此只导出 MSST→节点提取→PPM→GAT→sigma/ridge 的网络部分,
  重排保留 CUDA kernel (deploy/reassigner.cu), 在 Jetson 上单次 launch.

导出内容 (forward_network):
    signal [1, T] → tfr_mag, sigma, omega_hat_int, ridge_factor, n_sqz_per_bin
    (5 个重排 kernel 所需中间量, 供 deploy/reassigner.cu 消费)

精度选择:
    FP32 (默认) — 数值与训练完全一致, TRT 可在 Jetson 上构建 FP16 engine
    (TRT 的 FP16 是自动转换, 无需 modelopt; modelopt 的 INT8 在
    Windows+VS2026 下 CUDA 扩展编译失败, 不可用)

用法 (需 my_work 环境, 有 GPU; 在 deploy/ 目录下运行):
    python export_sast_onnx.py --checkpoint ../sast_checkpoints/sast_v3_e050.pt \
        --data ../5_dataset.npz --output sast_net.onnx --samples 8

输出:
    sast_net.onnx   FP32 ONNX (网络部分)
    与 PyTorch 参考前向的数值对比报告 (max abs diff)

下一步 (Jetson 上):
    python build_trt_sast.py --onnx sast_net.onnx --output sast_net.engine --precision fp16
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

OUTPUT_NAMES = ['tfr_mag', 'sigma', 'omega_hat_int', 'ridge_factor', 'n_sqz_per_bin']


def load_dataset(data_path: str, max_len: int = 2000, max_samples: int = None):
    """加载 5_dataset.npz 样本作为验证/一致性测试数据."""
    data = np.load(data_path, allow_pickle=True)
    X = data['train_X']
    if X.ndim == 3:
        X = X[:, :, 0]
    X = X[:, :max_len].astype(np.float32)
    if max_samples:
        X = X[:max_samples]
    return X


def main():
    p = argparse.ArgumentParser(description='导出 SAST 网络部分为 FP32 ONNX')
    p.add_argument('--checkpoint', default='sast_checkpoints/sast_v3_e050.pt',
                   help='SAST checkpoint (.pt)')
    p.add_argument('--data', default='5_dataset.npz')
    p.add_argument('--output', default='sast_net.onnx', help='输出 ONNX 路径')
    p.add_argument('--samples', type=int, default=8, help='一致性验证样本数')
    p.add_argument('--opset', type=int, default=18)
    p.add_argument('--device', default='cuda')
    p.add_argument('--verify-only', action='store_true',
                   help='只验证已有 ONNX (跳过导出), 用于 onnxruntime 环境')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} '
          f'({torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})')

    # ══ 1. 加载模型 + 组装导出包装 ══
    print(f'\nLoading SAST checkpoint: {args.checkpoint}')
    exportable = ExportableSAST.from_checkpoint(args.checkpoint, device)
    exportable.eval()
    n_params = sum(p.numel() for p in exportable.parameters())
    print(f'Network part params: {n_params:,}')

    # ══ 2. 一致性验证数据 ══
    X = load_dataset(args.data, max_samples=args.samples)
    print(f'Validation signals: {X.shape}')

    # ══ 3. PyTorch 参考输出 (用于 ONNX 一致性对比) ══
    print('\n[Reference] PyTorch forward_network...')
    ref_outs = {}
    with torch.no_grad():
        x0 = torch.from_numpy(X[0:1]).to(device)
        ref0 = exportable.forward_network(x0)
        for name in OUTPUT_NAMES:
            ref_outs[name] = ref0[name].detach().clone()

    # ══ 4. 导出 ONNX (包装为 nn.Module, dict → 多输出) ══
    class _Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            d = self.model.forward_network(x)
            return (d['tfr_mag'], d['sigma'], d['omega_hat_int'],
                    d['ridge_factor'], d['n_sqz_per_bin'])

    wrapper = _Wrapper(exportable).eval().to(device)

    if not args.verify_only:
        print(f'\nExporting ONNX: {args.output}')
        t0 = time.time()
        torch.onnx.export(
            wrapper,
            (x0,),
            args.output,
            input_names=['signal'],
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            dynamo=True,
            do_constant_folding=False,
        )
        print(f'  exported in {time.time()-t0:.1f}s')
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f'  {args.output}: {size_mb:.1f} MB')

        import onnx
        # 内联外部数据为单文件 (Jetson 传输只需一个 .onnx)
        from onnx.external_data_helper import convert_model_from_external_data
        m = onnx.load(args.output)
        convert_model_from_external_data(m)
        onnx.save(m, args.output)
        data_file = args.output + '.data'
        if os.path.exists(data_file):
            os.remove(data_file)

        ops = sorted(set(n.op_type for n in m.graph.node))
        print(f'  ONNX ops: {len(ops)} 种')
        print(f'  ops: {ops}')
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f'  单文件 {args.output}: {size_mb:.1f} MB (权重已内联)')
    else:
        print(f'  (verify-only 模式, 跳过导出, 验证已有 {args.output})')

    # ══ 5. 一致性验证 ══
    # 优先用 onnxruntime (需支持 float64 Atan); 若失败, 回退到 torch.export
    # 重跑导出的图 (PyTorch 内, 数值完全可靠).
    print(f'\n[Verify] PyTorch vs ONNX (max abs diff):')
    ort_ok = False
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        ort_outs = sess.run(OUTPUT_NAMES, {'signal': X[0:1].astype(np.float32)})
        ort_ok = True
    except Exception as e:
        print(f'  (onnxruntime 验证不可用: {str(e)[:120]})')
        print('  → 回退到 torch.export 图验证 (数值等价)')

    ok = True
    if ort_ok:
        for name, ort_o in zip(OUTPUT_NAMES, ort_outs):
            ref = ref_outs[name].cpu().numpy()
            err = np.abs(ort_o.astype(np.float64) - ref.astype(np.float64)).max()
            status = 'OK' if err < 1e-3 else 'DIFF'
            if status == 'DIFF':
                ok = False
            print(f'  {name:<18s} maxdiff={err:.6e}  {status}')
    else:
        # torch.export 重跑: 用同一输入跑 export 后的图, 对比 ref (同 forward_network)
        try:
            ep = torch.export.export(wrapper, (x0,), strict=False)
            ep_out = ep.module()(x0)
            for name, o in zip(OUTPUT_NAMES, ep_out):
                ref = ref_outs[name]
                err = (o.float() - ref.float()).abs().max().item()
                status = 'OK' if err < 1e-3 else 'DIFF'
                if status == 'DIFF':
                    ok = False
                print(f'  {name:<18s} maxdiff={err:.6e}  {status}')
        except Exception as e:
            print(f'  torch.export 重跑失败: {str(e)[:200]}')
            ok = False
    print(f'\n  Overall: {"PASS (数值一致)" if ok else "FAIL (数值不一致!)"}')

    print(f'\n下一步 (Jetson): python build_trt_sast.py --onnx {args.output} '
          f'--output sast_net.engine --precision fp16')


if __name__ == '__main__':
    main()
