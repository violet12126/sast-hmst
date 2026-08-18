"""
TensorRT engine 构建 (SAST 网络部分 ONNX → engine)
====================================================

在 Jetson 上运行 (需 JetPack + TensorRT Python API):
  python3 build_trt_sast.py --onnx sast_net.onnx --output sast_net.engine \
      --precision fp16

输入 ONNX (由 export_sast_onnx.py 生成, FP32):
    signal [1, T] → tfr_mag, sigma, omega_hat_int, ridge_factor, n_sqz_per_bin
    (5 个输出, 供 CUDA 重排 kernel 消费)

支持精度:
  fp32 (默认)  — 数值精确, 但慢
  fp16         — 推荐, TRT 自动转换, Jetson Orin FP16 吞吐 ~2x
  int8         — 需要校准数据 (--calib), TRT 的 PTQ (不经 modelopt)

注意:
  ONNX 含 DFT 算子 (复数 FFT). TensorRT 对 DFT 的支持取决于 JetPack 版本:
  - JetPack 6.x (TRT 8.6+): 部分支持
  - 若 DFT 算子不被 TRT 支持, 构建会报 unsupported layer 错误.
    此时需把 MSST 的 FFT 部分移到 Jetson 端用 cuFFT 实现 (见 docs/CUDA-HMST plan),
    网络部分只保留 GAT 之后 (sigma/ridge 计算).

用法 (Jetson):
  python3 build_trt_sast.py --onnx sast_net.onnx --output sast_net.engine --precision fp16
  python3 build_trt_sast.py --onnx sast_net.onnx --precision fp16 --trtexec  # 用 trtexec
"""
import argparse
import os
import sys
import time


def build_with_trtexec(args) -> int:
    """用 trtexec CLI 构建 (最可靠, 避免 Python TRT API 版本问题)."""
    cmd = [
        'trtexec',
        f'--onnx={args.onnx}',
        f'--saveEngine={args.output}',
    ]
    if args.precision == 'fp16':
        cmd.append('--fp16')
    elif args.precision == 'int8':
        cmd.append('--int8')
        if args.calib:
            cmd.append(f'--calib={args.calib}')
        else:
            print('[WARN] int8 需要校准缓存; 无则用 --calib 提供, 否则 TRT 默认随机')
    if args.workspace:
        cmd.append(f'--memPoolSize=workspace:{args.workspace}')
    if args.verbose:
        cmd.append('--verbose')
    cmd += ['--inputIOFormats=fp32:chw', '--outputIOFormats=fp16:chw']
    print(' '.join(cmd))
    rc = os.system(' '.join(cmd))
    return rc


def build_with_python_api(args) -> int:
    """用 TensorRT Python API 构建."""
    try:
        import tensorrt as trt
    except ImportError:
        print('tensorrt Python API 不可用. 请安装或使用 --trtexec.')
        return 1

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f'Parsing ONNX: {args.onnx}')
    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f'  ONNX parse error: {parser.get_error(i)}')
            return 1

    config = builder.create_builder_config()
    if args.workspace:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace))
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if args.precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        print('  Precision: FP16')
    elif args.precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        if args.calib:
            # 简单 min-max 校准: 从校准数据文件估计每个激活的极值
            # (生产环境建议用 TRT 的 EntropyCalibrator2)
            from trt_calibrator import SastEntropyCalibrator
            calibrator = SastEntropyCalibrator(args.calib, args.batch_size)
            config.int8_calibrator = calibrator
        print('  Precision: INT8')
    else:
        print('  Precision: FP32')

    print('Building engine... (可能几分钟)')
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print('[FAIL] engine 构建失败 (查看上方错误)')
        return 1

    with open(args.output, 'wb') as f:
        f.write(serialized)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'[OK] engine: {args.output} ({size_mb:.1f} MB, {time.time()-t0:.0f}s)')
    return 0


def main():
    p = argparse.ArgumentParser(description='构建 SAST 网络部分 TRT engine')
    p.add_argument('--onnx', required=True, help='输入 ONNX (export_sast_onnx.py 输出)')
    p.add_argument('--output', required=True, help='输出 engine 路径')
    p.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16')
    p.add_argument('--workspace', default=None, help='workspace 大小 (bytes, 如 1073741824)')
    p.add_argument('--calib', default=None, help='int8 校准数据 (npy/npz) 或缓存')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--trtexec', action='store_true',
                   help='用 trtexec CLI 而非 Python API')
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    if args.trtexec:
        rc = build_with_trtexec(args)
    else:
        rc = build_with_python_api(args)
    sys.exit(rc)


if __name__ == '__main__':
    main()
