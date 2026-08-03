"""
编译 MSST CUDA 扩展 (STFT + 硬挤压 + 线性挤压)
================================================

输出三个独立 CUDA 扩展:
  - msst_stft:           modulated-form STFT (精确匹配 numpy msst() Step 1, cuFFT)
  - msst_squeeze_hard:   硬最近邻挤压 (匹配 numpy MSST)
  - msst_squeeze_linear: 线性插值挤压 (连续 IF, 双 bin)

用法:
  # 本地 GPU (自动检测架构)
  cd deploy && python setup_msst_kernels.py build_ext --inplace

  # 指定 CUDA 架构
  TORCH_CUDA_ARCH_LIST="8.9" python deploy/setup_msst_kernels.py build_ext --inplace

  # 手动指定 MSVC 工具链版本 (VS 2026 用户需要)
  $env:VCToolsVersion="14.44.35207"
  python deploy/setup_msst_kernels.py build_ext --inplace

环境要求:
  - PyTorch >= 2.1, CUDA 12.x (若系统 nvcc 为 12.x，torch 必须为 cu12x 构建)
  - Windows: Visual Studio 2022 (MSVC 14.44) — VS 2026 用户见下文
  - Linux: GCC >= 9

VS 2026 兼容性说明:
  若系统安装了 VS 2026 (MSVC 14.51)，其 STL 的 static_assert(false) 与当前 PyTorch
  头文件不兼容。本脚本会自动设置 VCToolsVersion=14.44.35207 以使用 VS 2022 工具链。
  如自动检测失败，请手动设置 VCToolsVersion 环境变量。

输出: deploy/*.pyd (Windows) 或 *.so (Linux)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
import subprocess


def _pick_msvc_toolchain():
    """确保使用 MSVC 14.44 (VS 2022) 而非 14.51+ (VS 2026)，后者 STL 不兼容。

    通过设置 VCToolsVersion 环境变量让 vcvarsall.bat 选择正确工具链。
    """
    if sys.platform != 'win32':
        return

    # 用户已显式指定，尊重用户选择
    if os.environ.get('VCToolsVersion', ''):
        print(f"[setup_msst] VCToolsVersion={os.environ['VCToolsVersion']} (user-specified)")
        return

    # 查找 VS 安装目录
    vs_dir = os.environ.get('VSINSTALLDIR', '')
    if not vs_dir:
        program_files = os.environ.get('ProgramFiles(x86)', os.environ.get('ProgramFiles', ''))
        if program_files:
            vswhere = os.path.join(program_files, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
            if os.path.isfile(vswhere):
                try:
                    vs_dir = subprocess.check_output(
                        [vswhere, '-latest', '-property', 'installationPath'],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()
                except Exception:
                    pass

    if not vs_dir or not os.path.isdir(vs_dir):
        return

    # 枚举可用的 MSVC 工具链版本
    msvc_root = os.path.join(vs_dir, 'VC', 'Tools', 'MSVC')
    if not os.path.isdir(msvc_root):
        return

    versions = []
    for entry in os.listdir(msvc_root):
        p = os.path.join(msvc_root, entry)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'bin', 'Hostx64', 'x64', 'cl.exe')):
            versions.append(entry)

    if not versions:
        return

    # 按版本号排序，取最新的
    versions.sort(key=lambda v: [int(x) for x in v.split('.')])
    latest = versions[-1]

    # 如果最新工具链是 14.5x (VS 2026+)，回退到 14.44 (VS 2022)
    if int(latest.split('.')[0]) >= 14 and int(latest.split('.')[1]) >= 50:
        fallback = None
        for v in sorted(versions, reverse=True):
            parts = v.split('.')
            if int(parts[0]) == 14 and int(parts[1]) < 50 and int(parts[1]) >= 40:
                fallback = v
                break

        if fallback:
            os.environ['VCToolsVersion'] = fallback
            print(f"[setup_msst] VS 2026 detected (MSVC {latest}), auto-switching to MSVC {fallback}")
        else:
            print(f"[setup_msst] WARNING: MSVC {latest} detected but no VS 2022 fallback found. Build may fail.")
    else:
        print(f"[setup_msst] MSVC {latest} detected (compatible)")


if __name__ == '__main__':
    _pick_msvc_toolchain()

    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if arch_list:
        print(f"[setup_msst] Targeting CUDA arch: {arch_list}")
    else:
        print("[setup_msst] Auto-detecting CUDA arch (PyTorch default)")

    build_dir = os.path.dirname(os.path.abspath(__file__))

    # 公共 nvcc 编译参数
    _NVCC_COMMON = [
        '-O3',
        '-allow-unsupported-compiler',
        '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
        '-D_DISABLE_STL_NOEXCEPT_FUNCTION_TYPE',
        '-Xcompiler', '/Zc:noexceptTypes-',
    ]
    _NVCC_FAST_MATH = _NVCC_COMMON + ['--use_fast_math']

    _CXX_COMMON = ['-O3']

    setup(
        name='msst_cuda_kernels',
        ext_modules=[
            CUDAExtension(
                name='msst_squeeze_hard',
                sources=[os.path.join(build_dir, 'msst_squeeze_hard.cu')],
                extra_compile_args={
                    'cxx': _CXX_COMMON,
                    'nvcc': _NVCC_FAST_MATH,
                },
            ),
            CUDAExtension(
                name='msst_squeeze_linear',
                sources=[os.path.join(build_dir, 'msst_squeeze_linear.cu')],
                extra_compile_args={
                    'cxx': _CXX_COMMON,
                    'nvcc': _NVCC_FAST_MATH,
                },
            ),
            CUDAExtension(
                name='msst_stft',
                sources=[os.path.join(build_dir, 'msst_stft.cu')],
                libraries=['cufft'],
                extra_compile_args={
                    'cxx': _CXX_COMMON,
                    'nvcc': _NVCC_COMMON,  # cuFFT 不需要 --use_fast_math
                },
            ),
            CUDAExtension(
                name='reassigner',
                sources=[os.path.join(build_dir, 'reassigner.cu')],
                extra_compile_args={
                    'cxx': _CXX_COMMON,
                    'nvcc': _NVCC_COMMON,  # 不用 fast_math (保 exp 精度, backward 敏感)
                },
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension,
        },
    )
    print("[setup_msst] Done. Kernels compiled to deploy/*.pyd")
