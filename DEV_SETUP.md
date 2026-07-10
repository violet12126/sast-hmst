# SAST-HMST 开发环境设置

## 依赖

```bash
conda create -n sast python=3.10 -y && conda activate sast
pip install torch>=2.1.0 numpy scipy matplotlib scikit-learn ssqueezepy ninja
```

## CUDA C++ 编译环境 (Windows)

### 已安装组件

| 组件 | 版本 | 路径 |
|------|------|------|
| CUDA Toolkit | 11.8 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` |
| VS 2019 BuildTools | 16.11 | `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools` |
| PyTorch | 2.5.0+cu118 | pip |

### ⚠️ 重要：每次新终端编译前必须执行

CUDA C++ 扩展编译需要 MSVC 编译器环境。**每次打开新终端**，在运行任何会触发 JIT 编译的 Python 脚本之前，先执行：

```powershell
powershell -NoProfile -Command "
Import-Module 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -VsInstallPath 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools' -SkipAutomaticLocation -DevCmdArguments '-arch=x64'
python your_script.py
"
```

或者先进入 MSVC 环境再启动交互式 Python：

```powershell
# 步骤 1: 进入 MSVC 环境
powershell -NoProfile -Command "
Import-Module 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -VsInstallPath 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools' -SkipAutomaticLocation -DevCmdArguments '-arch=x64'
"

# 步骤 2: 在此 PowerShell 中运行任何脚本
python plot_hmst_tfr.py
python train_sast.py
```

### 编译 CUDA 扩展

** 中文 Windows 须先设 `PYTHONUTF8=1`**：`.cu` 源文件含 UTF-8 中文注释，torch 编译时会用系统默认编码（GBK）读文件做哈希，未开 UTF-8 模式会报 `UnicodeDecodeError: 'gbk' codec can't decode ...`。必须在**启动 Python 之前**设好（`compile_hmst_cuda.py` 里的 `os.environ` 那句太晚，解释器已启动）：

```powershell
$env:PYTHONUTF8 = "1"     # PowerShell
```
```bat
set PYTHONUTF8=1          :: cmd
```

**首次编译**（或缓存失效后）：

```bash
# 进入 MSVC 环境 + 设 PYTHONUTF8=1 后:
python deploy/compile_hmst_cuda.py
```

输出示例：
```
[compile_hmst] Compilation successful!
  Test 1 PASS: shape=torch.Size([1, 128, 20]), device=cuda:0
  Test 2 PASS: max_diff=0.00e+00, mean_diff=0.00e+00
  CUDA squeeze: 0.33 ms/call
  Speedup: ~45x
```

**后续使用（推荐：预编译 .pyd，任意 shell 免环境直接用）**：

编译一次后，把产物放到 `deploy/`，之后 `_load_hmst_cuda()` 的**路径 1** 会直接
`import hmst_cuda_ext`，**绕开 torch JIT 的源码哈希+重编译**，因此**无需 MSVC、无需 `PYTHONUTF8`**，普通终端即可加载：

```bash
# 方式 A: 从 JIT 缓存拷贝 (compile_hmst_cuda.py 编译后)
cp "$LOCALAPPDATA/torch_extensions/torch_extensions/Cache/py310_cu118/hmst_cuda_ext/hmst_cuda_ext.pyd" deploy/

# 方式 B: 用 setuptools 就地编译到 deploy/ (需 MSVC + PYTHONUTF8, 仅一次)
cd deploy && python setup_hmst.py build_ext --inplace && cd ..
```

之后任意 shell 直接：

```bash
python plot_hmst_tfr.py   # 打印 "CUDA squeeze kernel: 已加载 ✓"
```

> ⚠️ 直接调用 `torch.utils.cpp_extension.load()`（路径 2 的 JIT）**不算免环境**：它每次都会重读 `.cu` 做哈希（触发 GBK 问题），且构建配置不一致时会重编译（需 vcvars）。免环境只靠 `deploy/` 里的预编译 `.pyd`。
>
> `.pyd` 是**架构相关**的（本机为 GTX 1650, sm_75）；换 GPU 需重新编译，勿跨机器复用。

### 性能预期

| 操作 | Python 循环 | CUDA kernel | 加速比 |
|------|-----------|-------------|--------|
| 单次 squeeze | ~150 ms | ~0.55 ms | **~270×** |
| 完整 HMST (N=2, M=2) | ~300 ms | ~57 ms | **~5×** |

> 完整 HMST 的瓶颈在 `torch.stft`（cuFFT），squeeze 已不再是瓶颈。

### Jetson Orin 部署

Jetson 是 aarch64 Linux → 编译的是 **`.so`**（不是 Windows 的 `.pyd`）。加载机制与桌面端完全一致：`.so` 落到 `deploy/`，`_load_hmst_cuda()` 的路径 1 会自动 `import`。Linux 默认 UTF-8 locale，**无需 `PYTHONUTF8`，无需 MSVC**（用 gcc/nvcc）。

```bash
# 板端编译 (必须在 deploy/ 目录内跑, 因 sources 是相对路径)
cd deploy
TORCH_CUDA_ARCH_LIST="8.7" python setup_hmst.py     # Orin=8.7; Xavier=7.2; 老 Nano=5.3
cd ..
# → 产出 deploy/hmst_cuda_ext.cpython-3X-aarch64-linux-gnu.so
```

之后任意 shell 直接 `python your_script.py`，路径 1 自动加载该 `.so`。

> ⚠️ `.so` 与 **Jetson 的架构 + 板载 PyTorch 版本**绑定：务必用板子上实际运行推理的那个 PyTorch（NVIDIA L4T wheel）**在板端编译**，不要跨机复用桌面产物。

### 常见问题

**Q: 报错 `cl.exe` not found？**
A: 没有进入 MSVC 环境，执行上面的 PowerShell 命令。

**Q: 报错 `unsupported Microsoft Visual Studio version`？**
A: 已通过 `-allow-unsupported-compiler` flag 解决（VS 2019 兼容 CUDA 11.8）。

**Q: 报错 `Ninja is required`？**
A: `pip install ninja`

**Q: 报错 `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94 ...`？**
A: 中文 Windows 默认 GBK 编码读取含 UTF-8 中文注释的 `.cu` 失败。启动 Python 前设 `PYTHONUTF8=1`（PowerShell: `$env:PYTHONUTF8="1"`，cmd: `set PYTHONUTF8=1`）。注意 `PYTHONIOENCODING=utf-8` 只管 stdout，不解决此问题。
