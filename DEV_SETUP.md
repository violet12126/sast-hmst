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

**首次编译**（或缓存失效后）：

```bash
# 进入 MSVC 环境后:
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

**后续使用**：无需手动编译。`models/sast.py` 中的 `_load_hmst_cuda()` 会自动从 JIT 缓存加载已编译的扩展。

### 性能预期

| 操作 | Python 循环 | CUDA kernel | 加速比 |
|------|-----------|-------------|--------|
| 单次 squeeze | ~150 ms | ~0.55 ms | **~270×** |
| 完整 HMST (N=2, M=2) | ~300 ms | ~57 ms | **~5×** |

> 完整 HMST 的瓶颈在 `torch.stft`（cuFFT），squeeze 已不再是瓶颈。

### Jetson Orin 部署

```bash
# 预编译 (交叉或板端)
TORCH_CUDA_ARCH_LIST="8.7" python deploy/setup_hmst.py
# → deploy/hmst_cuda_ext.so
```

### 常见问题

**Q: 报错 `cl.exe` not found？**
A: 没有进入 MSVC 环境，执行上面的 PowerShell 命令。

**Q: 报错 `unsupported Microsoft Visual Studio version`？**
A: 已通过 `-allow-unsupported-compiler` flag 解决（VS 2019 兼容 CUDA 11.8）。

**Q: 报错 `Ninja is required`？**
A: `pip install ninja`
