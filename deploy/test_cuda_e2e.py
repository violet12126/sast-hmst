"""Quick end-to-end test of CUDA HMST integration."""
import torch, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.sast import compute_hmst, _load_hmst_cuda, _hmst_squeeze_cuda

# 1. 验证自动加载
ext = _load_hmst_cuda()
print(f"CUDA extension auto-loaded: {ext is not None}")

# 2. 端到端 HMST (CUDA)
x = torch.randn(1, 2000, device="cuda")
t0 = time.perf_counter()
tfr, IF, mag = compute_hmst(x, 1000, n_fft=512, hop_length=128, order=2, M=2)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) * 1000
print(f"compute_hmst (CUDA, N=2, M=2): {elapsed:.1f} ms")
print(f"  TFR shape: {tfr.shape}, max={tfr.max():.2f}")

# 3. CPU fallback 计时对比
x_cpu = torch.randn(1, 2000)
t0 = time.perf_counter()
tfr_cpu, _, _ = compute_hmst(x_cpu, 1000, n_fft=512, hop_length=128, order=2, M=2)
elapsed_cpu = (time.perf_counter() - t0) * 1000
print(f"compute_hmst (CPU fallback): {elapsed_cpu:.0f} ms")
print(f"CUDA speedup: {elapsed_cpu/elapsed:.0f}x")
