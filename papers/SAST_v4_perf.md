# SAST v4 训练性能优化文档

> 记录从 515h/60ep 优化到 5.1h/60ep(68x 提速,接近 ts2vec 4h)的方法、过程与关键技术点。

---

## 1. 概述

SAST v4 初始训练耗时极长(batch=16, max_len=2000: 75s/batch, 515min/epoch, 60ep 515h),完全不可用。经三个阶段优化,降至 0.75s/batch, 5.1min/epoch, 60ep 5.1h。

| 阶段 | ms/batch | epoch | 60ep | 提速 |
|---|---|---|---|---|
| 原始(per-frame + Py reassigner + numpy edge, max_len=2000) | 75432 | 515 min | 515 h | 1x |
| + 向量化 Step5(GAT, max_len=1000) | 6300 | 43 min | 43 h | 12x |
| + C++ reassigner CUDA kernel | 5168 | 35 min | 35 h | 14x |
| **+ torch 向量化 edge_features** | **751** | **5.1 min** | **5.1 h** | **68x** |

> 注:max_len 从 2000 降到 1000 是 8GB GPU 显存限制(reassigner + GAT 的 [B,F,T] 大),贡献约 2x;纯算法优化(向量化 + C++ + torch edge)贡献约 34x。

---

## 2. 初始性能分析(瓶颈定位)

用 `torch.profiler` + 手动分解定位瓶颈(batch=16, max_len=1000, 向量化前):

| 环节 | 耗时 | 占比 | 原因 |
|---|---|---|---|
| **per-frame GAT 循环** | ~61s | 81% | `for t in range(T_msst)` 2000 次 PPM+GAT Python 循环,GPU op launch 开销爆炸 |
| **Step4 edge_feats(numpy)** | ~8.4s | 11% | per-batch `compute_edge_features`(running_pearson/std),CPU 算 + GPU↔CPU 转换 |
| **reassigner(91 循环)** | ~5.4s | 7% | 自定义 autograd 91 循环(3*sigma_max=45),每循环 [B,F,T] mul/exp/scatter |
| MSST(torch) | 0.2s | 0.3% | skip_squeeze,已很快 |

**核心矛盾**:SupCon 要大 batch(≥8),reassigner/per-frame GAT 要小 batch。8GB GPU 上 batch=16 + max_len=2000 直接 OOM(reassigner 91 循环 autograd 中间 ~70GB)。

---

## 3. 优化方法与过程

### 3.1 向量化 Step 5(GAT)— 12x

**问题**:[sast.py](models/sast.py) Step 5 `for t in range(T_msst)` 对 2000 帧逐帧跑 PPM + GAT(Python 循环,每次多个小 GPU op,launch 开销爆炸)。

**方案**:把 T 维合并到 batch(`B*T` 一次 forward)。PPM/GAT 本就是 batched 的(N=4 节点、M=10 边很小),`B*T=32000` batch 内存无忧:

```python
# [B, N, T, 4] -> [B*T, N, 4]
raw_feats_bt = raw_feats.permute(0, 2, 1, 3).reshape(B*T, N, 4)
# PPM + GAT 一次 forward 所有帧
h_enhanced, ... = self.ppm(raw_feats_bt, ...)
w_i_bt, A_ij_bt = self.gat(h_gat_in, edge_feats_bt, ...)
# 重排回 [B, N, T]
w_i = w_i_bt.reshape(B, T, N).permute(0, 2, 1)
```

**效果**:per-frame 2000 循环 -> 1 次 batch forward。75s -> 6.3s(12x)。

**注意**:batch=16 max_len=2000 时 B*T=32000 显存超 8GB(PPM/GAT 中间 + reassigner)。降 max_len=1000(B*T=16000)解决。

### 3.2 C++ reassigner CUDA kernel — +20x(单独)

**问题**:reassigner 的 91 循环(2K+1, K=3*sigma_max=45)自定义 autograd Function,每循环对 [B,F,T] 做 mul/exp/scatter,forward + backward 共 4 pass。Python 循环 + GPU launch 开销大(profiler: `aten::mul` 651ms / 2168 calls)。

**方案**:写 [deploy/reassigner.cu](deploy/reassigner.cu) CUDA kernel,一次 launch 每 thread 一个 (b,f,t),省 Python 91 循环:

```cuda
// forward: 每 thread 一个源 bin, 算 Z + scatter w_k*tfr_w
__global__ void reassigner_forward_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Z = sum_k exp(-0.5*(k/sig)^2)
    // for k: atomicAdd(tfr_enhanced[target], w_k * tfr_w)
}
// backward: grad_sigma = sum_k grad_out[target] * tfr_w * dw_k
```

Python `ReassignerFunction` 调 C++ kernel(float32+cuda),Python 循环作 fallback。用 `torch.utils.cpp_extension.CUDAExtension` 编译。

**关键技术点**:
1. **double 内部累加**:backward 的 `dw_k = w_k*(k²/sig³ - dZ/Z)` 是两个接近大数相减,float32 灾难抵消(grad 差 6 数量级)。C++ backward 内部用 `double` 累加 Z/dZ/dw_k,写回 float。
2. **grad_out 非 contiguous**:autograd 传的 `grad_out` stride 非 row-major,C++ row-major 索引取到 0(非 1)。Python 端加 `grad_out = grad_out.contiguous()` 修复。
3. **gradcheck 验证**:C++ backward vs Python fallback + numpy double 模拟,确认梯度正确(<1e-6)。

**效果**:reassigner 单独 5.4s -> 264ms(20x)。SAST batch 6.3s -> 5.2s(reassigner 不再是主瓶颈)。

### 3.3 torch 向量化 edge_features — +6.9x

**问题**:Step4 `compute_edge_features` 用 numpy per-batch(`for b in range(B)`),`_running_pearson`/`_running_std` 滑动窗口在 CPU 算 + GPU→CPU→GPU 转换。profiler: forward CPU 3.8s(CUDA 只 1.2s),edge_feats 占大头。

**方案**:[sast_graph.py](models/sast_graph.py) 新增 `compute_edge_features_torch`,用 `F.pad`(edge) + `unfold` + 向量化 pearson/std,一次算所有 B(GPU):

```python
def _running_pearson_torch(x, y, window):
    x_pad = F.pad(x, (window, window), mode='replicate')  # 匹配 numpy edge pad
    x_unf = x_pad.unfold(-1, W, 1)  # [..., T, W]
    xm = x_unf - x_unf.mean(dim=-1, keepdim=True)
    ym = y_unf - y_unf.mean(dim=-1, keepdim=True)
    cov = (xm * ym).sum(dim=-1)
    denom = torch.sqrt((xm**2).sum(dim=-1) * (ym**2).sum(dim=-1))
    return (cov / denom.clamp(min=1e-8)).clamp(-1.0, 1.0)
```

SAST.forward Step4 GPU path 直接用 GPU tensor(不转 numpy),CPU path 保留 numpy fallback。

**关键技术点**:
1. **pad x/y 再 unfold**(不是 unfold 后 pad 结果):numpy 先 pad x(edge)再滑窗,边界窗口含 pad 值;torch 必须 pad x/y 后 unfold 才匹配(否则边界 diff 1.7)。
2. **std unbiased=False**:匹配 numpy ddof=0。
3. **数值验证**:torch vs numpy max diff <1e-3。

**效果**:SAST batch 5.2s -> 0.75s(6.9x)。forward CPU 瓶颈消除。

---

## 4. 对比 ts2vec(DCMR)

| | SAST v4(优化后) | ts2vec(DCMR) |
|---|---|---|
| batch | 16 | 32 |
| per-batch | 751 ms | ~1300 ms |
| epoch | 5.1 min | ~4.4 min |
| 60ep | 5.1 h | 4.4 h |
| acc | (待训练) | 0.9995 |

ts2vec 快的原因:直接 `torch.fft.rfft` 一次 + 编码器(全 batched GPU),**没有** SAST 的 MSST(per-sample)+ reassigner(91 循环)+ edge_feats(物理图)。SAST 的物理结构开销(reassigner + edge_feats + MSST)是 ts2vec 没有的,但优化后已接近。

---

## 5. 关键技术点总结(Lessons Learned)

1. **Python 循环 -> 向量化**:per-frame GAT(2000 循环)和 edge_feats(per-batch)用 T/B 合并 batch 向量化,消除 Python launch 开销。
2. **自定义 CUDA kernel**:reassigner 的 91 循环无法向量化([91,B,F,T] OOM),用 C++ kernel 一次 launch。可微(forward+backward 手写,autograd Function 调用)。
3. **autograd 数值精度**:
   - 灾难抵消(两个接近大数相减)用 double 内部累加。
   - autograd 传的 tensor 可能非 contiguous,C++ row-major 索引前必须 `contiguous()`。
4. **profiler 定位**:CUDA time vs CPU time 分离。forward CPU 3.8s(CUDA 1.2s)说明 CPU 瓶颈(numpy edge_feats),不是 GPU。
5. **数值验证**:每步优化后 vs 参考实现(numpy/Python)对比,确保 <1e-3。

---

## 6. 最终配置与训练命令

```bash
# 编译 C++ reassigner(首次)
python deploy/setup_msst_kernels.py build_ext --inplace

# 训练(batch=16, max_len=1000, 8GB GPU)
python train_sast.py --epochs 60 --batch_size 16 --max_len 1000 --device cuda
#   751 ms/batch, 5.1 min/epoch, 60ep 5.1h

# 平衡(少截断信号)
python train_sast.py --epochs 60 --batch_size 16 --max_len 1500 --device cuda
#   ~10 min/epoch, 60ep ~10h
```

**显存**:batch=16 max_len=1000 约 3.8GB(8GB GPU 充裕)。max_len=1500 约 6.5GB。

**观察指标**:`sc`(supcon 下降)、`w_spread`(>0.2 节点分化)、`Val acc`(KNN 上升)。

---

## 7. 仍可优化(后续)

- **extract_gpu per-sample 循环**(~1.8s,28%):`for b in range(B)` 逐样本 msst_torch + node 提取。MSST 本质 per-sample,但可减少 Python/numpy 转换开销。
- **max_len=2000 全信号**:当前 8GB GPU 限 max_len=1000/1500。更大 GPU 或 gradient checkpointing 可全信号。
- **reassigner 自适应 K**:小 sigma(2xBPF)用小 K,大 sigma(LOW_FREQ)用大 K,省循环。
