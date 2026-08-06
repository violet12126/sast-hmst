# SAST v4 训练性能优化文档

> 记录从 515h/60ep 优化到 5.1h/60ep(68x 提速,接近 ts2vec 4h)的方法、过程与关键技术点。

---

## 1. 概述

SAST v4 初始训练耗时极长(batch=16, max_len=2000: 75s/batch, 515min/epoch, 60ep 515h),完全不可用。经三个阶段优化,降至 0.75s/batch, 5.1min/epoch, 60ep 5.1h,接近对比框架 ts2vec(DCMR)的 4h25m。

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

## 3. 三个瓶颈详解

### 3.1 per-frame GAT 循环(81% -> 优化后消除)

#### 这是什么操作

SAST 的 GAT(图注意力网络)在物理图上运行,输出 `w_i`(IF 信任度,决定每节点挤压核宽 σ)和 `A_ij`(边注意力,诊断信号)。物理图 4 节点(OP 虚拟 + LOW_FREQ/BPF/2xBPF)、10 条有向边(HARMONIC/CONDITION/DRIFT/COMPETITION)。

每个时间帧 t 独立跑一次图推理(帧间共享权重,不用 RNN):
1. **PPM**(PhysicsPrototypeMemory):节点特征投影 + 原型交叉注意力 + 边门控(compute_edge_gates,按边类型算 HARMONIC 比值门控/CONDITION 上下文相似度/DRIFT 能量相关/COMPETITION 负相关)
2. **GAT**(EdgeConditionedGAT):2 层边条件图注意力,边特征注入注意力计算,scatter_softmax 分组归一化

#### 原实现(为什么慢)

```python
# sast.py Step 5 (原)
w_i_frames = []
for t in range(T_msst):           # ← 2000 次 Python 循环!
    raw_feats = torch.stack([f_norm[:,:,t], log_E[:,:,t], ...])  # [B, N, 4]
    h_enhanced, C_prior, gate_edge, ... = self.ppm(raw_feats, ...)  # PPM
    h_gat_in = self.ppm.gat_input_proj(...)                        # 注入 C_prior
    w_i_t, A_ij_t = self.gat(h_gat_in, edge_feats_frame, ...)     # GAT 2层
    w_i_frames.append(w_i_t)
w_i = torch.stack(w_i_frames, dim=-1)
```

**性能问题**:
- **2000 次 Python 循环**:每帧调 PPM + GAT,每次 ~30 个小 GPU op(node_proj/cross-attn/gates/GAT 2 层 scatter_softmax)
- **GPU launch 开销爆炸**:每个 GPU op launch ~10μs,2000 帧 × 30 op × 10μs = 600ms 纯 launch 开销(还不算计算)
- **Python 循环本身慢**:无向量化,GIL + 解释器开销
- profiler:占 batch 81%,`aten::mm`/`linear` 调用次数爆炸

#### 优化方案

把 T 维合并到 batch维度(`[B, N, T, ...] -> [B*T, N, ...]`),PPM/GAT 一次 forward 所有帧。PPM/GAT 本就是 batched 的(N=4 节点、M=10 边很小),`B*T=16000` batch 内存无忧:

```python
# sast.py Step 5 (优化后)
BT = B * T_msst
raw_feats_bt = raw_feats.permute(0,2,1,3).reshape(BT, N_phys, 4)  # [B*T, N, 4]
node_if_bt = node_if.permute(0,2,1).reshape(BT, N_phys)
cond_ctx_bt = cond_ctx.reshape(BT, d_cond)
# 一次 forward 所有帧 (PPM + GAT 的 op 都是 batched)
h_enhanced, ... = self.ppm(raw_feats_bt, node_if_bt, ...)
w_i_bt, A_ij_bt = self.gat(h_gat_in, edge_feats_bt, ...)
# 重排回 [B, N, T]
w_i = w_i_bt.reshape(B, T_msst, N_phys).permute(0, 2, 1)
```

op 数从 `2000帧 × 30op` 降到 `30op`(一次 batch),launch 开销消除。

#### 效果

75s -> 6.3s(**12x**)。注意 batch=16 max_len=2000 时 B*T=32000 显存超 8GB,降 max_len=1000(B*T=16000)解决。

---

### 3.2 reassigner 91 循环(7% -> C++ kernel 20x)

#### 这是什么操作

`SparseGaussianReassigner`(可微软高斯重排)是 SAST 的核心:把 STFT 幅值按 IF(瞬时频率)`omega_hat` 用高斯核重排到目标频率位置,生成增强 TFR。

对每个 TF bin (b, f, t):
- 高斯核 `w_k = exp(-0.5*(k/σ)²) / Z`,k 遍历 `-K..K`(K=3*sigma_max=45,共 91 个偏移)
- σ(核宽)由 `w_i` 控制:w_i 高 -> σ 小 -> 窄核(硬挤,集中);w_i 低 -> σ 大 -> 宽核(软挤,保留展宽)
- 把 `w_k * tfr_mag[b,f,t]` 累加到 `tfr_enhanced[b, omega_hat+k, t]`(scatter_add)

需要可微(训练时 `w_i -> σ -> tfr_enhanced -> loss` 反传)。

#### 原实现(为什么慢)

```python
# sast.py ReassignerFunction (自定义 autograd, Python 循环)
@staticmethod
def forward(ctx, tfr_weighted, sigma, omega_hat_int, K, F_dim):
    Z = torch.zeros(B, F, T)
    for k in range(-K, K+1):          # ← 91 次 Python 循环
        Z += torch.exp(-0.5 * (k/sigma)**2)
    tfr_enhanced = torch.zeros(B, F, T)
    for k in range(-K, K+1):          # ← 又 91 次
        w_k = torch.exp(-0.5*(k/sigma)**2) / Z
        tfr_enhanced.scatter_add_(1, omega_hat_int+k, w_k * tfr_weighted)

@staticmethod
def backward(ctx, grad_out):
    # 再 2*91 次循环 (算 dZ/dsigma, dw_k/dsigma, grad_sigma)
```

**性能问题**:
1. **Python 91 循环**:forward 91 + backward 2*91 = 273 次,每循环多个 GPU op(exp/mul/scatter)。profiler: `aten::mul` 651ms / 2168 calls
2. **autograd 中间张量爆炸**:91 循环的中间(ratio/exp/w_k)autograd 全保存(用于反向)。batch=16 时每个 `[B,F,T]=[16,1000,2000]=128MB`,91×4pass×3张量×128MB ≈ 70GB(8GB GPU 直接 OOM)
3. **float32 灾难抵消**:backward 的 `dw_k = w_k*(k²/σ³ - dZ/Z)`,两个接近大数相减,float32 丢精度(grad 差 6 数量级)

#### 优化方案(deploy/reassigner.cu)

写 CUDA kernel,一次 launch 每 thread 处理一个 (b,f,t),省 Python 91 循环:

```cuda
// forward: 每 thread 一个源 bin
__global__ void reassigner_forward_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 算 Z = sum_k exp(-0.5*(k/sig)^2)  (thread 内 91 循环)
    // for k: atomicAdd(tfr_enhanced[target], w_k * tfr_w)
}
// backward: grad_sigma = sum_k grad_out[target] * tfr_w * dw_k
```

Python `ReassignerFunction` 调 C++ kernel(float32+cuda 时),Python 循环作 fallback(CPU/float64)。

**关键技术点**:
1. **double 内部累加**:backward 的 `dw_k = w_k*(k²/σ³ - dZ/Z)` 灾难抵消。C++ backward 内部用 `double` 累加 Z/dZ/dw_k,写回 `float`。否则 float32 grad 差 6 数量级(0.269 vs 4e-7)
2. **grad_out 非 contiguous**:autograd 传的 `grad_out` stride 非 row-major,C++ row-major 索引 `grad_out[b*F*T+target*T+t]` 取到 0(应为 1)。Python 端加 `grad_out = grad_out.contiguous()` 修复
3. **gradcheck 验证**:C++ backward vs Python fallback + numpy double 模拟,确认 <1e-6

#### 效果

reassigner 单独 5.4s -> 264ms(**20x**)。SAST batch 6.3s -> 5.2s(reassigner 不再是主瓶颈)。显存从 70GB 降到 ~1GB(只存 sigma/tfr/omega/Z,不存 91 中间)。

---

### 3.3 Step4 edge_feats numpy(11% -> torch 向量化 6.9x)

#### 这是什么操作

`compute_edge_features` 从节点特征(node_if 瞬时频率、node_energy 能量、node_bw 带宽)计算物理图 10 条边的边特征 `[M, T, 5]`。每条边按类型算不同特征:

| 边类型 | dim0 | dim1 | dim2 | dim3 | dim4 |
|---|---|---|---|---|---|
| HARMONIC | r_obs(频率比值) | σ_r(比值稳定性) | Corr_E(能量相关) | w_type | p_min |
| DRIFT | Corr_E | E_ratio(能量比) | bw_coupling(带宽耦合) | w_type | p_min |
| COMPETITION | -Corr_E(负相关) | E_ratio | E_ratio_stability | w_type | p_min |
| CONDITION | cond_sim(占位) | 0 | 0 | w_type | 0 |

核心是 `running_pearson`(滑动窗口 Pearson 相关)和 `running_std`(滑动窗口标准差),窗口 W=2*window_size+1=11。

#### 原实现(为什么慢)

```python
# sast_graph.py (numpy, per-batch)
def compute_edge_features(node_if, node_energy, ...):  # [N, T] numpy
    for m in HARMONIC/DRIFT/COMPETITION/CONDITION:  # 10 边
        r_obs = f_dst / f_src
        edge_feats[m,:,0] = r_obs
        edge_feats[m,:,1] = _running_std(r_obs, window)   # for t 滑窗
        edge_feats[m,:,2] = _running_pearson(e_src, e_dst, window)  # for t 滑窗

# sast.py Step 4 (调用)
for b in range(B):               # ← 16 次 Python 循环
    ef = compute_edge_features(node_if_np[b], ...)  # numpy CPU
# numpy -> torch GPU 转换
```

**性能问题**:
1. **numpy CPU 计算**:running_pearson/std 在 CPU 算(T=2000 滑窗,for t 循环),GPU 空等
2. **per-batch Python 循环**:B=16 次,每次 10 边 × 多个 running_pearson/std
3. **GPU↔CPU 转换**:node_if/energy/bw 从 GPU 转 CPU(numpy),edge_feats 算完转回 GPU。转换 + CPU 计算
4. profiler:forward CPU 3.8s(CUDA 只 1.2s),edge_feats 占 ~2s(CPU 瓶颈)

#### 优化方案(sast_graph.py compute_edge_features_torch)

torch 向量化,GPU 一次算所有 B:

```python
def _running_pearson_torch(x, y, window):  # x,y [B, T] GPU
    x_pad = F.pad(x, (window, window), mode='replicate')  # 边界 pad (匹配 numpy)
    x_unf = x_pad.unfold(-1, W, 1)          # [B, T, W] 滑窗 (向量化, 无 for t)
    xm = x_unf - x_unf.mean(dim=-1, keepdim=True)
    ym = y_unf - y_unf.mean(dim=-1, keepdim=True)
    cov = (xm * ym).sum(dim=-1)
    denom = torch.sqrt((xm**2).sum(dim=-1) * (ym**2).sum(dim=-1))
    return (cov / denom.clamp(min=1e-8)).clamp(-1.0, 1.0)

def compute_edge_features_torch(node_if, node_energy, ...):  # [B, N, T] GPU
    for m, e in enumerate(edges):  # 10 边 (GPU op)
        edge_feats[:, m, :, 0] = _running_pearson_torch(e_src, e_dst, window)
        ...
    return edge_feats  # [B, M, T, 5] GPU
```

SAST.forward Step4 GPU path 直接用 GPU tensor(不转 numpy),CPU path 保留 numpy fallback。

**关键技术点**:
1. **pad x/y 再 unfold**(不是 unfold 后 pad 结果):numpy 先 pad x(edge)再滑窗,边界窗口含 pad 值;torch 必须 pad x/y 后 unfold 才匹配(否则边界 diff 1.7)
2. **std unbiased=False**:匹配 numpy ddof=0
3. **数值验证**:torch vs numpy max diff <1e-3

#### 效果

SAST batch 5.2s -> 0.75s(**6.9x**)。forward CPU 瓶颈消除(numpy 转换 + CPU 计算全消除)。

---


## 4. 关键技术点总结(Lessons Learned)

1. **Python 循环 -> 向量化**:per-frame GAT(2000 循环)和 edge_feats(per-batch)用 T/B 合并 batch 向量化,消除 Python launch 开销。判断标准:循环内 op 是 batched 的(GAT/PPM/pearson),且合并后 tensor 不 OOM(GAT 的 N=4 小,可以;reassigner 的 [91,B,F,T] 大,不行)
2. **自定义 CUDA kernel**:reassigner 的 91 循环无法向量化([91,B,F,T] OOM),用 C++ kernel 一次 launch。可微(forward+backward 手写,autograd Function 调用)
3. **autograd 数值精度**:
   - 灾难抵消(两个接近大数相减)用 double 内部累加
   - autograd 传的 tensor 可能非 contiguous,C++ row-major 索引前必须 `contiguous()`
4. **profiler 定位**:CUDA time vs CPU time 分离。forward CPU 3.8s(CUDA 1.2s)说明 CPU 瓶颈(numpy edge_feats),不是 GPU
5. **数值验证**:每步优化后 vs 参考实现(numpy/Python)对比,确保 <1e-3

---

## 5. 最终配置与训练命令

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

## 6. 仍可优化(后续)

- **extract_gpu per-sample 循环**(~1.8s,28%):`for b in range(B)` 逐样本 msst_torch + node 提取。MSST 本质 per-sample,但可减少 Python/numpy 转换开销
- **max_len=2000 全信号**:当前 8GB GPU 限 max_len=1000/1500。更大 GPU 或 gradient checkpointing 可全信号
- **reassigner 自适应 K**:小 sigma(2xBPF)用小 K,大 sigma(LOW_FREQ)用大 K,省循环
