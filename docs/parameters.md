# TFDCL 参数与数据参考

> CLI 参数、构造函数超参、数据加载器、下游评估的完整参考。

---

## CLI 超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--repr-dims` | 320 | 表征维度 (output_dims) |
| `--batch-size` | 8 | 训练批大小 |
| `--lr` | 0.001 | 学习率 |
| `--epochs` / `--iters` | None | 训练 epoch/iter 数，二选一 |
| `--seed` | None | 随机种子 |
| `--gpu` | 0 | GPU 设备号 |
| `--eval` | flag | 训练后执行下游评估 |
| `--eval-every` | None | 每 N epoch 评测 |
| `--save-every` | None | 每 N epoch/iter 保存 checkpoint |
| `--max-threads` | None | PyTorch 最大线程数 |
| `--irregular` | 0 | 数据随机失活比例 |
| `--num-subbands` | 4 | 子带数 K |
| `--comple-threshold` | 0.0 | 置信度阈值，推荐 0.05~0.2 |
| `--feb-modes` | 64 | FEB 傅里叶模态数 |
| `--topk-modes` | 128 | Top-K 频段选择个数 |
| `--max-freq-bins` | 2000 | 频域位置编码最大 bin 数 |
| `--compressed-len` | 256 | 时间维度压缩目标长度 |
| `--no-topk-freq` | flag | 禁用 Top-K，回退 O(F²) 注意力 |
| `--no-gradient-checkpoint` | flag | 禁用梯度检查点 |
| `--dense-routing` | flag | 使用 Dense 路由 (Joint Softmax) |

---

## 构造函数超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_branches` | 4 | 解耦分量数 B |
| `hidden_dims` | 64 | 隐藏层维度 |
| `depth` | 3 | 专家网络深度 |
| `lambda_bw` | 1.0 | VMD 基础带宽最小化权重 |
| `lambda_rep` | 0.5 | VMD 中心频率排斥权重 |
| `lambda_peak` | 1.0 | 峰值引导自适应带宽强度 |
| `lambda_overlap` | 0.3 | 相邻子带重叠惩罚权重 |
| `lambda_boundary` | 0.1 | μ 边界软约束权重 |
| `lambda_load_balance` | 0.01 | 专家负载均衡损失权重 |
| `lambda_diversity` | 0.1 | 分量路由多样性损失权重 |

---

## 数据加载器

所有加载器返回 `[N, T, C]` 格式，单变量 C=1。

| CLI `--loader` | 函数 | 说明 |
|------|--------|------|
| `UCR` | `load_UCR()` | UCR 时间序列分类档案库 |
| `UCR_entropy` | `load_UCR_with_entropy()` | UCR + 预计算物理熵曲线 |
| `UEA` | `load_UEA()` | UEA 多变量时间序列 |
| `my_data` | `load_mydata()` | 6 类自定义 .npz |
| `my_data2` | `load_mydata2()` | 3 类自定义 .npz |
| `my_data3` | `load_mydata3()` | 5 类自定义 .npz |
| `ptbxl_pretrain` | `load_ptbxl_unsupervised()` | PTB-XL 心电图预训练 |
| `forecast_csv` | `load_forecast_csv()` | CSV 预测数据集 |
| `forecast_npy` | `load_forecast_npy()` | NPY 预测数据集 |
| `anomaly` | `load_anomaly()` | 异常检测数据集 |
| `anomaly_coldstart` | `load_anomaly()` + FordA | 冷启动异常检测 |

---

## 下游评估

| 任务 | 评估函数 | 分类器 | 指标 |
|------|---------|--------|------|
| classification | `eval_classification3()` | SVM (RBF, C=100) | acc, auprc, f1, precision, recall, confusion_matrix |
| forecasting | `eval_forecasting()` | Ridge 回归 | — |
| anomaly | `eval_anomaly_detection()` | 阈值法 | — |

分类流程：`DCMR.encode(concat_freq=True)` → `[N, B_br*D + D]` → `StandardScaler` + `SVC`。

---

## Auto K-Means++ 聚类

- KMeans with `init='k-means++'` + silhouette_score 自动选最优 k
- 候选 k 集：`[k_min, k_base//2, k_base, k_base*2, k_max]`，k_base = clamp(sqrt(N), 16, 256)
- 子采样 (max 3000) 评估 silhouette，选最高分的 k
- 时域和频域各自独立聚类，每 20 epoch 重新聚类
