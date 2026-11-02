# P1 互补精确暴露实验结果

分析协议：`quality-enhancement-p1-analysis-v1`  
输入：60/60 个冻结 epoch-60 runs；40 个目标函数，每个函数 250 个共同评估 episode。

## 完整性

- 全矩阵验证通过，所有 run 的 checkpoint、预测、元数据与哈希均有效。
- 完整性审计未比较 A/B 效应；分析规范在效应计算前冻结。
- 主区间使用 20,000 次层级 percentile bootstrap；架构不重采样。

## 主结果

效应定义为 `exposed - unexposed`，单位为 percentage points (pp)。

- `tau_macro` = **1.832 pp**
- 95% CI = **[1.470, 2.189] pp**
- Gate = **Positive behavioral Go**

| Architecture | Exposed acc | Unexposed acc | Effect (pp) | 99.1667% simultaneous CI | Model claim |
|---|---:|---:|---:|---:|---|
| `symbolic_deepsets` | 0.5960 | 0.5349 | 6.110 | [4.349, 7.849] | eligible |
| `symbolic_relation_network` | 0.8834 | 0.8734 | 0.996 | [0.229, 1.749] | not eligible |
| `symbolic_set_transformer` | 0.7780 | 0.7676 | 1.042 | [-0.105, 2.247] | not eligible |
| `symbolic_hopfield` | 0.8866 | 0.8800 | 0.658 | [-0.192, 1.447] | not eligible |
| `symbolic_perceiver` | 0.7441 | 0.7312 | 1.284 | [-0.042, 2.525] | not eligible |
| `symbolic_transformer` | 0.8294 | 0.8204 | 0.902 | [-0.137, 1.870] | not eligible |

## 次指标

| Metric | Exposed | Unexposed | Exposed - unexposed |
|---|---:|---:|---:|
| Cross entropy | 0.83191 | 0.91052 | -0.07861 |
| Brier score | 0.31829 | 0.34501 | -0.02673 |

训练阶段未持久化逐 episode 概率，因此预注册的训练校准摘要不可估；该缺失被保留，不以训练 loss 替代校准指标。评估校准摘要已按冻结的 15-bin 定义计算。

## Gate 与 claim 边界

Exact cross-family semantic exposure improves macro accuracy for the six registered architectures at the registered training budget.

该实验识别的是注册架构与预算下的随机化 exact-exposure endpoint effect，不单独识别模型内部的 memorization mechanism。
