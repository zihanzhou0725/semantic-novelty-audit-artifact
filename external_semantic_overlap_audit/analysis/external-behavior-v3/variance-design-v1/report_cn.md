# CLUTRR V3 探索性方差分解与 V4 设计门控

## 证据状态

- 本分析在 V3 解盲后冻结，只能用于探索性诊断和未来设计。
- V3 的预注册结论保持 `imprecise_no_go`，未作追溯修改。

## 方差分解

- Bootstrap-Shapley 方差份额：seed 4.90%，pair 94.40%，source group 0.70%。
- 该分解是抽样机制归因，不等同于因果方差组件。

## V4 严格精度门

- 未参与 V3 的同标签语义 pair 容量：139。
- 即使 seed 数趋于无穷，达到 2 pp 区间宽度的投影最少 pair 数：634。
- 使用全部剩余 pairs 且 seed 数趋于无穷时，投影区间宽度：4.2690 pp。
- 设计门决策：`strict_v3_precision_replication_no_go`。
- 建议：`do_not_release_same_gate_v4_training`。

结论仅适用于复现 V3 的 2 pp 严格 precision gate。采用不同 estimand 或方向性门控必须建立新的前瞻性协议，不能由本探索性结果直接放行。
