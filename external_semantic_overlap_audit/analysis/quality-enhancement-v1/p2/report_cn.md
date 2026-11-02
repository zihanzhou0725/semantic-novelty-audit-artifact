# P2 SLR 调整关联识别前置检查

协议：`quality-enhancement-p2-identification-v1`

## 决策

- **Identification No-Go before outcome fit**
- Witnessed 保留：352/452 = 77.88%
- No-witness 保留：498/548 = 90.88%
- 注册门槛：每组至少保留 80%，且每组至少 100 行。

Witnessed 组在 curriculum exact-level precheck 已低于 80%，因此不得放宽 strata、改变 covariate、拟合 propensity/outcome GLM，也不得执行 10,000 次 bootstrap。

## Prediction source

- 合格 confirmatory source 数：0
- 当前 repository-hosted output 与 1,000 个测试 ID/rule 对齐，但未绑定 checkpoint revision/完整 decoding config，且未复现 ACL 2026 Table 2，因此仅可作 exploratory source。

## 可报告结论

- SLR-Bench witnessed semantic-overlap 下界仍为 452/1000（45.2%）。
- P2 没有 adjusted association 点估计、置信区间或方向性结论。
- `no registered witness` 不等同于 semantic novelty。
