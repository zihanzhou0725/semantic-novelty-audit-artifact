# Directional V4 功效与 Block 门控

## 冻结假设

- Pilot-shrunk 设计效应：7.66875 pp。
- Gate：效应点估计至少 1 pp，且双侧 95% CI 下界大于 0。
- Power 与 95% Wilson 下界均须至少 90%。

## 门控结果

- 决策：`directional_v4_design_go`。
- 选择 blocks：2。
- 选择 pairs：40。
- 计划 runs：32。
- 模拟 power：0.9583；Wilson 下界：0.9570。
- 投影 SE：2.0784 pp；95% 半宽：4.0736 pp。

## 约束

- 仅选择 V3 未使用的 deterministic pair prefix。
- 每个 block 保持 V3 的 20-pair、24k-row 和 exposure dose。
- 当前仅冻结设计；训练仍未放行，必须先通过 materialization gate。
