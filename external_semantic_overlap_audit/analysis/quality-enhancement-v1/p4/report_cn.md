# P4 Reproducible Audit Toolkit 验证报告

## 结论

- **P4 Tool-Contribution Go**：冻结 Gate 的五项要求全部通过。
- 该结论只确认工具方法与复现性，不改变 P3 Breadth No-Go。

## 公共 benchmark adapters

- `tensor_logic_fb15k237_v1`：知识图谱 direct triple、endpoint-conditioned path 与 relation schema。
- `occam_markov_transition_v1`：Markov order、transition tensor semantic ID 与 pigeonhole exposure bound。

两种 adapter 使用不同语义表示，但输出同一 claim-specific schema；不会把 direct identity、sound alias 与 unresolved 合并成单一布尔值。

## Gate 验证

- Common schema：2/2 audit outputs 通过校验。
- Unit canaries：`exact_function`、`sound_alias`、`unresolved` 全部通过。
- Clean environment：使用 `python3 -I`、空 `PYTHONPATH`、零第三方运行时依赖完成两次独立复现。
- Determinism：adapter outputs、manifest 和 checksum bytes 两次完全一致。
- Frozen counts：Tensor Logic `484/1000` endpoint lower bound、`999/1000` schema lower bound；Occam 两个实验均为 `2-4/20`。

## 允许主张

- 工具包提供可迁移的 claim-specific evidence schema、两种公共 benchmark adapter 和可复现的 lower-bound audit。
- `unresolved` 表示未识别，不表示 semantic novelty。

禁止据此声称 P3 已通过、overlap 导致了行为变化，或未发现 witness 等价于语义新颖。
