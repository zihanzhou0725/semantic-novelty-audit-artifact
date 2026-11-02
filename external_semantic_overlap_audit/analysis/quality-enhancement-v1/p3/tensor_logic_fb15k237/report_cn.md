# Tensor Logic FB15k-237 Compositional Split 语义审计

范围：冻结的 Experiment 3B，代码 revision `9b497c27992843bfbfdd99801dab6881cebb465e`，数据 revision `2e440e0f9c687314d5ff67ead68ce985dc446e3a`。

## 结论

- **Protocol defect detected**：direct-edge withholding 成立，但“每条评估样本均保留可执行两跳路径”的前提失败。
- 本结果只审计 split 与语义 witness；未重跑 H200 训练，不对论文 MRR 作替代估计。

## Test split

- Direct edge 成功删除：1000/1000。
- 指定中间点的两条边均保留：379/1000。
- 允许任意替代中间点后，精确 `(a,r1,r2,c)` witness：484/1000。
- 无任何保留两跳 witness：516/1000；这些行不能称为 novel，只能称为 unresolved。
- `(r1,r2 -> r_direct)` schema 在 scrubbed train 中出现：999/1000。

指定 witness 被级联删除的分布：

- 0/2 edges retained：3
- 1/2 edges retained：618
- 2/2 edges retained：379

## Validation split

- 指定路径完整保留：336/1000。
- 任意精确两跳 witness 保留：441/1000。

## 原因

生成器先选出 2,000 个 endpoint pairs，再统一删除这 2,000 条 direct edges。代码没有重新验证待删除边是否同时是其他评估路径 的第一跳或第二跳，因此删除操作破坏了大量先前选中的 witness。

## 行为证据边界

- 仓库没有发布 composition checkpoint、逐条 ranks/predictions、训练日志或机器可读结果 manifest。
- 公开脚本固定一个训练 seed，无法从仓库产物核验论文中“all random seeds”表述。
- 因此该候选可计为一个完成的 protocol-level semantic audit，但不能计为 P3 所需的 credible behavioral evidence。
