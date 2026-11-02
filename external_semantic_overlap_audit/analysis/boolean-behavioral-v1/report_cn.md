# Boolean ICL 严格语义划分行为实验

分析日期：2026-10-08  
输入：18 个冻结 full runs；每个 run 20,000 steps、1,280 个固定 eval episodes。

## 完整性

- 18/18 runs 通过 checkpoint、输出哈希与有限值检查。
- Strict train/eval semantic overlap 为 0。
- Original/Strict paired semantic IDs、labels 与 exposure flags 完全一致。

## 主指标

主指标为位置 20--39 的 tail accuracy，效应定义为

\[
\Delta_{\rm pp}=100(A_{\rm original}-A_{\rm strict}).
\]

| Task | Seed 0 | Seed 1 | Seed 2 | Mean | Hierarchical 95% CI | Core Gate |
|---|---:|---:|---:|---:|---:|---|
| Conjunction-28 | -0.00 | -0.06 | 0.02 | -0.01 | [-0.09, 0.04] | No-Go |
| Majority-28 | -0.14 | -0.10 | 0.05 | -0.06 | [-0.20, 0.08] | No-Go |
| Sparse Disjunction-50 | -0.03 | 0.25 | 0.06 | 0.09 | [-0.06, 0.28] | No-Go |

置信区间不参与 Gate；Gate 仅使用预注册的 `>=2 pp` 与 3/3 seeds 同方向规则。

## Gate

- Overall Core Gate: **No-Go**
- Passing tasks: none
- Strong-pattern Gate: **No-Go**
- Transformer seeds 3/4 expansion: none
- LSTM ranking module: not triggered

## 次指标

| Task | Final acc Δ (pp) | Positions 1--39 Δ (pp) | Tail BCE Δ |
|---|---:|---:|---:|
| Conjunction-28 | 0.05 | 0.01 | 0.0007 |
| Majority-28 | -0.13 | -0.07 | -0.0001 |
| Sparse Disjunction-50 | 0.16 | 0.07 | -0.0019 |

BCE 使用 Original - Strict；负值表示 Original 更好。

## Claim 边界

Core No-Go：不得声称该受测 Transformer 依赖 semantic exposure，行为影响部分按预注册规则停止扩展。

Original 内部 seen/unseen 分层仅作描述，因为 exposure 不是随机分配，不能替代 Original/Strict 受控对照。
