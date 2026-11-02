# In-Context Occam's Razor 层级 ICL 语义审计

范围：冻结候选 `7b9ebd45a3528701`，代码 revision `3b8fe0d69dd9bb56c39d933525bf0234feefcc39`，论文 `arXiv:2506.19351v1`。按冻结协议，Markov chain 与 linear regression 都是必审范围，PCFG/GPT-4 不能替代。

## 结论

- **Markov protocol defect detected**：论文称 inference chains 为 "unseen"，但公开代码让 evaluation 与 training 共享 latent-task seed 空间。
- **Fresh-task premise failed**：训练 seed 定义为 `step+i`，相邻 mini-batches 大量复用同一个 transition tensor。
- **Frozen scope incomplete**：pinned repository 完全没有 linear-regression generator、训练、评估或 curriculum 实现。
- 因此该候选不能计为 P3 的 complete public semantic audit，也没有 credible behavioral evidence。

## Markov 精确碰撞界

论文 Fig. 2 与 Fig. 3 的横轴均到 800k sequences；Appendix C 指定 batch size 32，对应 nominal 25,000 steps。公开评估代码为每个 order 固定 10 个 transition matrices，每个 matrix 生成 100 条序列。

- Fig. 2（orders 1/3）：2-4/20 个 eval latent tasks 与训练精确重合；对应 200-400/2000 条 eval sequences。
- Fig. 3（orders 1/2）：2-4/20 个 eval latent tasks 与训练精确重合；对应 200-400/2000 条 eval sequences。
- 碰撞 eval seeds 为 `5` 与 `12579`。具体落在哪个 order 无法恢复，因为 order assignment 使用未设 seed 的全局 `np.random`，且没有发布 task manifest；但每个 seed 在两个训练 order 中至少落入一个，所以上述 10% 下界是确定的。

## 训练任务复用

- 每个主实验生成 800,000 个训练 episodes，只有 25,031 个不同 seed ids。
- 即使按最有利情况让每个 seed 同时覆盖两个 orders，也至多有 50,060 个不同 latent tasks。
- 因而至少 749,940/800,000（93.7425%）训练 episodes 复用了已出现的 `(order, transition tensor)`；这与“fresh tasks at each optimization iteration”不一致。

## Linear regression 未解析分母

- 论文范围为 `d=10/20`、`T=39`、12 layers、8 heads、dim 256，每个报告聚合使用 2,000 个独立 regressors 与 100 次 bootstrap。
- pinned repository 的 12 个 tracked files 中没有 regression generator、coefficient vectors、task seeds、run config、checkpoint 或逐 context prediction。
- 所有 linear-regression evaluation contexts 均属于 unresolved denominator；`0` 个已验证碰撞不能解释为语义新颖。

## 行为证据与 P3 Gate

- `eval.py` 静态引用缺失的 `train.py`、`linear_probe.py` 及不存在的 exports；公开 evaluation entrypoint 在 pinned revision 上结构性不可执行。
- 仓库没有 checkpoint、训练日志、机器可读结果或逐条 predictions，因此论文 aggregate plots 未达到 P2-strength reproducibility。
- P3 当前为 `1/2` 个 complete audits、`0/1` 个 credible behavioral benchmarks，Breadth Gate = **No-Go**。两个冻结候选均保留报告，不存在结果依赖排除。
