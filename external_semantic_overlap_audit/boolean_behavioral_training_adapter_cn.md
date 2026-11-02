# Boolean ICL 冻结协议训练 Adapter

状态：已实现；2-step smoke、六个 seed-0 200-step canary 和 18-run
20k-step 正式矩阵均已完成并通过验收。

## 1. 入口

```text
scripts/run_boolean_behavioral.py
```

该入口不调用上游 `src/train.py`，因为原入口：

- 不能指定逐项 semantic ID；
- 不能排除固定 eval functions；
- 不保存逐 episode 预测；
- 硬编码 CUDA；
- 没有完整随机种子控制。

Adapter 保留公开 `san` Transformer 架构、Boolean 任务方程和输入分布，
同时增加 manifest 驱动的数据控制和审计输出。

## 2. 启动前强制检查

每次执行都会：

1. 校验 frozen `SHA256SUMS`。
2. 校验上游 `args.py`、`bool_tasks.py`、`eval.py`、`models.py` 和
   `train.py` 的 SHA-256。
3. 校验 protocol generator、training adapter 和 requirements 自身哈希。
4. 要求 `run_id` 在冻结的 18 行 `run_manifest.csv` 中恰好出现一次。
5. 重建完整 640,000-function stream 并核对顺序哈希。
6. 对 Strict 再验证 train/eval semantic overlap 为 0。

任一检查失败都会在模型初始化前终止。

## 3. 执行模式

| 模式 | Steps | Eval episodes | 可进入 Gate |
|---|---:|---:|---|
| `smoke` | 2 | 8 | 否 |
| `canary` | 200 | 128 | 否 |
| `full` | 20,000 | 1,280 | 是 |

模式必须显式传入。Canary 和 smoke 输出带
`eligible_for_gate=false`，不能被正式分析脚本读取为主结果。

## 4. 设备

`--device auto` 按 CUDA、MPS、CPU 顺序选择。也可显式指定：

```bash
--device cuda
--device mps
--device cpu
```

当前项目环境：

```text
Python 3.11.1
NumPy 2.4.6
PyTorch 2.12.1
Transformers 4.57.1
```

依赖文件：`behavioral_requirements.txt`。

## 5. 命令

只验证 manifest，不初始化或训练模型：

```bash
cd <SOURCE_WORKSPACE>/external_semantic_overlap_audit
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/run_boolean_behavioral.py \
  --run-id conjunction_d28__original__seed0 \
  --mode full \
  --validate-only
```

200-step canary：

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/run_boolean_behavioral.py \
  --run-id conjunction_d28__original__seed0 \
  --mode canary \
  --device cuda \
  --checkpoint-every 25
```

正式 run：

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/run_boolean_behavioral.py \
  --run-id conjunction_d28__original__seed0 \
  --mode full \
  --device cuda \
  --checkpoint-every 1000
```

中断后恢复：

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/run_boolean_behavioral.py \
  --run-id conjunction_d28__original__seed0 \
  --mode full \
  --device cuda \
  --checkpoint-every 1000 \
  --resume
```

## 6. 输出契约

每个 run 输出到：

```text
runs/boolean-behavioral-v1/<mode>/<run_id>/
```

包含：

- `state.pt`：模型、优化器、已完成步数和 frozen hashes；
- `training_log.jsonl`：loss、累计时间和 throughput；
- `eval_predictions.jsonl`：逐 episode semantic ID、exposure 状态、标签、
  logits 和四项指标；
- `metrics.json`：run-level 聚合指标；
- `COMPLETED.json`：设备、依赖版本、adapter hash、checkpoint/output hash
  和完整 run metadata。

已有 `COMPLETED.json` 的 run 会拒绝覆盖。`--resume` 只接受相同 `run_id`、
模式和 frozen hashes 的 checkpoint。

## 7. 已验证性质

本地 MPS 2-step paired smoke 已验证：

- 17,689,345 个模型参数可完成 forward、backward、checkpoint 和 eval。
- smoke 输出明确标记为不可进入 Gate。
- 非法 `run_id` 在训练前失败。
- 同 seed Original/Strict 在前两步无 collision 时，全部模型张量与 8 条
  eval logits 逐位一致。

六个 seed-0 200-step canary 已完成并通过
`scripts/validate_boolean_canaries.py`：

- 6/6 runs 完成 200 steps 和 128 条固定 eval episodes；
- 6/6 checkpoints、predictions 和 frozen hashes 校验通过；
- 所有模型参数、loss 和 logits 均为有限值；
- paired eval semantic IDs 与 labels 完全一致；
- Strict actual exposure 全部为 0；
- Conjunction、Majority、Sparse Disjunction 的前 6,400 个 Original
  function draws 分别有 1、1、51 个 eval-semantic collisions；
- 六个 run 总训练时间 62.00 秒，聚合吞吐 19.35 steps/s；
- 按该吞吐外推，18-run 正式矩阵纯训练约 5.17 小时，不含启动、评估、
  checkpoint I/O 和热降频。

验证产物：

```text
runs/boolean-behavioral-v1/canary/canary_validation.json
```

Canary 结果均为 `eligible_for_gate=false`，未比较或解释行为 effect。下一步
是执行冻结的 18-run 正式矩阵。

## 8. 正式矩阵调度

顺序调度器：

```text
scripts/run_boolean_behavioral_matrix.py
```

它按 frozen `run_manifest.csv` 顺序执行 18 个 full runs，自动跳过合法完成项、
恢复已有 checkpoint，并在任一 run 失败时停止。矩阵状态不读取或比较 effect。

正式矩阵已通过 `caffeinate` 在 MPS 完成。调度记录：

```text
runs/boolean-behavioral-v1/matrix/full_matrix_status.json
runs/boolean-behavioral-v1/matrix/full_matrix_driver.log
```

单 run 训练日志：

```text
runs/boolean-behavioral-v1/full/<run_id>/training_log.jsonl
```

当前调度参数：

```text
device=mps
checkpoint_every=1000
eval_batch_size=64
```

## 9. 正式结果

18/18 runs 通过 checkpoint、optimizer、逐题输出、训练日志、Strict 零
exposure 和 paired eval identity 验收。预注册 Core Gate 与 Strong-pattern
Gate 均为 **No-Go**：

| Task | Original - Strict mean | Hierarchical 95% CI |
|---|---:|---:|
| Conjunction-28 | -0.01 pp | [-0.09, +0.04] |
| Majority-28 | -0.06 pp | [-0.20, +0.08] |
| Sparse Disjunction-50 | +0.09 pp | [-0.06, +0.28] |

因此不扩展 seeds 3/4，不触发 LSTM ranking module，也不允许
neural-memorization、strict-split performance-drop 或 ranking-change claim。
完整报告：

```text
analysis/boolean-behavioral-v1/report_cn.md
```
