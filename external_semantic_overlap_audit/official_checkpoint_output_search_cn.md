# 官方 checkpoint 与逐题输出检索

检索日期：2026-10-07  
目标：优先复用公开 checkpoint、日志或逐题输出，避免重新训练模型。  
状态：检索完成；SLR 找到 checkpoint 和逐题文件，Boolean ICL 未找到可复用行为产物。

## 1. 结论

| 工作 | checkpoint | 逐题输出 | 能否代表论文主表 |
|---|---|---|---|
| Boolean ICL | 未公开 | 未公开 | 否 |
| SLR Llama-3.1-8B | 已公开，约 16.1 GB | 已公开，1,000 条 | 否，分层准确率与论文不符 |
| SLR Qwen3-8B | 已公开，约 16.4 GB | 未公开 | 未验证 |

当前不能通过公开产物直接得到可用于论文主张的 matched behavioral
effect。SLR 逐题文件可用于验证分析管线和探索性分层，但不能冒充 ACL 2026
Table 2 所报告模型的行为。

## 2. Boolean ICL

### 2.1 检索范围

- 正式仓库：
  `https://github.com/satwik77/incontext-bool`
- 固定 commit：
  `a25a2cb0c62ad541cd30b0672df136230d878d9c`
- ICLR proceedings / OpenReview Supplementary ZIP。
- GitHub releases、仓库历史与作者公开仓库。
- Hugging Face model/dataset search。
- Supplementary 中的 notebook、配置、W&B 字段和本地输出引用。

### 2.2 结果

正式仓库没有 Git LFS 对象、release、checkpoint、CSV、JSONL、Parquet 或
已保存指标文件。Hugging Face 按论文名、仓库名和任务名检索也没有匹配的
model 或 dataset。

Supplementary 的 `eval.ipynb` 不是结果包。它读取未随附件发布的
`../models_sample`，并使用示例 UUID：

```text
f8e3a309-aac1-4dc7-aebf-e34ca8ea1085
```

代码中还存在另一个硬编码 eval UUID：

```text
0fd47ac6-96a9-46f2-8bd6-fde24eed1fff
```

但附件里没有对应 run 目录。W&B 配置的 entity 是占位值 `name`，无法据此
解析公开实验项目。

### 2.3 为什么聚合图不能恢复逐题结果

Boolean evaluator 会重新采样目标函数和输入序列。公开图表只保留按
in-context position 聚合的 loss/accuracy，没有保存：

- 每个 eval function 的 truth table 或参数；
- function ID；
- 每条 prompt 的输入输出；
- checkpoint 与 eval seed 的完整映射。

因此不能从曲线反推出某个测试函数是否在训练中出现，更不能构造
seen/unseen matched comparison。

### 2.4 可执行方案

零训练方案只有两种：

1. 向作者索取 `models_sample`、主表 checkpoint、逐题 eval dump、训练和
   eval seeds。
2. 若作者只提供 checkpoint，修改 evaluator，在采样时额外保存 canonical
   truth-table ID、prompt、prediction 和 checkpoint hash，再执行纯推理。

若都不可得，才进入已预注册的最小重训：一个架构、三个任务、三种子、两个
协议，共 18 runs。

## 3. SLR

### 3.1 找到的官方产物

Hugging Face 组织 `AIML-TUDA` 发布：

- `https://huggingface.co/AIML-TUDA/Llama-3.1-8B-SLR`
- `https://huggingface.co/AIML-TUDA/Qwen3-8B-SLR`

Llama 仓库 revision：

```text
f817183e6b19f822dcdd79c24f4a4b976248c99f
```

四个 safetensors 分片合计 `16,060,556,096` bytes。仓库还包含
`model_outputs.json`：

```text
size:    18,069,529 bytes
rows:    1,000
SHA-256: ecd979d5c1f5ed54e11197830b2afb8e4924075a97babf80b188bea630b5a15b
```

Qwen 仓库有四个 checkpoint 分片，但没有逐题输出。SLR leaderboard 和
evaluation Space 只提供聚合评测流程或其他模型结果，没有补齐该 tuned
checkpoint 的逐题行为。

### 3.2 对齐与评分

Llama 输出文件的 1,000 个 `problem_id` 与本地固定的 `v1-All` test split
全部对齐，`ground_truth rule` 也逐条完全一致。

评分脚本镜像官方 verifier 的：

- target rule 抽取；
- validation program 修正；
- 正负例 extensional correctness；
- exact rule accuracy。

本地执行使用 Trealla Prolog WASI v3.12.6，而非官方 SWI-Prolog：

```text
Trealla SHA-256:
f1e4f72d5794d6330611d3d369e46e1a328485acd66ac680e0481d0c714b7710
```

全量结果：

| 分组 | n | Rule accuracy |
|---|---:|---:|
| Overall | 1,000 | 33.10% |
| Flat semantic duplicate | 382 | 55.50% |
| Token duplicate | 70 | 21.43% |
| Unresolved or novel | 548 | 18.98% |

上述粗分层严重受 curriculum difficulty 混杂。把两类 duplicate 合并，只在
两组都出现的 17 个 curriculum level 内比较：

| 口径 | Duplicate - unresolved |
|---|---:|
| level delta 未加权均值 | -0.36 pp |
| 按每层较小组样本数加权 | +1.53 pp |

这只是描述性结果。overlap 没有随机化，同一 level 内仍可能存在规则复杂度
差异，不能解释为 memorization 的因果效应。

### 3.3 与论文主表不一致

| 来源 | Basic | Easy | Medium | Hard |
|---|---:|---:|---:|---:|
| 托管逐题文件，本地评分 | 51.6% | 55.2% | 20.4% | 5.2% |
| ACL 2026 final Table 2 | 96% | 54% | 18% | 5% |
| arXiv v2 Table 2 | 92% | 77% | 17% | 2% |

本地有 32 条 runtime error，其中 30 条属于 Basic。即使把全部 Basic runtime
error 都改判正确，Basic 上界也只有 63.6%，仍远低于 final 的 96%。全部
runtime error 都改判正确时，overall 上界为 36.3%，也低于 final 四档宏平均
43.25%。

因此差异不能仅归因于 Trealla/SWI runtime。文件内部 `model_name` 只有
`merged`，模型卡也没有给出生成配置或该文件与论文表格的映射。合理状态是：

```text
behavioral_use_status = exploratory_only_not_reported_main_table_behavior
```

### 3.4 可执行方案

优先级如下：

1. 向作者索取 ACL final Table 2 对应的逐题输出、checkpoint revision、
   decoding 参数和 SWI verifier 版本。
2. 若只确认当前 HF checkpoint 就是 final checkpoint，则下载约 16.1 GB
   权重并在固定 `v1-All` 上重新生成；只有 tier aggregate 复现主表后，才做
   overlap 分层。
3. 若无法确认 provenance，不为当前输出补做昂贵推理。它只保留为 scorer
   canary 和探索性负结果。
4. 若仍需无学习参数证据，可实现 canonical-rule retrieval control；它只能
   证明 overlap 可被检索利用，不能替代模型 behavioral effect。

## 4. 产物

- `scripts/analyze_slr_official_outputs.py`
- `scripts/run_trealla_wasi.js`
- `artifacts/slr_official_output_stratification.json`
- `artifacts/slr_official_output_per_item.csv`
- `upstream/llama31_8b_slr_model_outputs.json`
- `upstream/prolog_runtime/wasm/tpl-wasm-wasi/tpl.wasm`

复现：

```bash
cd <SOURCE_WORKSPACE>/external_semantic_overlap_audit
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/analyze_slr_official_outputs.py
```
