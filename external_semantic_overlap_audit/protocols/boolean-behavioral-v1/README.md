# Boolean ICL Behavioral Experiment Preregistration v1

Freeze date: 2026-10-07  
Status: frozen before training  
Training performed during freeze: no

## Objective

Measure whether prior exposure to the exact evaluation semantics changes the
behavior of one released Boolean ICL Transformer under a controlled
intervention.

This is an independent experiment using the released generator. It is not a
reproduction of unreleased ICLR 2024 checkpoints or private task
configurations.

## Source

- Repository: `https://github.com/satwik77/incontext-bool`
- Revision: `a25a2cb0c62ad541cd30b0672df136230d878d9c`
- Generator: released `src/bool_tasks.py`
- Sparse Disjunction scope: released-code `d=50`, exactly four positive
  literals. It must not be described as the paper's signed `k<=3` task.

The exact source-file hashes are recorded in `preregistration.json`.

## Semantic Identity

Semantic identity is task-specific and exact for these released families:

| Task | Semantic ID | Why it is exact |
|---|---|---|
| Conjunction-28 | Base-3 encoding of the 28-dimensional vector in `{0,+1,-1}` | Each signed literal set induces a unique conjunction truth table |
| Majority-28 | 28-bit active-variable mask | Each subset induces a unique released majority truth table |
| Sparse Disjunction-50 | 50-bit mask with exactly four set bits | Each four-variable subset induces a unique positive disjunction truth table |

Conjunction and Majority injectivity were independently checked by exhaustive
truth-table enumeration at `d=6`. Sparse Disjunction injectivity follows by
assigning one differing variable to separate any two subsets.

## Frozen Split

For each task, `eval_functions.jsonl` contains 1,280 function draws sampled
with replacement from the released distribution. Every row stores:

- encoded and readable semantic ID;
- the decoded task parameter;
- fixed NumPy and Torch prompt seeds.

The same evaluation functions and prompts are used for every model seed and
both protocols.

Each training run contains:

```text
20,000 optimizer steps x batch size 32 = 640,000 function draws
```

The protocols are:

- `original`: use the source-distribution function at each training slot.
- `strict_function_disjoint`: retain the same function at every noncolliding
  slot and replace only functions whose semantic ID occurs in the frozen eval
  manifest. Replacement draws come from an independently seeded stream and
  are rejected until they are outside the eval set.

Thus every strict stream has zero train/eval semantic overlap, while all
noncolliding slots remain paired with Original.

### Frozen Exposure

Original-protocol eval exposure across model seeds:

| Task | Seed 0 | Seed 1 | Seed 2 | Mean | Analytic expectation |
|---|---:|---:|---:|---:|---:|
| Conjunction-28 | 0.94% | 0.70% | 0.78% | 0.81% | 0.91% |
| Majority-28 | 8.83% | 8.28% | 9.53% | 8.88% | 9.46% |
| Sparse Disjunction-50 | 93.05% | 92.81% | 94.45% | 93.44% | 93.79% |

All strict-protocol exposure rates are exactly zero.

## Model And Budget

The core experiment contains 18 runs:

```text
3 tasks x 2 protocols x 3 model seeds
```

Frozen configuration:

- architecture: released `san` / `src.models.TransformerModel`;
- hidden size: 256;
- layers: 6;
- heads: 8;
- dropout: 0;
- context capacity: 150 positions;
- examples per episode: 40;
- precision: float32;
- optimizer: Adam, learning rate `1e-4`, default betas, no weight decay;
- loss: released sigmoid plus binary cross entropy;
- checkpoint: fixed final step 20,000, with no validation-based selection;
- model seeds: 0, 1, 2.

Paired Original/Strict runs share the model seed and input RNG seeds.

## Metrics

### Primary endpoint

For each episode, compute sign accuracy over zero-indexed positions 20 through
39, then average over episodes:

\[
A_{\mathrm{tail}}=\frac{1}{20}\sum_{t=20}^{39}
\mathbf{1}[\operatorname{sign}(\hat y_t)=y_t].
\]

The registered effect is:

\[
\Delta_{\mathrm{pp}}
=100\left(A_{\mathrm{original}}-A_{\mathrm{strict}}\right).
\]

Report each model-seed delta. Uncertainty uses paired cluster bootstrap over
semantic function IDs within each seed; prompt positions are not treated as
independent samples.

### Secondary endpoints

- final-position accuracy;
- accuracy over positions 1 through 39;
- tail binary cross entropy;
- Original accuracy stratified by whether that eval semantic ID was actually
  sampled during training.

Secondary endpoints cannot replace the primary endpoint for the Gate.

## Go/No-Go Gate

Core Go requires at least one preregistered task to satisfy both:

1. mean Original-minus-Strict tail accuracy is at least `+2.0 pp`;
2. all three model-seed deltas are positive.

Strong-pattern evidence additionally requires:

- Sparse Disjunction delta at least `+5.0 pp`;
- absolute Conjunction delta at most `2.0 pp`;
- task effects increase from low to high expected semantic exposure.

If no task passes Core Go, report all results, stop the behavioral expansion,
and do not claim observed neural reliance on semantic exposure.

## Conditional Expansion

- Add Transformer seeds 3 and 4 only for tasks passing Core Go.
- Run the released LSTM ranking experiment on Sparse Disjunction only if the
  Transformer Sparse Disjunction result passes Core Go.
- A ranking-change claim requires the Transformer/LSTM ordering to reverse
  between Original and Strict. Otherwise that claim is forbidden.

## Claim Boundary

If Core Go passes, the experiment may support:

- the tested Transformer benefits from prior exposure to evaluation semantics;
- Original exceeds Strict by the measured paired delta for the passing task.

It does not by itself support:

- claims about the unreleased original-paper checkpoint;
- universal neural memorization;
- ranking changes without the conditional LSTM experiment;
- accuracy inflation in every benchmark containing semantic overlap.

## Files

- `preregistration.json`: machine-readable frozen specification.
- `eval_functions.jsonl`: exact evaluation function identities and prompt seeds.
- `run_manifest.csv`: all 18 runs, seeds, stream hashes, and exposure counts.
- `split_manifest.json`: aggregate split metadata and invariants.
- `SHA256SUMS`: frozen artifact hashes.
- `validation_report.json`: latest independent validation result.

Regenerate and validate without training:

```bash
cd <SOURCE_WORKSPACE>/external_semantic_overlap_audit
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/freeze_boolean_behavioral_protocol.py
<SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/validate_boolean_behavioral_protocol.py
```
