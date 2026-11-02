# Code and Data Supplement

This anonymous archive accompanies:

> **When Is a Symbolic Rule Really Unseen? Auditing Function-Level
> Generalization Across Neural Architectures**

The paper is a benchmark audit whose primary evidence comes from matched task
adaptations of six published architecture families:

- Deep Sets
- Relation Network
- Set Transformer
- modern Hopfield-style retrieval
- Perceiver-style latent attention
- Transformer

The audit compares the legacy parameter-identity protocol with a corrected
function-equivalence-aware protocol and then evaluates frozen strict
checkpoints on held-out rule families.

## Main Results

- The base generators contain 560 parameter identities but only 518 complete
  functions.
- Across 20 legacy split seeds, 14.8/140 held-out identities on average
  duplicate a training function (10.57%).
- Across the six public architectures, legacy nominal accuracy exceeds strict
  accuracy by an architecture macro-average of 2.31 percentage points, with
  paired 95% CI [0.84, 3.79].
- The architecture ranking is unchanged: Spearman rho is 1.0 and every rank
  movement is zero.
- Strict known-family accuracy spans 38.86% to 72.93%, but held-out-family
  accuracy spans only 8.14% to 11.53%.
- Family-conditioned exact Bayes reaches 99.17% on the public held-out-family
  manifests. The six-family family-agnostic posterior-expected ceiling is
  98.93%.

KCE/LOP is included only as the failed hypothesis that motivated the audit.

## One-Command Verification

Create the pinned environment:

```bash
python3 -m venv src/.venv
source src/.venv/bin/activate
python -m pip install -r src/kfwa_experiments/requirements.txt
```

Then run:

```bash
python reproduce.py all
```

This command:

1. validates anonymity and all compact result gates;
2. validates the strict/legacy splits, all six public architectures, exact
   oracles, and the family-agnostic information reference;
3. runs a tiny Deep Sets/Transformer training smoke test;
4. regenerates the public-architecture figure from compact CSV files;
5. compiles the main and supplementary papers.

Outputs are written under `build/`. Individual targets are:

```bash
python reproduce.py validate
python reproduce.py smoke
python reproduce.py figures
python reproduce.py paper
python reproduce.py clean
```

The one-command path validates released summaries and exercises the complete
software path; it does not retrain the 60 full public-baseline runs.

## Archive Layout

- `src/kfwa_experiments/kfwa/`: datasets, models, metrics, and training code.
- `src/kfwa_experiments/scripts/`: validators, launchers, and evaluators.
- `src/paper_scripts/`: paper figure generator.
- `results/public_baseline_audit/analysis/`: matched legacy/strict aggregates,
  paired intervals, rankings, and claim gates.
- `results/public_baseline_audit/family_transfer/`: frozen public-baseline
  quadratic/piecewise-affine results.
- `results/phase2_5/`: strict exact-ceiling and attribution controls.
- `results/phase2_9/`: family-agnostic Bayesian information reference.
- `results/core_symbolic/`, `results/boundary_symbolic/`, and
  `results/table123_recent_baselines/`: historical pre-audit summaries retained
  for provenance; they are not primary paper evidence.
- `paper_sources/`: manuscript and figure source snapshots.

Large artifacts are intentionally omitted: there are no checkpoints,
episode-level CSV files, caches, or external datasets. Symbolic episodes are
generated from fixed seeds.

## Primary Public-Baseline Audit

The frozen protocol is:

- alphabet size `N=20`;
- support size `K=8`;
- 10% support-label corruption;
- query always outside support;
- training families `affine,block,parity`;
- seeds `0-4`;
- 20,000/5,000/5,000 train/validation/test episodes;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size 256, maximum 60 epochs, patience 10;
- shared width 64, two layers, four heads, feed-forward width 128 where
  applicable;
- no architecture-specific tuning after audit inspection.

Run structural and split validation:

```bash
cd src
python kfwa_experiments/scripts/validate_symbolic_public_baselines.py
```

Run all 60 matched training jobs and aggregate them:

```bash
cd src
python kfwa_experiments/scripts/run_public_baseline_split_audit.py \
  --out-dir ../runs/public_baseline_split_audit_k8_5seeds \
  --models symbolic_deepsets,symbolic_relation_network,symbolic_set_transformer,symbolic_hopfield,symbolic_perceiver,symbolic_transformer \
  --protocols legacy_identity,strict_function \
  --seeds 0-4 \
  --device auto
```

The launcher is restart-safe: completed matching runs are discovered and
skipped. Recompute only the aggregate after training with:

```bash
cd src
python kfwa_experiments/scripts/run_public_baseline_split_audit.py \
  --out-dir ../runs/public_baseline_split_audit_k8_5seeds \
  --analysis-only
```

Evaluate the strict checkpoints on held-out families without retraining:

```bash
cd src
python kfwa_experiments/scripts/evaluate_public_baseline_family_transfer.py \
  --runs-dir ../runs/public_baseline_split_audit_k8_5seeds/strict_function \
  --out-dir ../runs/public_baseline_split_audit_k8_5seeds/family_transfer \
  --models symbolic_deepsets,symbolic_relation_network,symbolic_set_transformer,symbolic_hopfield,symbolic_perceiver,symbolic_transformer \
  --seeds 0-4 \
  --families quadratic,piecewise_affine \
  --device auto
```

The evaluator verifies that the six checkpoints for each seed share the same
strict split checksum and records checkpoint hashes and held-out candidate-set
checksums.

## Protocol and Oracle Validation

Validate deterministic generation, out-of-support queries, corruption rate,
and zero strict function overlap:

```bash
cd src
python kfwa_experiments/scripts/validate_symbolic_protocol.py
```

Validate Bayesian and maximum-likelihood probabilities against an independent
brute-force implementation:

```bash
cd src
python kfwa_experiments/scripts/validate_symbolic_oracles.py
```

Run the strict K=8 exact information ceiling:

```bash
cd src
python kfwa_experiments/scripts/evaluate_symbolic_ceiling.py \
  --out-dir ../runs/symbolic_ceiling_strict_k8_5seeds \
  --predictors bayes,ml,legacy \
  --seeds 0-4 \
  --k-values 8 \
  --support-noises 0.10 \
  --device cpu \
  --num-symbols 20 \
  --rule-families affine,block,parity \
  --holdout-function-frac 0.25 \
  --query-out-of-support-prob 1.0 \
  --test-limit 5000 \
  --batch-size 256
```

Validate the family-agnostic union prior:

```bash
cd src
python kfwa_experiments/scripts/validate_symbolic_family_agnostic_ceiling.py
```

## Artifact Policy

The compact validator rejects:

- absolute user paths or identity fragments;
- checkpoints and Python bytecode;
- episode-level output;
- files larger than 10 MiB;
- mismatched primary result values or decision gates.

Environment details and the fresh-environment acceptance procedure are in
`ENVIRONMENT.md`.
