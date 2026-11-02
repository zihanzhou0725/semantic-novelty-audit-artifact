# Semantic Novelty Audit Quality-Enhancement Protocol v1

Freeze date: 2026-10-09  
Status: frozen before new confirmatory execution  
P1 training started at freeze: no  
P2 registered analysis run at freeze: no  
P3 candidate-result inspection started at freeze: no

## Purpose

This protocol freezes the article-quality enhancement program before any new
confirmatory P1 training, P2 registered analysis, or P3 candidate audit. It
separates five questions that must not be substituted for one another:

1. P1: Does exact prior exposure to an evaluation function change model
   behavior under a controlled intervention?
2. P2: Is witnessed SLR semantic overlap associated with correctness after a
   frozen adjustment, if a reproducible prediction artifact exists?
3. P3: Does the audit finding replicate across prospectively selected public
   benchmarks?
4. P4: Is the audit implementation portable and independently reproducible?
5. P5: Does the formal analysis add a testable result rather than only new
   terminology?

The article remains an evaluation-methodology and measurement study. No phase
may turn overlap alone into evidence of memorization.

## Registration Status And Prior Knowledge

The status differs by phase and must be reported accurately:

- P1 is prospective. No complementary-exposure training stream or checkpoint
  has been generated.
- P2 is a registered analysis plan for existing data, not a blind
  preregistration. The 45.2% SLR overlap lower bound and the failure of the
  repository-hosted output artifact to reproduce the paper's main table were
  known at freeze time.
- P3 is prospective with respect to candidate-level overlap and behavioral
  results. A candidate manifest must be frozen before semantic extraction.
- P4 and P5 are prospective contribution gates, not statistical hypotheses.

The following historical results are context, not confirmatory evidence for
the new Gates:

- The internal Legacy-minus-Strict contrast is +2.31 percentage points, but it
  jointly changes split unit, function prior, and stratification.
- SLR-Bench has a sound semantic-overlap lower bound of 452/1000 (45.2%).
- The released Boolean Transformer behavioral experiment was No-Go, with
  effects near zero at the registered budget. No new Boolean seeds or LSTM
  ranking experiment may be added under this protocol.

## Global Rules

### Units and thresholds

All accuracy effects are reported in percentage points (pp).

- Meaningful-effect threshold (SESOI): 1.0 pp.
- Equivalence margin: [-0.5, +0.5] pp.
- Confirmatory uncertainty: two-sided 95% intervals.
- The gap between 0.5 and 1.0 pp is intentional and is classified as
  inconclusive rather than rounded into a positive or null result.

### Reporting and multiplicity

- Every registered run, exclusion, endpoint, and failed precondition is
  reported.
- The P1 architecture-macro endpoint and the P2 adjusted marginal association
  are the only primary statistical endpoints.
- P1 model-specific claims use six Bonferroni-simultaneous 99.1667% intervals.
- Secondary metrics and subgroup analyses cannot replace a failed primary
  Gate.
- No seed, model, benchmark, overlap class, or covariate may be added or
  removed after its primary result is inspected.

### Claim vocabulary

The following distinctions are mandatory:

- `protocol overlap`: an evaluation semantic object occurs in training.
- `behavioral effect`: randomized exposure changes a model endpoint.
- `adjusted association`: correctness differs observationally after a frozen
  adjustment.
- `memorization mechanism`: an internal computational mechanism, which is not
  identified by overlap or endpoint changes alone.

Even a positive P1 result allows an exact-exposure effect claim, not an
unqualified claim that a model "memorizes."

## P1: Complementary Exact-Exposure Intervention

### Question and estimand

P1 estimates the intention-to-treat effect of exact semantic exposure on
out-of-support episodic rule-inference accuracy. The target population is the
complete set of cross-family alias-eligible functions in the frozen
`N=20`, `affine,block,parity` generator.

For architecture `m`, model seed `s`, and target function `f`, let
`A(m,s,f,E)` be mean accuracy over the frozen evaluation episodes when the
training arm assigns exact exposure `E` to `f`. The model-level estimand is

```text
tau_m = (100 / (5 * 40)) *
        sum_s sum_f [A(m,s,f,1) - A(m,s,f,0)].
```

The sole primary estimand is the equal-architecture macro:

```text
tau_macro = (1 / 6) * sum_m tau_m.
```

Architectures and functions are fixed targets; the Gate does not generalize
to an unspecified population of all neural architectures or rule families.

### Frozen function census

The protocol must materialize and validate these facts before training:

- 560 parameter identities collapse to 518 complete functions.
- Exactly 40 functions have identities in more than one registered family.
- All 40 have an affine canonical identity and a parity alias.
- Of the 40, 38 have family signature `affine+parity`, and 2 have
  `affine+block+parity`.
- The 40 targets are exactly affine maps with multiplier `a` in `{1,11}` and
  intercept `b` in `{0,...,19}`.
- Pair `f_(1,b)` with `f_(11,b)` for every intercept `b`, giving 20 fixed
  function pairs.

Evaluation uses the affine identity. Target exposure is recorded using the
parity alias. Because family and identity metadata are not model inputs, this
is a parameter-disjoint, cross-family representation of exact function
exposure.

If any census fact fails, P1 is Design No-Go. No nearest replacement family,
target subset, or alternative pairing is allowed in v1.

### Complementary arms

For each model seed and each of the 20 pairs, a deterministic bit derived from
SHA-256 assigns one function to Arm A and the other to Arm B:

```text
payload = "quality-enhancement-v1:20261009:p1-arm:{model_seed}:{b}"
bit = int.from_bytes(SHA256(payload ASCII)[0:8], "big") & 1
```

Bit 0 assigns `f_(1,b)` to Arm A and `f_(11,b)` to Arm B; bit 1 reverses the
assignment. The opposite function is therefore unexposed in that arm. Every
target function is exposed in exactly one arm and unexposed in the
complementary arm for every seed.

The common base pool contains all 478 non-target complete functions and no
identity implementing any of the 40 targets. Each arm contains:

```text
478 common base functions + 20 exposed target functions = 498 functions.
```

The two arms have identical canonical-family counts, the same number of
target exposures, and the same target dose. Arm labels have no substantive
meaning.

### Paired training stream

Each arm has a frozen 20,000-episode training manifest:

- every one of the 478 base functions appears 40 times;
- 80 preselected base functions receive one additional episode;
- every one of the 20 exposed targets appears exactly 40 times;
- total: `478*40 + 80 + 20*40 = 20,000` episodes.

All seed derivation uses:

```text
derive(namespace) =
  int.from_bytes(
    SHA256("quality-enhancement-v1:20261009:" + namespace)[0:8],
    "big"
  ) & (2^63 - 1).
```

For each model seed, sort the 478 base function IDs numerically, permute them
with NumPy `Generator(PCG64(derive("p1-extra:{seed}")))`, and give the first 80
one extra episode. Construct the base-slot multiset and the 20 pair IDs
repeated 40 times, concatenate them, and permute all 20,000 slots with
`PCG64(derive("p1-slots:{seed}"))`. This single paired slot manifest supplies
both arms.

Episode randomness for slot index `i` uses
`PCG64(derive("p1-episode:{seed}:{i}"))` and the existing generator order:
draw eight distinct support digits, draw one query uniformly from the 12
out-of-support digits, and draw one support-corruption uniform plus, when
corrupted, one integer offset in `[1,19]` per support position. A corrupted
label is `(clean_label + offset) mod 20`. The materializer stores full support,
query, corruption, and target tensors for both arms; validation reconstructs
them from the seeds.

Base-function slots, support/query draws, corruption indicators, corruption
offsets, initialization seed, and optimizer schedule are
identical between complementary arms. At a target slot, Arm A and Arm B use
the paired target functions with the same episode-generation seed. The
different clean labels are part of the intervention.

For each epoch `e` from 1 through 60, the adapter consumes the explicit index
permutation produced by `PCG64(derive("p1-epoch:{seed}:{e}"))`; DataLoader
shuffle is disabled. The same epoch permutations and freshly reset model and
dropout seed `model_seed` are used in both arms. Deterministic backend mode is
required; an unsupported nondeterministic operation is a run failure, not a
reason to change the manifest.

The 20,000 frozen episodes are replayed for 60 complete epochs. There is no
arm-specific early stopping or validation-based checkpoint selection. The
primary checkpoint is the final epoch-60 state:

```text
60 * ceil(20,000 / 256) = 4,740 optimizer updates.
```

### Frozen models and optimization

P1 includes all six existing public architecture families:

- Deep Sets;
- Relation Network;
- Set Transformer;
- Modern Hopfield-style memory;
- Perceiver-style model;
- Transformer.

Frozen model seeds are `0,1,2,3,4`, producing:

```text
6 architectures * 5 seeds * 2 arms = 60 training runs.
```

Configuration:

- alphabet size: 20;
- support size: 8;
- support corruption probability: 0.1;
- query: always outside support;
- width: 64;
- layers: 2;
- attention heads where applicable: 4;
- feed-forward width: 128;
- dropout: 0.05;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0.0001;
- batch size: 256;
- gradient clipping: 1.0;
- checkpoint: fixed final epoch 60.

No architecture-specific tuning is allowed after results are observed.

### Frozen evaluation

The common evaluation manifest contains 250 episodes for each of the 40
targets, for 10,000 episodes total. The same byte-identical evaluation tensors
and targets are used for every arm, model seed, and architecture. For target
function ID `f` and episode index `j`, generation uses
`PCG64(derive("p1-eval:{f}:{j}"))` with the same support, query, and corruption
algorithm as training. Evaluation identities are affine, support size is 8,
support noise is 0.1, and every query is outside support. The materializer
stores the full tensors and their hashes.

Primary metric:

- exact query-label accuracy, equally weighted first by target function, then
  model seed, then architecture.

Secondary metrics:

- cross entropy;
- Brier score;
- per-function exposure effects;
- effects by the two frozen alias signatures;
- training and evaluation calibration summaries.

Secondary results are descriptive and cannot trigger the P1 Gate.

### Uncertainty

Use 20,000 hierarchical bootstrap replicates with master seed `20261009`.
For each replicate:

1. resample the five model seeds with replacement;
2. resample the 20 matched function pairs with replacement;
3. within each selected function, resample its 250 evaluation episodes;
4. recompute exposed-minus-unexposed differences;
5. average architectures with equal fixed weights for `tau_macro`.

The primary interval is the percentile 95% interval for `tau_macro`.
Architecture-specific intervals use percentile 99.1667% intervals, equivalent
to Bonferroni family-wise alpha 0.05 across six models.

### P1 decision Gate

Evaluate in this order:

1. **Positive behavioral Go**: `tau_macro >= +1.0 pp`, its 95% lower bound is
   above zero, and at least four of six architecture point estimates are
   positive.
2. **Negative behavioral Go**: `tau_macro <= -1.0 pp`, its 95% upper bound is
   below zero, and at least four of six architecture point estimates are
   negative.
3. **Null-equivalence Go**: the complete macro 95% interval lies inside
   `[-0.5,+0.5] pp`, every model point estimate has absolute magnitude below
   1.0 pp, and no simultaneous model interval excludes zero.
4. **Inconclusive No-Go**: every other outcome.

A model-specific effect claim additionally requires an absolute point estimate
of at least 1.0 pp and that model's simultaneous interval to exclude zero in
the same direction.

Positive Go permits the statement that exact cross-family semantic exposure
improves macro accuracy for the registered architectures and budget. Negative
Go permits an exposure-effect statement but not an inflation claim.
Null-equivalence Go permits a bounded macro null statement, not universal
invariance. Inconclusive No-Go permits only the numerical estimate and
uncertainty.

There is no optional seed expansion, model deletion, dose sweep, or alternative
checkpoint Gate in v1.

## P2: Registered SLR Adjusted Association

### Identification precondition

The sole confirmatory target is the final-paper `Llama-3.1-8B-IT-SLR` result
reported in ACL 2026 Table 2. P2 behavior is analyzed only if a prediction
source for that exact checkpoint/result passes all of the following before
overlap-stratified accuracy is computed:

- all 1,000 test IDs occur exactly once;
- test rule text and gold targets match the pinned dataset for all rows;
- predictions are cryptographically tied to a public checkpoint revision and
  a complete decoding/evaluation configuration;
- the same checkpoint and configuration can regenerate the artifact;
- regenerated per-tier correct counts match the registered artifact exactly,
  or a documented nondeterministic tolerance was frozen before rerun;
- no row was selected or removed using correctness or overlap status.

The currently pinned repository-hosted output fails the published-result
reproducibility requirement and cannot be used to characterize the model
reported in the paper. Unless a new source passes the full precondition, P2
behavior is Identification No-Go. The 45.2% structural lower bound remains
reportable. No different model may replace the confirmatory target. Any
additional prediction sources are exploratory and all passing and failing
sources must be listed in a source manifest frozen before joining predictions
to overlap labels or correctness.

### Exposure label and comparator

The registered exposure indicator is the disjoint union of:

- 70 token-identical rules; and
- 382 additional sound flat-rule semantic aliases.

For bootstrap clustering, a token-identity semantic ID is SHA-256 of canonical
JSON for its complete `token_key`; a flat-alias semantic ID is SHA-256 of
canonical JSON for the final `profile_key` with duplicate collapse and
subsumption enabled. Canonical JSON uses sorted keys, compact separators, and
ASCII encoding. These two witnessed sets have zero intersection. A row
without a witness is clustered by test item ID.

The remaining 548 rows are called `no registered witness`, not `novel`.
Unresolved aggregate, negated, arithmetic, and disjunctive rules prevent a
complete semantic partition.

### Primary adjusted estimand

Outcome is binary exact execution correctness. The sole P2 primary endpoint is
the standardized adjusted marginal association:

```text
100 * mean_i[p_hat(Y=1 | W=1, X_i) - p_hat(Y=1 | W=0, X_i)]
```

over the frozen common-support population, where `W` is witnessed overlap and
`X` contains only pre-outcome covariates:

- curriculum level/ID;
- generator or sampling source;
- rule token length;
- atom count;
- registered logical-structure class.

Covariates are computed from the pinned test Parquet and the pinned
`audit_slr.py` tokenizer:

- curriculum is integer column `curriculum level`;
- generator is exact string column `rule sampling`;
- rule token length is `len(token_key(ground-truth rule))`, including the head
  and punctuation;
- atom count is the number of identifier tokens immediately followed by `(`
  in the body after `:-`, including built-ins and nested calls;
- logical structure is the first matching class in this fixed order:
  `aggregate_or_quantified` for `findall`, `bagof`, `setof`, or `forall`;
  `negated` for `\+` or `not(`; `disjunctive` for `;`;
  `arithmetic_or_comparison` for `is`, `>`, `<`, `>=`, `=<`, `=:=`, or
  `=\=`; `flat_positive` when `parse_flat_rule` succeeds; otherwise
  `other_complex`.

Tokenization failure, a missing body, or any missing covariate is a P2
Identification No-Go; no row-wise imputation or category merging is allowed.

The frozen outcome model is an unregularized Binomial GLM with logit link and
an intercept. Its design is `W`, categorical fixed effects for curriculum,
generator, and structure, plus Patsy natural cubic regression splines
`cr(x, knots=(q25,q50,q75), lower_bound=min, upper_bound=max)` for rule length
and atom count. Quantiles, minima, and maxima are computed on the untrimmed
1,000 rows without outcomes. Categorical reference levels are ascending
lexicographic values. Fit with statsmodels IRLS, tolerance `1e-8`, and maximum
100 iterations; regularization and robust fallback fits are forbidden.

### Common support

Before any propensity fit, retain only curriculum levels containing at least
one witnessed and one no-witness row. This outcome-free check leaves levels
`2,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20`, retaining 352/452 witnessed
rows (77.88%) and 498/548 no-witness rows (90.88%). Levels 1 and 3 contain
only witnessed rows; level 15 contains only no-witness rows.

Fit an unregularized Binomial-logit GLM for `W` with the identical covariate
design, software, and fitting settings, but without correctness. Retain rows
with estimated propensity in `[0.05,0.95]`. Then fit the outcome model on the
retained rows. Identification passes only if:

- at least 80% of each exposure group is retained;
- at least 100 rows from each group remain;
- every categorical coefficient used in the outcome model is estimable;
- no registered covariate is missing.

Otherwise P2 is Identification No-Go. The threshold, covariates, or model may
not be relaxed after seeing accuracy. Nonconvergence, a singular design,
perfect separation, or a nonfinite coefficient in either original-data model
is also Identification No-Go.

The exact-level precheck already misses the registered 80% witnessed-row
minimum, so P2 v1 is Identification No-Go before any outcome fit. No
confirmatory adjusted association will be estimated under v1, even if a new
prediction artifact becomes available. This negative identification result
and the structural 45.2% lower bound remain reportable.

### Uncertainty and Gate

Use 10,000 percentile cluster-bootstrap replicates with seed `20261009`.
Clusters are canonical witnessed semantic IDs; rows without a witness use
their test item ID. Resample clusters with replacement within each observed
`W x curriculum level` stratum, then refit the propensity, trim common
support, and refit the outcome model in every replicate. Continue until
10,000 successful replicates or 20,000 attempts. Failed fits are logged; fewer
than 10,000 successful replicates is Analysis No-Go.

Decision rules mirror P1 but the wording is observational:

- Positive association Go: estimate at least +1.0 pp and 95% lower bound above
  zero.
- Negative association Go: estimate at most -1.0 pp and 95% upper bound below
  zero.
- Null-association Go: the complete 95% interval lies inside
  `[-0.5,+0.5] pp`.
- Otherwise: Inconclusive No-Go.

Token-identity and non-token semantic-alias estimates are secondary. No P2
result permits a causal benefit, contamination inflation, or memorization
claim.

## P3: Prospective External Replication

P3 audits between two and five public benchmark candidates. Before semantic
extraction, a versioned candidate manifest must freeze:

- search date and search strings;
- paper, dataset, code, and checkpoint revisions;
- the exact unseen-rule/function/generalization claim being audited;
- generator and split definitions;
- available semantic signature or executable equivalence procedure;
- behavioral artifact availability;
- inclusion or exclusion reason independent of audit results.

Discovery uses the first 100 results returned on the recorded search date from
Semantic Scholar and arXiv for each exact query:

```text
"rule inference" benchmark unseen
"in-context learning" Boolean functions
"symbolic reasoning" benchmark held-out rules
"compositional generalization" unseen functions benchmark
```

Deduplicate by DOI, then arXiv ID, then normalized title. Screen in ascending
normalized-title order. Include the first five eligible candidates, or all
eligible candidates when fewer than five; if fewer than two are eligible, P3
is No-Go. The raw result pages, order, metadata, and exclusion reasons are
frozen before any candidate dataset is downloaded or semantically inspected.
SLR-Bench, `satwik77/incontext-bool`, and the internal
`N=20 affine/block/parity` generator are historical manuscript cases and are
ineligible for P3; variants that only repackage the same generator or split
are also ineligible. P3 counts only newly audited external benchmarks.

A candidate is eligible only when its paper and evaluation data are public,
its target claim is semantic rather than only lexical, its training/evaluation
generator can be inspected, and at least a sound signature or executable
semantic witness can be produced. Behavioral evidence additionally requires
a controlled intervention or a prediction source satisfying a P2-strength
reproducibility check.

Candidates cannot be ranked or retained using measured overlap or model
accuracy. Every eligible candidate in the frozen manifest is reported.

P3 Breadth Go requires:

- at least two eligible public benchmarks with complete protocol-level
  semantic audits;
- a sound numerator and explicit unresolved denominator for each;
- at least one benchmark with credible behavioral evidence from a controlled
  intervention or a fully reproducible registered adjusted analysis;
- no result-dependent candidate exclusion.

Otherwise P3 is No-Go for a cross-benchmark prevalence or behavior claim. The
paper may still present bounded case studies.

## P4: Reproducible Audit Toolkit

The toolkit must expose one versioned command and a common schema containing:

- source revisions and hashes;
- train/evaluation identity records;
- semantic IDs and equivalence-witness type;
- `exact`, `sound_lower_bound`, or `unresolved` completeness status;
- overlap counts and denominators;
- behavioral design and estimand metadata when present;
- claim Gate outcome and allowed wording.

Required validation:

- unit tests for exact-function, sound-alias, and unresolved canaries;
- schema validation for every output;
- clean-environment reproduction of frozen summary counts;
- deterministic hashes for manifests and analysis inputs;
- adapters for at least two public benchmarks with materially different
  semantic representations.

P4 Tool-Contribution Go requires all validations and both public adapters.
A command-line wrapper around benchmark-specific scripts is not a method
contribution and is P4 No-Go, though it may remain an artifact.

## P5: Formal Contribution

P5 may formalize:

- why parameter or string disjointness does not imply function disjointness
  under a non-injective representation-to-semantics map;
- finite-support exposure probability;
- the separation between protocol contamination and behavioral exploitation.

P5 Theory-Contribution Go requires at least one proposition that:

- has explicit assumptions and a complete proof;
- yields a falsifiable audit rule, bound, or estimator;
- is instantiated on at least two audited benchmarks;
- changes an empirical design or interpretation in this protocol.

Definitions or restatements without a new consequence are P5 No-Go and must
be presented as framework language, not a theory contribution.

## Submission Gates

### Workshop-ready Go

An internal workshop-ready decision requires:

- this P0 protocol validates;
- P4 reproduces all currently claimed numbers from a clean environment;
- either P1 reaches Positive, Negative, or Null-equivalence Go, or both P2
  Identification and P3 Breadth reach Go;
- all attempted analyses and No-Go outcomes remain in the paper;
- the title, abstract, and conclusion obey the claim vocabulary above.

An inconclusive P1 alone is not upgraded by the historical +2.31 pp contrast.

### CCF-A main-track Go

Main-track escalation requires all workshop conditions plus:

- P3 Breadth Go on at least two public benchmarks;
- at least one credible behavioral result;
- P4 Tool-Contribution Go;
- a contribution statement that is not only a re-analysis of one benchmark.

P5 is optional for submission, but a theory contribution may be claimed only
if P5 reaches Go. Failing this Gate does not invalidate the measurements; it
sets a venue and claim boundary.

## Stopping, Amendments, And Order

The execution order is:

1. validate P0;
2. implement the P1 materializer and run non-training canaries;
3. freeze P1 target, pair, train-slot, evaluation, and run manifests;
4. revalidate all P1 Design Gate invariants;
5. only then start the 60 P1 runs;
6. evaluate the P2 reproducibility precondition before any registered P2 fit;
7. freeze the P3 candidate manifest before candidate semantic extraction;
8. evaluate P4 and P5 against their fixed Gates.

P1 has no efficacy stopping rule. Infrastructure failures may be retried only
from the same initialization and manifest; all retries are logged. Missing
runs are not replaced by new seeds.

Any change to an estimand, threshold, target set, model set, source selection,
common-support rule, or submission Gate requires a new version frozen before
the affected result is inspected. Typographical corrections that do not
change execution or interpretation are recorded with old/new hashes.

## Claim Boundary

Allowed if supported by the corresponding Gate:

- parameter/string disjointness can fail to ensure semantic novelty;
- exact exposure changes or does not materially change the registered P1
  macro endpoint within the stated bounds;
- witnessed SLR overlap has a registered adjusted association for a named,
  reproducible prediction source;
- the protocol defect or behavioral finding replicates in the exact public
  benchmarks passing P3.

Forbidden without additional evidence:

- semantic overlap necessarily improves accuracy;
- the internal historical +2.31 pp is caused only by aliases;
- the current SLR repository output represents the published main-table
  model;
- all architectures, tasks, checkpoints, or budgets rely on memorization;
- unresolved SLR rows are semantically novel;
- a failed or inconclusive Gate is evidence of exact equivalence;
- workshop or main-track readiness when its full Gate is not met.

## Frozen Files

- `README.md`: human-readable protocol.
- `preregistration.json`: machine-readable source of truth.
- `validate.py`: read-only structural, consistency, source-pin, and checksum
  validator.
- `SHA256SUMS`: hashes of the three frozen inputs above.
- `validation_report.json`: generated validation result; excluded from the
  checksum set to avoid a circular artifact.

Validate without training:

```bash
cd <SOURCE_WORKSPACE>
.venv/bin/python \
  external_semantic_overlap_audit/protocols/quality-enhancement-v1/validate.py
```
