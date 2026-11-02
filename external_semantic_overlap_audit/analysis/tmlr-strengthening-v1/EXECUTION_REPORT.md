# TMLR Strengthening Execution Report

## Summary

| Task | Status | Key result | Anchor/checks | Wall time | Skip reason |
|---|---|---|---:|---:|---|
| 2 | complete | mean 23.3391 pp; Spearman -0.84748 | 15/15 | 5 s | - |
| 3(a) | complete | mean 1.8320 pp; Spearman -0.72740 | 10/10 | 3 s | - |
| 10 | revised | parameterized sensitivity, structural M=548 only | exhaustive width verification | <1 min | model-score application withdrawn |
| 9, reduced | complete | MILP 100/100; dual checks 100/100 | 5/5 | 2 s | - |
| 6 | complete | C1-C0 +22.475 pp; C2-C0 +3.709 pp | no preset anchor | about 3 h 35 min | - |
| 1 | skipped | - | - | - | excluded by user |
| 3(b) | skipped | - | - | - | excluded by user |
| 4 | skipped | - | - | - | excluded by user |
| 5 | skipped | - | - | - | excluded by user |

## Deviations from the Full Specification

1. The task set follows the explicitly approved reduced scope rather than
   the original full sequence.
2. Task 9 implements exact MILP bounds, LP outer bounds, independently
   verified dual certificates, and focused properties. It does not add the
   laminar closed form, total-unimodularity theorem, or complexity theorem.
3. Task 3(b) displacement was not run even though checkpoints are present,
   because the user excluded it.
4. Task 6 used deterministic float32 MPS execution instead of the original
   V4 CPU device; architecture, optimizer, batch sizes, seeds, epoch orders,
   and training budget are unchanged.
5. Two interrupted scheduling attempts produced no final metrics and their
   incomplete generated directories were deleted before clean reruns. The
   frozen protocol and implementation hashes did not change.
6. Per-task Git commits were unavailable because the project is not a Git
   worktree.

## Scope and Path Mapping

This execution follows the user-approved reduced scope:

- Task 2: CLUTRR V4 pattern heterogeneity.
- Task 3(a): internal finite-function heterogeneity.
- Task 10: novelty-conditioned score intervals.
- Task 9, reduced: MILP/LP bounds, independently verified dual
  certificates, and focused property/toy tests. The unneeded laminar
  closed form, total-unimodularity argument, and complexity theorem were
  not added.
- Task 6: semantic exposure versus surface-similar decoys.

Tasks 1, 3(b), 4, and 5 are excluded by explicit user decision.

The specification's packaged paths were mapped to the complete private
repository as follows:

- `$CODE`: repository root.
- `$ROOT/04_RESULTS/analysis`: `analysis/`.
- `$ROOT/04_RESULTS/artifacts`: `artifacts/`.
- `$ROOT/05_PROTOCOLS_AND_MANIFESTS`: `protocols/` and `manifests/`.
- Manuscript: sibling directory
  `../semantic_novelty_audit_paper/tmlr/`.

This directory is not in a Git worktree, so per-task commits are not
available.

## Environment

- Python: 3.11.1 from the project `.venv/bin/python`.
- Added environment dependencies: SciPy 1.17.1 and pytest 9.1.1.
- Existing dependencies used: NumPy 2.4.6, matplotlib 3.11.1, and
  PyTorch 2.12.1.
- `scipy.optimize.milp` is available.
- MPS is available for Task 6.

## Task 2: CLUTRR V4 Effect Heterogeneity

Status: complete.

All required inputs were present. The optional V4 `common_base.jsonl` was
also present and its relation-frequency covariate was computed.

| Anchor | Expected | Actual | Pass |
|---|---:|---:|:---:|
| Mean effect (pp) | 23.339 | 23.3391 | yes |
| Median effect (pp) | 19.4 | 19.4063 | yes |
| Sample SD (pp) | 19.8 | 19.7703 | yes |
| Positive patterns | 76/80 | 76/80 | yes |
| Mean unexposed accuracy (%) | 61.23 | 61.2297 | yes |
| Spearman(effect, baseline) | -0.847 | -0.84748 | yes |
| Relation eta-squared | 0.494 | 0.49389 | yes |
| Leave-one-relation range (pp) | [20.914, 24.644] | [20.9141, 24.6444] | yes |
| Decoded semantic IDs | 80/80 | 80/80 | yes |
| Block means (pp) | [25.0109, 21.6672] | [25.0109, 21.6672] | yes |

Output:
`analysis/tmlr-strengthening-v1/t2_clutrr_heterogeneity/`.

Test:
`tests/test_tmlr_clutrr_heterogeneity.py` passed.

## Task 3(a): Internal Function Heterogeneity

Status: complete. Task 3(b) was intentionally not run because it is
outside the approved reduced scope, not because checkpoints are absent.

| Anchor | Expected | Actual | Pass |
|---|---:|---:|:---:|
| Mean effect (pp) | 1.832 | 1.8320 | yes |
| Sample SD (pp) | 0.970 | 0.97012 | yes |
| Median effect (pp) | 1.80 | 1.8000 | yes |
| Effect range (pp) | [0.133, 4.48] | [0.1333, 4.4800] | yes |
| Positive functions | 40/40 | 40/40 | yes |
| Spearman(effect, baseline) | -0.727 | -0.72740 | yes |
| Mean baseline | 0.7679 | 0.76794 | yes |
| Median normalized gain | 0.0768 | 0.07677 | yes |

Output:
`analysis/tmlr-strengthening-v1/t3_internal_heterogeneity/`.

Test:
`tests/test_tmlr_internal_heterogeneity.py` passed.

## Task 10: Novelty-Conditioned Score Bounds

Status: complete.

Revised on 2026-10-21 under the parameterized strengthening specification.
The original model-score application was withdrawn because the released SLR prediction
artifact does not reproduce the reported aggregate and is not tied to a
reproducible checkpoint. The current analysis uses only structural counts
from `artifacts/slr_overlap_results.json`: 1,000 test items minus 452
witnessed overlap items leaves M=548 unresolved items.

Correct fraction p is an unspecified parameter. The bounds are
max(0,p*M-U)/(M-U) and min(p*M,M-U)/(M-U), with p*M integral for finite
binary endpoint attainment. The figure shows only maximum interval width
min(1,U/(M-U)), attained at k=M/2 for this even M. No prediction outcomes,
partial scores, observed accuracy, or benchmark accuracy reference line
are used. All integer U from 0 to M-1 are evaluated.

The related no-upper-bound statement is corrected: for fixed observed
scores and a nonempty novel subset, the range is [min(y),max(y)]. If all
items may be exposed, a novel mean may be undefined.

Output:
`analysis/tmlr-strengthening-v1/t10_novelty_bounds/`.

Test:
`tests/test_tmlr_novelty_bounds.py` passed, including 300 exhaustive
enumeration cases.

## Task 9: Overlapping Certificates, Reduced

Status: complete for the approved reduced scope.

| Check | Expected | Actual | Pass |
|---|---:|---:|:---:|
| MILP versus exhaustive cases | 100 | 100 | yes |
| Independently verified LP dual cases | 100 | 100 | yes |
| Odd-cycle MILP feasible | false | false | yes |
| Odd-cycle LP interval | [1.5, 1.5] | [1.5, 1.5] | yes |
| Existing disjoint-bound tests | pass | pass | yes |

Output:
`analysis/tmlr-strengthening-v1/t9_overlapping_certificates/`.

Test:
`tests/test_tmlr_overlapping_bounds.py` passed.

## Task 6: Semantic Versus Surface Exposure

Status: complete.

All required inputs are present. Deterministic allocation produced 40/40
tier-2 patterns, each with 48 surface-similar decoy rows; no fallback or
main-analysis exclusion was needed. All three arms contain 24,640 unique
surface rows and have identical label counts. The shared evaluation has
8,000 rows and exactly matches the frozen V4 block-0 evaluation IDs.

The protocol was frozen before training at:
`protocols/tmlr-strengthening-v1/t6_semantic_vs_surface/protocol.json`.

Protocol SHA256:
`0fb766ba884fa82057b02613d8438f64f575f4ff5698765ceb250483585b6209`.

The one-epoch C0/seed-0 timing run on MPS took 32.74 seconds including
evaluation. All 12 formal runs completed 30 epochs and 5,790 updates with
24,640 training rows and 8,000 shared evaluation rows. Within each seed,
all three arms have the same initial model hash.

| Contrast | Effect (pp) | Pattern-cluster 95% CI | Seed 95% t CI |
|---|---:|---:|---:|
| C1 - C0 | 22.475 | [16.827, 28.743] | [19.042, 25.908] |
| C2 - C0 | 3.709 | [1.321, 5.956] | [-0.028, 7.447] |
| C1 - C2 | 18.766 | [13.202, 25.254] | [13.047, 24.484] |

Mean arm accuracies are 0.6028 (C0), 0.8276 (C1), and 0.6399 (C2).
The pattern interval for the decoy contrast is positive, while its
four-seed t interval includes zero. The direct C1-C2 contrast is positive
under both intervals. The prespecified interpretation is therefore:
"semantic identity, not surface similarity, drives the gain," with the
smaller positive decoy estimate reported rather than suppressed.

Output:
`analysis/tmlr-strengthening-v1/t6_semantic_vs_surface/`.

Test:
`tests/test_tmlr_clutrr_data.py` passed.

## Manuscript Integration

- Main text: novelty-conditioned proposition and SLR application in
  `Certified Exposure Interval`.
- Main text: CLUTRR and internal heterogeneity sentences, Task 6 result
  sentence, and the 80-pattern CLUTRR figure in
  `Why Exposure Effects Differ Across Settings`.
- Appendix: `Exploratory Function-Level Heterogeneity`,
  `Exploratory Effect Heterogeneity`,
  `Exploratory Semantic Versus Surface Exposure`,
  `Novelty-Conditioned Score Interval`, and `Overlapping Certificates`.
- Main-text figures: exactly three.
- Introduction through Conclusion ends on the same page as the pre-change
  baseline; body-page increase is zero. Total PDF length changed from 32
  to 37 pages because the new detailed material is in the appendix.

## Final Verification

- Full tests: 88 passed.
- Clean LaTeX build: passed, 37 pages.
- Undefined references/citations: none.
- Overfull boxes: none.
- Non-font/non-underfull warnings: none.
- Forbidden-term scan: empty.
- Acronym findings: none. The checker emits only path-resolution warnings
  for nested generated macro includes; the clean TeX build reads them.
- Anonymous-path scan: empty.
- All five task `SHA256SUMS` files verify.
- Current PDF hashes are recorded in the synchronized release manifests.
- Task 10 revision verification: all 548 admissible U values compared
  against all 549 binary correct counts; maximum-width formula agrees,
  is monotone, and stays in [0,1]. Existing novelty-bound tests: 5 passed.
- Revised LaTeX build: 37 pages, no warnings or unresolved references.
