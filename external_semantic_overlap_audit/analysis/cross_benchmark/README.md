# Cross-Benchmark Summary And Figures

All values are generated from structured result artifacts by:

```bash
cd <SOURCE_WORKSPACE>/external_semantic_overlap_audit
MPLCONFIGDIR=<SOURCE_WORKSPACE>/paper_sources/.mplconfig \
  <SOURCE_WORKSPACE>/.venv/bin/python \
  scripts/build_cross_benchmark_figures.py
```

## Unified Summary

`cross_benchmark_summary.csv` contains five rows:

1. Internal symbolic benchmark matched five-seed public audit.
2. SLR-Bench sound semantic-overlap lower bound.
3. Boolean ICL Conjunction-28.
4. Boolean ICL Majority-28.
5. Boolean ICL released Sparse Disjunction-50.

The `overlap_or_exposure_rate` column preserves each case's stated
denominator and evidence type:

- internal: held-out parameter identities duplicating a training function;
- SLR: train-test semantic-overlap lower bound;
- Boolean: frozen evaluation functions actually sampled during training.

These rates are related audit quantities, not interchangeable estimands.

## Figure 1

Files:

- `fig1_semantic_identity_framework.pdf`
- `fig1_semantic_identity_framework.png`

Caption:

> Three levels of novelty in procedurally generated benchmarks. Distinct
> generator parameters or rule strings can map to the same complete
> input-output function, while multiple functions can belong to one rule
> family. Parameter-, function-, and family-disjoint evaluation therefore test
> different generalization claims.

## Figure 2

Files:

- `fig2_slr_semantic_overlap.pdf`
- `fig2_slr_semantic_overlap.png`

Caption:

> SLR-Bench `v1-All` has zero verbatim train-test rule overlap, but 70 test
> rules are Prolog-token identical to training rules and another 382 flat
> positive rules are provably semantically equivalent. Their disjoint union
> gives a sound semantic-overlap lower bound of 452/1,000 = 45.2%; unresolved
> complex rules are not counted.

## Figure 3

Files:

- `fig3_boolean_exposure_effect.pdf`
- `fig3_boolean_exposure_effect.png`

Caption:

> Exact evaluation-function exposure spans more than two orders of magnitude
> across the released Boolean ICL tasks, but removing that exposure has
> near-zero effect on tail accuracy for the tested Transformer. Points are
> three-seed Original-minus-Strict means; intervals are preregistered
> hierarchical 95% bootstrap intervals. The +2 pp Go threshold lies outside
> the displayed range.

## Paper Copies

The generated PDFs are copied without modification to:

- `paper_sources/figures/fig_semantic_identity_framework.pdf`
- `paper_sources/figures/fig_slr_semantic_overlap.pdf`
- `paper_sources/figures/fig_boolean_exposure_effect.pdf`
