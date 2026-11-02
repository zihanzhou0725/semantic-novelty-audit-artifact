# External Behavioral Validation v3

This protocol repairs the CLUTRR V1 surface-capacity defect without changing
the benchmark candidate, semantic identity, model, target count, evaluation
budget, exposure dose, or behavioral estimand.

V3 binds the 34,145 validated V1 source surfaces and deterministically
enumerates all six orderings of each three-sentence clean story. Sentence order
is already an invariance of the registered semantic identity. The six rows
share a source-group ID.

Evaluation and exposure allocations are source-group disjoint. A target is
eligible only with at least 45 complete six-permutation groups. Evaluation uses
34 groups for 200 rows; exposure uses the next 11 groups for 64 rows. This
prevents permutations of the same sentence multiset from crossing between
evaluation and training.

The design gate materializes candidates, targets, pairs, evaluation, exposure,
common base, and all 16 arm manifests. It does not start model training. A
failed gate preserves V1 No-Go and cannot trigger candidate replacement.
