# CLUTRR V3/V4 Paper-Ready Evidence Bundle

This anonymous, lightweight bundle supports the CLUTRR external behavioral
validation reported in the TMLR manuscript. It contains frozen protocols,
validation reports, run-artifact hashes, bootstrap draws, paper-facing tables,
and exact analysis-code snapshots.

## Main Confirmatory Result

- V4 exposed-minus-unexposed accuracy: `23.3391 pp`
- Hierarchical bootstrap 95% CI:
  `[17.7498, 29.2257] pp`
- Registered decision: `directional_positive_go`
- Descriptive consistency: 2/2 blocks, 8/8 seeds, 40/40 pairs positive

V3 is retained as a preregistered pilot and is not pooled with V4. Its effect
was positive, but its frozen 2 pp precision Gate was infeasible and therefore
returned `imprecise_no_go`.

## Scope

The intervention covers CLUTRR Task 1, relation length 3, clean
three-sentence stories, one frozen byte Transformer, and the registered
training budget. It measures transfer from semantic-pattern exposure to
source-group-disjoint surface realizations. It does not identify an internal
memorization mechanism.

## Bundle Layout

- `SUMMARY.json`: machine-readable paper claims and boundaries.
- `paper/`: publication table and claim-evidence matrix.
- `protocols/`: frozen preregistrations and Gate reports.
- `design/`: V3 variance diagnosis and V4 power/block design.
- `execution/`: post-training integrity and artifact hashes.
- `results/`: exact V3/V4 estimates, bootstrap draws, and subgroup tables.
- `code/`: exact analysis/design code snapshots.
- `SOURCE_FILES.json`: source-to-bundle provenance and hashes.
- `SHA256SUMS`: checksums for every bundle file except itself.
- `verify_bundle.py`: dependency-free integrity and anonymity check.

Raw predictions, checkpoints, training logs, and machine environments are
intentionally excluded. Their frozen SHA-256 commitments remain in the
run-artifact manifests.

Run `python3 verify_bundle.py` from this directory to validate the bundle.
