# External Behavioral Validation v3 Effect Analysis v1

This protocol unblinds the 16 completed CLUTRR V3 runs only after binding the
analysis implementation and every prediction input by SHA-256.

The primary estimand is the equal-seed, equal-target, row-weighted accuracy
difference between the arm exposed to a target semantic pattern and its paired
unexposed arm. Exposure is aligned separately for every model seed and target.
The point estimate therefore contains 8 x 40 paired effect units.

The 95% interval uses 20,000 hierarchical percentile-bootstrap replicates.
Each replicate resamples eight model seeds, twenty complete target-pattern
pairs, and 34 source groups within each selected pair member. Source-group
draws are shared across selected seeds and between exposed/unexposed
predictions, while the two pair members receive independent group draws.

The frozen precision, directional, and equivalence rules are inherited from
the parent V3 protocol. Cross-entropy, Brier score, calibration, seed,
pattern, pair, and relation summaries are descriptive and cannot change the
primary Gate.
