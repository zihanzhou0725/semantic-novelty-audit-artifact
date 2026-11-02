# Directional V4 Effect Analysis v1

This protocol unblinds the 32 completed V4 runs only after binding the analysis
implementation and every prediction artifact by SHA-256.

The primary estimand equally weights eight model seeds and eighty frozen target
semantic patterns. For each seed and target, the arm exposed to that pattern is
compared with its complementary unexposed arm on the same 200 evaluation rows.

The 95% interval uses 20,000 hierarchical percentile-bootstrap replicates.
Each replicate resamples eight seeds, forty complete global pattern pairs, and
34 source groups within each selected pair member. Pair selection preserves the
frozen block identity.

The only confirmatory decision is directional: the point estimate must be at
least one percentage point and the two-sided 95% interval lower endpoint must
be above zero. Block, seed, pattern, relation, loss, and calibration summaries
are descriptive and cannot modify this Gate.
