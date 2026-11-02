# CLUTRR V3 Variance Decomposition and V4 Design Gate

This protocol is frozen after the V3 effect was unblinded. It is exploratory:
it may diagnose V3 uncertainty and inform a future protocol, but it cannot
change the registered V3 decision or support a confirmatory claim.

The analysis evaluates all eight combinations of resampling model seeds,
pattern pairs, and source groups. Shapley attribution makes the resulting
bootstrap variances additive while averaging over factor-entry order.

A separate two-way random-effects decomposition of the model-seed by
pattern-pair matrix supports prospective precision projections. Projected
normal widths are calibrated to recover the actual V3 hierarchical percentile
width at eight seeds and twenty pairs.

The strict V4 design gate asks only whether semantic patterns unused by V3 can
support the original two-percentage-point precision requirement. A No-Go does
not rule out a newly preregistered estimand or directional design; it prevents
releasing training under an infeasible replica of the V3 precision Gate.
