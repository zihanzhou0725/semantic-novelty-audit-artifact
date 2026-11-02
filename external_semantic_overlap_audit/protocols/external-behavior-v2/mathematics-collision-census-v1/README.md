# Mathematics Collision Census v1

This protocol releases the previously sealed train/interpolate overlap
aggregate for the two validated `external-behavior-v2` polynomial streams.

The primary denominator is the retained interpolate row. Distinct
interpolate semantic IDs are reported separately so repeated evaluation
semantics cannot silently change the estimand. `polynomials__expand` and
`polynomials__collect` are always matched and reported separately.

Verbatim overlap uses exact question-string equality. Semantic overlap uses
the frozen exact-rational, alpha-normalized polynomial identity. Every
semantically overlapping interpolate row receives the smallest-row-index
train witness and must pass an independently implemented symbolic
zero-difference check.

The released result applies only to regenerated streams under the pinned
source revision and seeds. It is not a behavioral effect and does not alter
any prior CLUTRR or suite decision.
