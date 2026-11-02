# External Behavioral Validation v1

This directory freezes an independent prospective expansion for the TMLR
version of the semantic-novelty audit. It does not amend or reclassify the
`quality-enhancement-v1` P3 No-Go result.

## Registered Candidates

1. CLUTRR at revision `d045fae289d3746503677ceed7631c999202501e`
2. DeepMind Mathematics Dataset at revision
   `427f45075f84b8b9774950196ad63867ca20ffb3`

The candidate set was frozen before either repository was cloned, before data
were generated or downloaded, and before any semantic-overlap result was
computed. The receipt is `candidate_freeze_receipt.json`.

## Primary Behavioral Design

CLUTRR Task 1 at relation length 3 supplies the public behavioral test. Forty
entity-invariant ordered kinship patterns are paired within target relation.
For every model seed, one member of each pair is exposed in Arm A and the other
in Arm B. The assignment is complementary, while common training rows,
initialization, slot order, optimizer settings, and all 8,000 evaluation rows
are paired.

The primary estimand is exposed-minus-unexposed target-pattern accuracy,
averaged equally over 40 patterns and eight model seeds. Sixteen final-epoch
runs are required. Positive, negative, null-equivalent, and inconclusive
outcomes are all reportable; effect direction is not a suite acceptance
condition.

## Independent Semantic Audit

The Mathematics candidate audits regenerated train and interpolation streams
for:

- `polynomials__expand`
- `polynomials__collect`

The semantic signature is the exact rational-coefficient polynomial after
alpha-normalization and expansion. Every collision must pass a second symbolic
equivalence verifier.

## Freeze Boundary

The following actions are forbidden before the protocol and implementation
manifests validate:

- generating CLUTRR or Mathematics rows;
- computing train/test semantic overlap;
- selecting targets using observed model outcomes;
- starting any behavioral training;
- changing a candidate because its result is null, negative, incomplete, or
  inconvenient.

Any substantive change listed in `preregistration.json` requires a new protocol
version.

## Validation

Run from the project root:

```bash
python3 protocols/external-behavior-v1/validate.py
```

At this stage the validator checks the frozen documents, candidate receipt,
source revisions and hashes, internal arithmetic invariants, and absence of
result artifacts. Materialization and training add separate manifests without
modifying the frozen files in this directory.
