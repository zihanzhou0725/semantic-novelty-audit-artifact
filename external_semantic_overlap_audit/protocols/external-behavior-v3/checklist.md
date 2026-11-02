# External Behavioral Validation v3 Checklist

## Frozen Before Expansion

- [x] Preserve V1 CLUTRR and suite No-Go decisions.
- [x] Preserve candidate, source revision, semantic ID, model, and budgets.
- [x] Freeze the 34,145-row V1 source manifest and hash.
- [x] Freeze the exact three-sentence parser.
- [x] Freeze all six sentence permutations and row ordering.
- [x] Freeze source-group identity and cross-split disjointness.
- [x] Freeze 45-group target eligibility.
- [x] Freeze 200-row evaluation and 64-row exposure allocation.
- [x] Freeze same-label pairing and arm assignment.
- [x] Freeze source-group-aware bootstrap.
- [x] Confirm no CLUTRR behavioral outcome exists.

## Implementation Freeze

- [ ] Freeze V3 adapter and validator hashes.
- [ ] Freeze Python and dependency environment.
- [ ] Pass parser, permutation, allocation, and arm-balance tests.
- [ ] Verify all V1 input and upstream hashes.
- [ ] Inspect the frozen byte Transformer architecture.

## Design Gate

- [ ] Expand exactly 204,870 pre-dedup permutation rows.
- [ ] Verify all source rows parse into exactly three sentences.
- [ ] Verify every complete group has six unique permutations.
- [ ] Verify at least 20 eligible same-label pairs.
- [ ] Freeze exactly 40 targets and 20 pairs.
- [ ] Freeze 8,000 evaluation rows from 1,360 source groups.
- [ ] Freeze 2,560 exposure rows from 440 source groups.
- [ ] Verify evaluation/exposure source-group disjointness.
- [ ] Freeze 22,720 non-target common-base rows.
- [ ] Verify 16 arm manifests contain 24,000 rows each.
- [ ] Verify paired label counts and complementary exposure.
- [ ] Verify all allocated sequences fit 512 bytes.
- [ ] Freeze and validate all design hashes.
- [ ] Confirm model training has not started.
