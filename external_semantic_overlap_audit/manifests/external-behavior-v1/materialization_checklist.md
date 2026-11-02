# External Behavioral Validation Materialization Gate

This report supplements the immutable preregistration checklist. Failed items
remain failed; no candidate, semantic scope, or threshold was replaced.

- [x] Freeze adapter and model implementation hashes.
- [x] Freeze Python and dependency environment.
- [x] Verify pinned upstream source hashes.
- [x] Generate exactly 120,000 CLUTRR candidate rows.
- [x] Independently reconstruct all 120,000 CLUTRR signatures.
- [ ] Verify at least 20 eligible same-label semantic-pattern pairs.
- [ ] Freeze exactly 40 target patterns and 20 pairs.
- [ ] Freeze 8,000 evaluation rows.
- [ ] Freeze 64 exposure rows per target pattern.
- [x] Freeze 22,720 common base rows with zero target signatures.
- [ ] Verify A/B size and label-count equality for all eight seeds.
- [ ] Verify evaluation/training surface disjointness for the absent targets.
- [x] Verify allocated inputs fit within 512 bytes.
- [x] Freeze partial CLUTRR manifests and SHA-256 values.
- [ ] Generate all 220,000 retained Mathematics examples.
- [x] Independently regenerate all 187,937 materialized Math signatures.
- [ ] Complete parser/signature validation for 32,063 unmaterialized rows.
- [x] Preserve Math aggregate overlap blindness.
- [x] Confirm that behavioral training never started.

Gate decision: **No-Go**.

The CLUTRR design has zero eligible patterns after registered surface
deduplication. Mathematics generation stopped at retained
`train/polynomials__expand` row index 67,937 on a pinned-upstream assertion.
The frozen v1 implementation was not changed or resumed.
