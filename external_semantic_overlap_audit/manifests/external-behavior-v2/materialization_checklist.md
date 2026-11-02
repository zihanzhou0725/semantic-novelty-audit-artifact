# External Behavioral Validation v2 Materialization Gate

- [x] Freeze V2 implementation hashes.
- [x] Freeze Python and dependencies.
- [x] Verify pinned source revision and hashes.
- [x] Generate all 220,000 retained rows.
- [x] Validate every anomaly against the registered assertion.
- [x] Keep assertion drops below the per-stream cap.
- [x] Match all three V1-complete streams byte-for-byte.
- [x] Match the V1 train/expand 67,937-row prefix.
- [x] Independently regenerate all semantic IDs.
- [x] Verify contiguous row indices and row IDs.
- [x] Freeze and validate all artifact hashes.
- [x] Keep overlap join and aggregate release sealed.
