# External Behavioral Validation v2 Checklist

## Protocol Freeze

- [x] Preserve V1 CLUTRR and suite No-Go decisions.
- [x] Preserve the original candidate, source revision, modules, budgets,
  seeds, parsers, and semantic identity.
- [x] Record all V1 evidence available before amendment freeze.
- [x] Confirm that no Mathematics overlap aggregate was available.
- [x] Register one exact recoverable assertion location.
- [x] Register the anomaly log and per-stream assertion-drop cap.
- [x] Require all four streams to restart from row zero.
- [x] Require byte identity for three complete streams and the train/expand
  prefix.
- [x] Keep overlap aggregate release outside this stage.

## Implementation Freeze

- [ ] Freeze V2 adapter and validator hashes.
- [ ] Freeze Python and dependency environment.
- [ ] Verify pinned upstream source hashes and clean checkout.
- [ ] Pass synthetic assertion-recovery tests.
- [ ] Pass a no-aggregate materialization canary.

## Materialization Gate

- [ ] Generate all 220,000 retained rows.
- [ ] Log every recoverable assertion with the registered traceback fields.
- [ ] Verify every logged assertion matches the exact registered location.
- [ ] Verify no stream exceeds 1,000 assertion drops.
- [ ] Verify three V1-complete streams are byte-identical.
- [ ] Verify the 67,937-row train/expand prefix is byte-identical.
- [ ] Verify 100% parser coverage.
- [ ] Independently regenerate all 220,000 semantic IDs.
- [ ] Verify contiguous row indices within every stream.
- [ ] Freeze all row, anomaly, implementation, and environment hashes.
- [ ] Confirm no overlap join or aggregate release occurred.
