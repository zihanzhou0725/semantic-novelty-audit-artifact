# Mathematics Collision Census v1 Checklist

## Frozen Before Join

- [x] Parent materialization gate passed.
- [x] Four row-manifest hashes and row counts frozen.
- [x] Module-local verbatim and semantic identities declared.
- [x] Row-weighted and unique-semantic denominators declared separately.
- [x] Representative witness selection declared.
- [x] Multiplicity distributions declared.
- [x] Independent symbolic zero-difference verifier declared.
- [x] Output ordering and serialization declared.
- [x] Complete-census and claim gates declared.
- [x] No overlap count or collision witness inspected.

## Implementation Gate

- [ ] Freeze census and independent-validator implementation hashes.
- [ ] Freeze environment hash.
- [ ] Pass synthetic exact, alias, no-match, multiplicity, and module-isolation
  tests.
- [ ] Verify all frozen input hashes before join.

## Census Gate

- [ ] Emit exactly 20,000 evaluation audit rows.
- [ ] Emit one witness for every semantically overlapping evaluation row.
- [ ] Verify every collision by exact symbolic zero difference.
- [ ] Verify minimum-row-index witness selection.
- [ ] Verify per-module and overall count conservation.
- [ ] Verify multiplicity distribution conservation.
- [ ] Independently recompute all released aggregates.
- [ ] Report unresolved count zero.
- [ ] Freeze output hashes and release aggregate.
