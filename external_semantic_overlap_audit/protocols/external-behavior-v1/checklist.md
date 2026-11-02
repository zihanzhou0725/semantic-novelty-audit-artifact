# External Behavioral Validation Checklist

## Frozen Before Results

- [x] Prior P3 decision remains unchanged.
- [x] Availability-only selection criteria declared.
- [x] Candidate set and audit order frozen.
- [x] Candidate freeze receipt records no prior clone, download, extraction,
  census, training, or outcome inspection.
- [x] Official code revisions pinned.
- [x] Required source files hashed.
- [x] Semantic units declared before census.
- [x] CLUTRR target-selection rule declared before materialization.
- [x] CLUTRR paired arm construction declared before materialization.
- [x] Model architecture, optimizer, seeds, and final-checkpoint rule declared.
- [x] Primary estimand, bootstrap, SESOI, equivalence margin, and precision
  threshold declared.
- [x] Mathematics modules, budgets, parsers, and exact verifier declared.
- [x] Result-independent suite gate declared.

## Materialization Gate

- [ ] Freeze adapter and model implementation hashes.
- [ ] Freeze Python and dependency environment.
- [ ] Verify pinned upstream source hashes.
- [ ] Generate exactly 120,000 CLUTRR candidate rows or log generator
  exhaustion.
- [ ] Reconstruct every CLUTRR semantic signature from graph fields.
- [ ] Verify at least 20 eligible same-label semantic-pattern pairs.
- [ ] Freeze exactly 40 target patterns and 20 pairs.
- [ ] Freeze 8,000 evaluation rows.
- [ ] Freeze 64 exposure rows per target pattern.
- [ ] Freeze 22,720 common base rows with zero target signatures.
- [ ] Verify A/B size and label-count equality for all eight seeds.
- [ ] Verify evaluation/training surface disjointness.
- [ ] Verify no sequence truncation at 512 bytes.
- [ ] Freeze all CLUTRR manifests and SHA-256 values before training.
- [ ] Generate 220,000 retained Mathematics examples.
- [ ] Verify 100% parser coverage.
- [ ] Verify deterministic signature regeneration.
- [ ] Freeze Mathematics row manifests and SHA-256 values before aggregate
  overlap release.

## Execution Gate

- [ ] Complete 16/16 CLUTRR runs at epoch 30.
- [ ] Validate finite model, optimizer, and prediction tensors.
- [ ] Validate paired initialization and epoch order.
- [ ] Validate 8,000 identically ordered predictions per run.
- [ ] Validate observed arm exposure against the frozen assignment.
- [ ] Independently recompute every Mathematics collision witness.
- [ ] Independently recompute the CLUTRR primary effect and bootstrap interval.
- [ ] Validate all artifact hashes.

## Reporting Gate

- [ ] Report positive, negative, null, or inconclusive effect classification.
- [ ] Report the primary 95% interval width.
- [ ] Report every Mathematics module and unresolved count.
- [ ] Report all failed design or integrity conditions.
- [ ] Report both candidates without result-dependent exclusion.
- [ ] State the final external-behavior suite decision.
- [ ] Preserve forbidden claims and the original P3 No-Go.
