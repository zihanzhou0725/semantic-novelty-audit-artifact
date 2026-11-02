# External Behavioral Validation v3 Execution Checklist

## Frozen Before Training

- [x] Design gate and checksum hashes bound.
- [x] Initialization namespace and hash algorithm declared.
- [x] Epoch-order namespace and PCG64 algorithm declared.
- [x] Fixed padding length and batch sizes declared.
- [x] Optimizer, epochs, update count, and final checkpoint declared.
- [x] CPU thread and process concurrency declared.
- [x] Run artifact schema and integrity gate declared.
- [x] Primary paired effect remains sealed.
- [x] No behavioral run has started.

## Execution Freeze

- [ ] Freeze training, driver, and integrity-validator hashes.
- [ ] Freeze environment and deterministic-runtime settings.
- [ ] Freeze 16-row run manifest.
- [ ] Freeze eight initialization hashes.
- [ ] Freeze 240 epoch-order hashes.
- [ ] Pass synthetic optimizer and artifact canaries.

## Integrity Gate

- [ ] Complete 16/16 runs at epoch 30.
- [ ] Verify 5,640 optimizer updates per run.
- [ ] Verify finite model, optimizer, history, logits, and metrics.
- [ ] Verify paired initialization and epoch-order hashes.
- [ ] Verify paired initial and final Torch RNG hashes.
- [ ] Verify 8,000 ordered predictions per run.
- [ ] Verify observed row IDs and complementary target exposure.
- [ ] Validate every run artifact checksum.
- [ ] Confirm no primary paired aggregate was computed.
