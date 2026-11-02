# Directional V4 Execution Freeze Checklist

- [x] Require the materialization Gate to pass.
- [x] Bind all 32 training and two evaluation manifests.
- [x] Freeze one shared fixed-length tensor cache.
- [x] Freeze 32 train-index arrays.
- [x] Freeze eight model initialization hashes.
- [x] Freeze 240 epoch-order hashes.
- [x] Require four-way paired randomness within each seed.
- [x] Freeze optimizer, update count, and final checkpoint.
- [x] Freeze CPU float32 deterministic execution.
- [x] Freeze maximum concurrency and per-process threads.
- [x] Seal the directional effect for a later analysis stage.
- [x] Confirm no execution artifacts or training directory exist.
