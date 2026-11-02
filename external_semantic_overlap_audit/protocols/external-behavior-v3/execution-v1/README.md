# External Behavioral Validation v3 Execution v1

This stage executes the 16 frozen CLUTRR V3 runs. It does not change the
design, select checkpoints, or compute the paired exposure effect.

Each model seed has two isolated CPU processes, one per arm. A/B runs share
the same frozen parameter initialization, 30 epoch permutations, fixed tensor
shape, and Torch RNG trajectory. Training rows differ only according to the
frozen complementary exposure manifests.

Eight runs may execute concurrently, each with two PyTorch intra-op threads.
All runs use float32 deterministic CPU algorithms. Every run must reach epoch
30 and optimizer update 5,640 before its `COMPLETED.json` is accepted.

Primary effect estimation remains sealed for a later analysis stage with
separately frozen input hashes.
