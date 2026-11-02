# Directional External Behavioral Validation V4 Execution v1

This protocol freezes the implementation and randomization for 32 V4 runs. It
does not start training or compute the directional exposure effect.

Each model seed has four isolated runs: two blocks by two complementary arms.
All four runs share the same parameter initialization, thirty epoch
permutations, fixed tensor shape, and Torch RNG trajectory. Training rows
differ only through the frozen block-specific exposure manifests.

The execution bundle contains one shared tensor cache, 32 train-index arrays,
two block-specific evaluation caches, eight initialization records, and 240
epoch-order records. Every run uses deterministic float32 CPU training for 30
epochs and 5,640 optimizer updates.

At most eight runs may execute concurrently. The directional effect remains
sealed until all 32 runs pass post-training integrity and a separate analysis
protocol binds the resulting prediction hashes.
