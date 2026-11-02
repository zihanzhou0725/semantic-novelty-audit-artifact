# Directional V4 Materialization v1

This stage converts the frozen 40-pair, two-block directional V4 design into
evaluation, exposure, common-base, and training-row manifests. It does not
train models.

Each block contains 20 pairs and 40 target patterns. Every target receives 200
evaluation rows from 34 source groups and 64 exposure rows from the next 11
source groups. Evaluation and exposure groups cannot overlap.

One 22,720-row common base is shared byte-for-byte across both blocks, all
seeds, and both arms. It excludes every V3 target and every selected V4 target.
Each run appends 1,280 exposure rows to produce exactly 24,000 training rows.

The materialization Gate reconstructs all allocations from the frozen V3
candidate census, validates all 32 run manifests, and confirms that no V4
training directory exists. A passing Gate permits a separate execution
protocol freeze; it does not itself release training.
