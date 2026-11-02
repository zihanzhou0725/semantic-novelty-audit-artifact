# Verified Environment

## Software

- Python: 3.11.1
- PyTorch: 2.12.1
- torchvision: 0.27.1
- NumPy: 2.4.6
- tqdm: 4.68.3
- mpmath: 1.3.0
- matplotlib: 3.11.1
- TeX Live / latexmk: latexmk 4.88

Exact versions are pinned in `src/kfwa_experiments/requirements.txt`.

## Fresh-Install Verification

On 2026-10-07, the final ZIP was extracted into a new temporary directory and
installed into a Python 3.11.1 virtual environment with
`include-system-site-packages = false`. `pip check` reported no broken
requirements, and `python reproduce.py all` completed the compact-result
validation, protocol/oracle/model validators, CPU smoke training, figure
generation, and both LaTeX builds.

## Verification Platform

- Operating system: macOS 26.6.2
- Architecture: arm64
- Processor: Apple M5 Pro
- Unified memory: 48 GiB

The code supports CPU, CUDA, and Apple MPS through `--device`. Exact Bayesian
and ML predictors are evaluated on CPU. The released compact archive was
validated on the platform above; the tiny reproduction smoke test defaults to
CPU so it does not require an accelerator.

## Randomness and Determinism

- Python, NumPy, and PyTorch seeds are set for every training run.
- Function splits are deterministic and recorded by complete-map checksums.
- Every symbolic dataset item uses a seed derived from its split seed and item
  index, so episodes can be regenerated independently.
- Formal paired comparisons share function splits and episodes within seed.
- Exact predictors and compact-result aggregation are deterministic on CPU.
- Neural training can exhibit small backend-dependent floating-point
  differences, especially on CUDA or MPS. Stored result manifests and
  checkpoint hashes identify the original runs; conclusions are based on
  paired multi-seed effects rather than bitwise checkpoint equality.

## Resource Scale

The `reproduce.py validate`, `figures`, and `paper` targets require no model
training or checkpoints. The `smoke` target trains two small models for one
epoch on 128 synthetic episodes and is intended only to verify the end-to-end
pipeline. Reproducing formal five-seed neural results requires the commands
listed in `README.md` and substantially more compute.
