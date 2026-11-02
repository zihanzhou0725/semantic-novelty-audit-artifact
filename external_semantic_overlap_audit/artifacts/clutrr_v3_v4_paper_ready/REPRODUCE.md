# Reproduction Notes

## Lightweight Verification

```bash
python3 verify_bundle.py
```

This verifies every bundled byte, checks key paper values, rejects unlisted
files, and scans text artifacts for local absolute paths and workspace
identifiers.

## Statistical Audit

The exact 20,000-draw bootstrap arrays are included at:

- `results/v3/bootstrap.npz`
- `results/v4/bootstrap.npz`

The corresponding draw hashes, confidence endpoints, and decision objects are
recorded in each `results.json` and independently reproduced in each
`analysis_validation.json`.

## Full Training Reproduction

The lightweight bundle does not include checkpoints or raw predictions.
Full reproduction requires the separately archived training artifact whose
hashes are committed in:

- `execution/v3_run_artifact_hashes.json`
- `execution/v4_run_artifact_hashes.json`

The exact training and analysis implementations are pinned by the frozen
protocol input manifests. No paper value should be recomputed under a modified
protocol in place; changes require a new version.
