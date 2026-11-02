# Semantic-Novelty Audit Supplement

This package supports the manuscript:

**Unseen Strings, Familiar Tasks: Certified Semantic Exposure in
Procedural Reasoning Benchmarks**

It contains the paper source, compact numerical results, preregistrations,
validation reports, audit code, the portable semantic-audit toolkit, and
public upstream source snapshots required to inspect the reported evidence.
Model checkpoints, virtual environments, per-run logs, review history, local
paths, and large public datasets are excluded.

## Layout

- `semantic_novelty_audit_paper/tmlr/`: manuscript, integrated appendix,
  figures, bibliography, and TMLR build files.
- `external_semantic_overlap_audit/analysis/`: paper-facing numerical
  analyses, bootstrap arrays, and structured result tables.
- `external_semantic_overlap_audit/artifacts/`: SLR witnesses and the compact
  CLUTRR V3/V4 evidence bundle, plus compact per-item outputs for the
  semantic-versus-surface intervention.
- `external_semantic_overlap_audit/protocols/`: frozen preregistrations and
  validation reports.
- `external_semantic_overlap_audit/manifests/`: compact execution and
  materialization metadata.
- `external_semantic_overlap_audit/scripts/`, `toolkit/`, and `tests/`:
  analysis, validation, and schema code.
- `semantic_novelty_audit_supplement/`: reusable finite-rule benchmark code
  used by the internal intervention.

Venue identifiers were redacted from frozen protocol files; hashes of the
unredacted originals will be released after publication. The unredacted
private originals were not overwritten.

## Reproduce Paper-Facing Outputs

Prerequisites:

- Python 3.11;
- NumPy, SciPy, and matplotlib;
- a TeX Live installation with `latexmk`;
- PyTorch only for training-oriented validators, not for the paper-facing
  figure and PDF rebuild.

Run:

```bash
make reproduce-paper
```

The target regenerates the cross-benchmark summary CSV and three paper
figures from structured result artifacts, regenerates the semantic-identity
figure from its dedicated source, rebuilds the manuscript PDF, and runs the
portable-toolkit and CLUTRR bundle verifiers. Expected runtime is about
1--3 minutes on a recent CPU after dependencies and LaTeX fonts are installed.
It does not retrain the 126 neural runs or recreate excluded checkpoints.
The strengthening analysis directories contain generated CSV/JSON tables,
figures, macros, run metadata, and checksums. The Task 6 compact evidence
retains configs, histories, metrics, and per-item predictions for all 12
runs, but not model checkpoints or tensor caches.

For read-only evidence validation:

```bash
make verify
```

## Integrity

`ANONYMIZED_SHA256SUMS` lists every released file except itself.
`ANONYMIZED_PROTOCOL_SHA256SUMS` separately binds the venue-redacted
preregistration and design-manifest copies.

The public artifact repository is:
https://github.com/zihanzhou0725/semantic-novelty-audit-artifact

On Windows without GNU make, run
`semantic_novelty_audit_paper/tmlr/reproduce-paper.cmd` instead of
`make reproduce-paper`; it regenerates the figures, rebuilds the PDF, and
prints an overfull-box summary.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Upstream source snapshots under `external_semantic_overlap_audit/upstream/`
retain their original licenses and are included only for audit provenance.
