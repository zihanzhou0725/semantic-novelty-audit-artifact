# TMLR Manuscript

This directory contains the integrated, anonymous TMLR manuscript. Historical
non-TMLR sources remain outside the anonymous submission package.

## Build

```bash
make
```

The output is one submission PDF, `main.pdf`, with references followed by the
appendix.

Generated numerical macros are committed under `generated/`, and generated
paper figures are committed under `figures/`. Their source tables and analysis
scripts live in the sibling `external_semantic_overlap_audit` project.

## Recompute the strengthening results

```bash
make reproduce-paper PYTHON=/path/to/python
```

Use Python with `requirements-tmlr-strengthening.txt` installed and a TeX
installation providing `latexmk`. This recomputes T2/T3a/T6/T9/T10, copies
macros and figures into the manuscript, regenerates the overview, and builds
the PDF. It uses retained predictions, not model training. Historical figures
and results are retained as supplied artifacts.

## Narrative Boundary

- The original Tensor Logic / Occam breadth criterion remains unmet.
- CLUTRR V3 remains positive but imprecise.
- CLUTRR V4 is a separate, target-disjoint directional confirmation.
- The semantic-versus-surface intervention is exploratory and reports the
  smaller positive surface-decoy estimate alongside the larger semantic
  contrast.
- No intervention is presented as evidence of an internal memorization
  mechanism or as a cross-task universal effect.

See `TEMPLATE_SOURCE.md` for the pinned official style source.
