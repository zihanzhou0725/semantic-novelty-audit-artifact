# Submission supplement preparation — 22 October 2026

The manuscript, audit code, frozen protocols, and published numerical results
are unchanged from the anonymous repository snapshot. Python bytecode caches
were excluded, and the root `ANONYMIZED_SHA256SUMS` was regenerated for this
upload. The original protocol checksum manifest is unchanged.

## Optional inputs for the complete test suite

The paper-facing analysis and PDF-reproduction commands in `README.md` run
from the compact evidence already present in this package. To also run all
88 tests, restore the additional frozen inputs:

```bash
python3 prepare_test_inputs.py
cd external_semantic_overlap_audit
python3 -m pytest -q tests
```

The restoration uses only Python's standard library. The complete test suite
also requires NumPy, SciPy, matplotlib, PyTorch, and pytest. Restoration adds
approximately 299 MB of generated CLUTRR data and public FB15k-237 snapshots
to the extracted directory; allow additional disk space.

`OPTIONAL_TEST_INPUTS.tar.xz` contains ten files that were omitted from the
compact repository. Each restored file must match its original frozen hash
in `OPTIONAL_TEST_INPUTS_SHA256SUMS`. The CLUTRR hashes were taken from the
existing materialization manifests, and the FB15k-237 hashes were taken from
the existing audit report. Archive owner names were removed.

All 88 tests and both paper-reproduction targets were checked on an independent
copy. The preparation does not retrain models or include model checkpoints.

For the full T2, T3(a), T6, T9, and T10 analysis rebuild, run from the extracted
package root:

```bash
make -C semantic_novelty_audit_paper/tmlr reproduce-paper
```

The anonymous repository URL remains:
https://anonymous.4open.science/r/semantic-novelty-audit-artifact-4920/

This upload adds the optional test-input archive and its restoration helper.
The online mirror may lack these additional files until it is updated.
