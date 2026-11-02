#!/usr/bin/env python3
"""Build a lightweight anonymous CLUTRR V3/V4 paper evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "clutrr_v3_v4_paper_ready"


COPY_FILES = {
    "protocols/v3_design_preregistration.json":
        "protocols/external-behavior-v3/preregistration.json",
    "protocols/v3_design_validation.json":
        "manifests/external-behavior-v3/design_validation_report.json",
    "protocols/v3_execution_preregistration.json":
        "protocols/external-behavior-v3/execution-v1/preregistration.json",
    "protocols/v3_execution_validation.json":
        "manifests/external-behavior-v3/execution-v1/"
        "execution_validation_report.json",
    "protocols/v3_effect_preregistration.json":
        "protocols/external-behavior-v3/effect-analysis-v1/"
        "preregistration.json",
    "protocols/v3_effect_freeze_validation.json":
        "protocols/external-behavior-v3/effect-analysis-v1/"
        "validation_report.json",
    "protocols/v3_variance_preregistration.json":
        "protocols/external-behavior-v3/variance-design-v1/"
        "preregistration.json",
    "protocols/v4_directional_design_preregistration.json":
        "protocols/external-behavior-v4/directional-design-v1/"
        "preregistration.json",
    "protocols/v4_materialization_preregistration.json":
        "protocols/external-behavior-v4/materialization-v1/"
        "preregistration.json",
    "protocols/v4_execution_preregistration.json":
        "protocols/external-behavior-v4/execution-v1/"
        "preregistration.json",
    "protocols/v4_execution_validation.json":
        "manifests/external-behavior-v4/execution-v1/"
        "execution_validation_report.json",
    "protocols/v4_effect_preregistration.json":
        "protocols/external-behavior-v4/effect-analysis-v1/"
        "preregistration.json",
    "protocols/v4_effect_freeze_validation.json":
        "protocols/external-behavior-v4/effect-analysis-v1/"
        "validation_report.json",
    "design/v3_variance_results.json":
        "analysis/external-behavior-v3/variance-design-v1/"
        "variance_results.json",
    "design/v3_to_v4_design_gate.json":
        "analysis/external-behavior-v3/variance-design-v1/"
        "v4_design_gate_report.json",
    "design/v4_power_results.json":
        "analysis/external-behavior-v4/directional-design-v1/"
        "power_results.json",
    "design/v4_block_gate.json":
        "analysis/external-behavior-v4/directional-design-v1/"
        "block_gate_report.json",
    "design/v4_design_manifest.json":
        "manifests/external-behavior-v4/directional-design-v1/"
        "design_manifest.json",
    "design/v4_selected_pairs.csv":
        "manifests/external-behavior-v4/directional-design-v1/"
        "selected_pairs.csv",
    "execution/v3_run_artifact_hashes.json":
        "manifests/external-behavior-v3/execution-v1/"
        "run_artifact_manifest.json",
    "execution/v4_run_artifact_hashes.json":
        "manifests/external-behavior-v4/execution-v1/"
        "run_artifact_manifest.json",
    "results/v3/results.json":
        "analysis/external-behavior-v3/effect-analysis-v1/results.json",
    "results/v3/analysis_validation.json":
        "analysis/external-behavior-v3/effect-analysis-v1/"
        "analysis_validation_report.json",
    "results/v3/bootstrap.npz":
        "analysis/external-behavior-v3/effect-analysis-v1/bootstrap.npz",
    "results/v3/seed_effects.csv":
        "analysis/external-behavior-v3/effect-analysis-v1/"
        "seed_effects.csv",
    "results/v3/pair_effects.csv":
        "analysis/external-behavior-v3/effect-analysis-v1/"
        "pair_effects.csv",
    "results/v3/pattern_effects.csv":
        "analysis/external-behavior-v3/effect-analysis-v1/"
        "pattern_effects.csv",
    "results/v3/relation_effects.csv":
        "analysis/external-behavior-v3/effect-analysis-v1/"
        "relation_effects.csv",
    "results/v4/results.json":
        "analysis/external-behavior-v4/effect-analysis-v1/results.json",
    "results/v4/analysis_validation.json":
        "analysis/external-behavior-v4/effect-analysis-v1/"
        "analysis_validation_report.json",
    "results/v4/bootstrap.npz":
        "analysis/external-behavior-v4/effect-analysis-v1/bootstrap.npz",
    "results/v4/block_effects.csv":
        "analysis/external-behavior-v4/effect-analysis-v1/"
        "block_effects.csv",
    "results/v4/seed_effects.csv":
        "analysis/external-behavior-v4/effect-analysis-v1/"
        "seed_effects.csv",
    "results/v4/pair_effects.csv":
        "analysis/external-behavior-v4/effect-analysis-v1/"
        "pair_effects.csv",
    "results/v4/pattern_effects.csv":
        "analysis/external-behavior-v4/effect-analysis-v1/"
        "pattern_effects.csv",
    "results/v4/relation_effects.csv":
        "analysis/external-behavior-v4/effect-analysis-v1/"
        "relation_effects.csv",
    "code/v3_effect_analysis.py":
        "toolkit/external_behavior_v3/effect_analysis.py",
    "code/v3_variance_design.py":
        "toolkit/external_behavior_v3/variance_design.py",
    "code/v4_directional_design.py":
        "toolkit/external_behavior_v4/directional_design.py",
    "code/v4_effect_analysis.py":
        "toolkit/external_behavior_v4/effect_analysis.py",
    "paper/tmlr_external_behavioral_validation.tex":
        "../semantic_novelty_audit_paper/"
        "tmlr_external_behavioral_validation.tex",
    "paper/references.bib":
        "../semantic_novelty_audit_paper/references.bib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def read_effects(path: Path) -> list[float]:
    with path.open(newline="", encoding="ascii") as handle:
        return [float(row["effect_pp"]) for row in csv.DictReader(handle)]


def counts(path: Path) -> dict[str, Any]:
    values = read_effects(path)
    return {
        "count": len(values),
        "negative": sum(value < 0 for value in values),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "minimum_pp": min(values),
        "maximum_pp": max(values),
    }


def build_summary() -> dict[str, Any]:
    v3_root = ROOT / "analysis/external-behavior-v3/effect-analysis-v1"
    v4_root = ROOT / "analysis/external-behavior-v4/effect-analysis-v1"
    v3 = read_json(v3_root / "results.json")
    v4 = read_json(v4_root / "results.json")
    variance = read_json(
        ROOT
        / "analysis/external-behavior-v3/variance-design-v1/"
        "variance_results.json"
    )
    strict_gate = read_json(
        ROOT
        / "analysis/external-behavior-v3/variance-design-v1/"
        "v4_design_gate_report.json"
    )
    power = read_json(
        ROOT
        / "analysis/external-behavior-v4/directional-design-v1/"
        "power_results.json"
    )
    return {
        "benchmark": {
            "name": "CLUTRR",
            "scope": (
                "Task 1, relation length 3, clean three-sentence stories, "
                "23 relation classes"
            ),
            "semantic_identity": (
                "ordered target relation and three-edge relation "
                "composition"
            ),
            "surface_intervention": (
                "sentence-order permutations with evaluation/exposure "
                "source-group separation"
            ),
        },
        "claim_boundary": {
            "allowed": [
                (
                    "For the frozen CLUTRR scope, semantic-pattern exposure "
                    "causally increased accuracy on source-group-disjoint "
                    "surface realizations for the registered byte "
                    "Transformer and training budget."
                ),
                (
                    "The prospective, target-disjoint V4 study passed its "
                    "registered directional Positive Gate."
                ),
            ],
            "forbidden": [
                "The intervention identifies an internal memorization mechanism.",
                "The effect generalizes to all CLUTRR tasks or architectures.",
                "Semantic overlap necessarily improves accuracy.",
                "V3 and V4 are one pooled confirmatory estimate.",
            ],
        },
        "schema_version": "clutrr-v3-v4-paper-evidence-v1",
        "v3_pilot": {
            "decision": v3["decision"],
            "integrity": v3["integrity"],
            "pair_signs": counts(v3_root / "pair_effects.csv"),
            "pattern_signs": counts(v3_root / "pattern_effects.csv"),
            "primary": v3["primary"],
            "seed_signs": counts(v3_root / "seed_effects.csv"),
            "status": (
                "registered No-Go because the 95% interval width exceeded "
                "the frozen 2 pp precision requirement"
            ),
        },
        "v3_variance_diagnosis": {
            "bootstrap_shapley": variance["bootstrap_shapley"],
            "strict_replication_gate": strict_gate["gate"],
            "strict_replication_projection": strict_gate["projection"],
        },
        "v4_confirmatory": {
            "block_signs": counts(v4_root / "block_effects.csv"),
            "decision": v4["decision"],
            "design_power_gate": power["gate"],
            "integrity": v4["integrity"],
            "pair_signs": counts(v4_root / "pair_effects.csv"),
            "pattern_signs": counts(v4_root / "pattern_effects.csv"),
            "primary": v4["primary"],
            "secondary": v4["secondary"],
            "seed_signs": counts(v4_root / "seed_effects.csv"),
            "status": "prospective target-disjoint directional confirmation",
        },
    }


def paper_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    v3 = summary["v3_pilot"]
    v4 = summary["v4_confirmatory"]
    return [
        {
            "decision": "Registered precision No-Go",
            "effect_pp": v3["primary"]["effect_pp"],
            "interval_lower_pp": v3["primary"]["confidence_interval_95"][
                "lower_pp"
            ],
            "interval_upper_pp": v3["primary"]["confidence_interval_95"][
                "upper_pp"
            ],
            "pairs": v3["integrity"]["pattern_pairs"],
            "patterns": v3["integrity"]["patterns"],
            "role": "V3 preregistered pilot",
            "runs": v3["integrity"]["prediction_files"],
            "seeds": v3["integrity"]["model_seeds"],
        },
        {
            "decision": "Directional Positive Go",
            "effect_pp": v4["primary"]["effect_pp"],
            "interval_lower_pp": v4["primary"]["confidence_interval_95"][
                "lower_pp"
            ],
            "interval_upper_pp": v4["primary"]["confidence_interval_95"][
                "upper_pp"
            ],
            "pairs": v4["integrity"]["pattern_pairs"],
            "patterns": v4["integrity"]["patterns"],
            "role": "V4 target-disjoint confirmation",
            "runs": v4["integrity"]["prediction_files"],
            "seeds": v4["integrity"]["model_seeds"],
        },
    ]


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Study & Pairs & Targets & Runs & Effect (pp) & 95\% CI / Gate \\",
        r"\midrule",
    ]
    for row in rows:
        label = "V3 pilot" if row["role"].startswith("V3") else "V4 confirm."
        lines.append(
            f"{label} & {row['pairs']} & {row['patterns']} & "
            f"{row['runs']} & {row['effect_pp']:+.3f} & "
            f"$[{row['interval_lower_pp']:.3f},"
            f"{row['interval_upper_pp']:.3f}]$; "
            f"{row['decision']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{CLUTRR exposure interventions. V3 and V4 use "
                r"disjoint target semantic patterns and are not pooled.}"
            ),
            r"\label{tab:clutrr-external-validation}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


def claim_matrix(summary: dict[str, Any]) -> str:
    v3 = summary["v3_pilot"]
    v4 = summary["v4_confirmatory"]
    return f"""# CLUTRR V3/V4 Claim-Evidence Matrix

| Claim | Evidence | Allowed wording |
|---|---|---|
| V3 detected a positive but imprecise effect | `{v3['primary']['effect_pp']:.4f} pp`, 95% CI `[{v3['primary']['confidence_interval_95']['lower_pp']:.4f}, {v3['primary']['confidence_interval_95']['upper_pp']:.4f}]`; registered decision `{v3['decision']['classification']}` | V3 was a preregistered pilot whose directional evidence was positive but whose registered precision Gate failed |
| V3 uncertainty was dominated by pair heterogeneity | Bootstrap-Shapley pattern-pair share `94.40%` | Pattern selection, not additional seeds or surface groups, dominated V3 uncertainty |
| The original 2 pp precision rule was infeasible | At infinite seeds, projected minimum `634` pairs; only `139` V3-disjoint pairs remained | Do not rerun the same strict precision design |
| V4 was prospectively frozen after V3 | Pilot-shrunk design effect `7.66875 pp`; 100,000-draw power Gate selected 2 blocks, 40 pairs, 32 runs | V4 is a target-disjoint prospective confirmation informed by a declared pilot |
| V4 passed the directional Gate | `{v4['primary']['effect_pp']:.4f} pp`, 95% CI `[{v4['primary']['confidence_interval_95']['lower_pp']:.4f}, {v4['primary']['confidence_interval_95']['upper_pp']:.4f}]`; `{v4['decision']['classification']}` | Semantic-pattern exposure increased held-out-surface accuracy in the frozen CLUTRR scope |
| V4 direction was descriptively consistent | 2/2 blocks, 8/8 seeds, and 40/40 pairs positive; 76/80 patterns positive | Report as descriptive consistency, not additional multiplicity-free tests |

## Forbidden Claims

- The intervention identifies an internal memorization mechanism.
- The effect applies to all CLUTRR tasks, relation lengths, or model classes.
- V3 and V4 form one pooled confirmatory estimate.
- Semantic overlap necessarily improves model accuracy in every benchmark.
- The sentence-permutation intervention proves general natural-language reasoning.
"""


def readme(summary: dict[str, Any]) -> str:
    v4 = summary["v4_confirmatory"]
    return f"""# CLUTRR V3/V4 Paper-Ready Evidence Bundle

This anonymous, lightweight bundle supports the CLUTRR external behavioral
validation reported in the TMLR manuscript. It contains frozen protocols,
validation reports, run-artifact hashes, bootstrap draws, paper-facing tables,
and exact analysis-code snapshots.

## Main Confirmatory Result

- V4 exposed-minus-unexposed accuracy: `{v4['primary']['effect_pp']:.4f} pp`
- Hierarchical bootstrap 95% CI:
  `[{v4['primary']['confidence_interval_95']['lower_pp']:.4f}, {v4['primary']['confidence_interval_95']['upper_pp']:.4f}] pp`
- Registered decision: `{v4['decision']['classification']}`
- Descriptive consistency: 2/2 blocks, 8/8 seeds, 40/40 pairs positive

V3 is retained as a preregistered pilot and is not pooled with V4. Its effect
was positive, but its frozen 2 pp precision Gate was infeasible and therefore
returned `imprecise_no_go`.

## Scope

The intervention covers CLUTRR Task 1, relation length 3, clean
three-sentence stories, one frozen byte Transformer, and the registered
training budget. It measures transfer from semantic-pattern exposure to
source-group-disjoint surface realizations. It does not identify an internal
memorization mechanism.

## Bundle Layout

- `SUMMARY.json`: machine-readable paper claims and boundaries.
- `paper/`: publication table and claim-evidence matrix.
- `protocols/`: frozen preregistrations and Gate reports.
- `design/`: V3 variance diagnosis and V4 power/block design.
- `execution/`: post-training integrity and artifact hashes.
- `results/`: exact V3/V4 estimates, bootstrap draws, and subgroup tables.
- `code/`: exact analysis/design code snapshots.
- `SOURCE_FILES.json`: source-to-bundle provenance and hashes.
- `SHA256SUMS`: checksums for every bundle file except itself.
- `verify_bundle.py`: dependency-free integrity and anonymity check.

Raw predictions, checkpoints, training logs, and machine environments are
intentionally excluded. Their frozen SHA-256 commitments remain in the
run-artifact manifests.

Run `python3 verify_bundle.py` from this directory to validate the bundle.
"""


def reproduce() -> str:
    return """# Reproduction Notes

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
"""


VERIFY_SCRIPT = r'''#!/usr/bin/env python3
"""Verify the lightweight CLUTRR paper evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures = []
    expected = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
        path = ROOT / name
        if not path.is_file():
            failures.append(f"missing:{name}")
        elif sha256(path) != digest:
            failures.append(f"hash:{name}")
    observed = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if observed != set(expected):
        failures.append(
            "file_set:"
            + json.dumps(
                {
                    "extra": sorted(observed - set(expected)),
                    "missing": sorted(set(expected) - observed),
                },
                sort_keys=True,
            )
        )
    summary = json.loads((ROOT / "SUMMARY.json").read_text(encoding="ascii"))
    v4 = summary["v4_confirmatory"]
    if v4["primary"]["effect_pp"] != 23.3390625:
        failures.append("paper_value:v4_effect")
    interval = v4["primary"]["confidence_interval_95"]
    if interval["lower_pp"] != 17.749768816685584:
        failures.append("paper_value:v4_lower")
    if interval["upper_pp"] != 29.22568903348098:
        failures.append("paper_value:v4_upper")
    if v4["decision"]["classification"] != "directional_positive_go":
        failures.append("paper_value:v4_decision")
    forbidden = (
        "/" + "Users" + "/",
        "byte" + "dance",
        "OPK" + "MRC",
    )
    text_suffixes = {".csv", ".json", ".md", ".py", ".tex"}
    for name in sorted(observed):
        path = ROOT / name
        if path.suffix not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                failures.append(f"anonymity:{name}:{marker}")
    result = {
        "checked_files": len(expected),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    provenance: dict[str, dict[str, Any]] = {}
    for destination_name, source_name in sorted(COPY_FILES.items()):
        source = ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        provenance[destination_name] = {
            "bundle_sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "source_path": source_name,
            "source_sha256": sha256(source),
        }
    summary = build_summary()
    write_json(output / "SUMMARY.json", summary)
    rows = paper_rows(summary)
    (output / "paper").mkdir(exist_ok=True)
    write_table_csv(output / "paper/clutrr_results.csv", rows)
    write_table_tex(output / "paper/clutrr_results.tex", rows)
    (output / "paper/claim_evidence_matrix.md").write_text(
        claim_matrix(summary),
        encoding="utf-8",
    )
    (output / "README.md").write_text(readme(summary), encoding="utf-8")
    (output / "REPRODUCE.md").write_text(reproduce(), encoding="utf-8")
    (output / "verify_bundle.py").write_text(
        VERIFY_SCRIPT,
        encoding="ascii",
    )
    write_json(output / "SOURCE_FILES.json", provenance)
    manifest_entries = []
    for path in sorted(
        (
            item
            for item in output.rglob("*")
            if item.is_file() and item.name not in {"SHA256SUMS", "MANIFEST.json"}
        ),
        key=lambda item: item.relative_to(output).as_posix(),
    ):
        manifest_entries.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256(path),
            }
        )
    write_json(
        output / "MANIFEST.json",
        {
            "files": manifest_entries,
            "schema_version": "clutrr-paper-bundle-manifest-v1",
        },
    )
    checksum_paths = sorted(
        (
            path
            for path in output.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.relative_to(output).as_posix(),
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="ascii",
    )
    return {
        "bundle": str(output),
        "files": len(checksum_paths),
        "sha256s_sha256": sha256(output / "SHA256SUMS"),
        "status": "built",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
