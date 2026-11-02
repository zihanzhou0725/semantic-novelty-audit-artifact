from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


IDENTITY_FRAGMENTS = (
    b"/" + b"Users" + b"/",
    b"\\" + b"Users" + b"\\",
    b"byte" + b"dance",
    b"OP" + b"KMRC",
)
IDENTITY_PATTERN = re.compile(
    b"|".join(re.escape(fragment) for fragment in IDENTITY_FRAGMENTS),
    re.IGNORECASE,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_scope(rows: list[dict[str, str]], scope: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("scope") == scope]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for scope={scope}; found {len(matches)}.")
    return matches[0]


def find_row(
    rows: list[dict[str, str]],
    **expected: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one row matching {expected}; found {len(matches)}."
        )
    return matches[0]


def assert_close(observed: float, expected: float, tolerance: float = 1.0e-10) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"{observed} != {expected} within {tolerance}.")


def validate_results(root: Path) -> None:
    results = root / "results"
    public = results / "public_baseline_audit"
    phase2_9 = results / "phase2_9" / "family_agnostic_bayes"
    required = (
        public / "analysis" / "claim_gates.json",
        public / "analysis" / "group_summary.csv",
        public / "analysis" / "protocol_rankings.csv",
        public / "family_transfer" / "group_summary.csv",
        public / "family_transfer" / "paired_vs_transformer.csv",
        phase2_9 / "gate.json",
        phase2_9 / "group_summary.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing compact artifacts: {missing}")

    public_gate = read_json(public / "analysis" / "claim_gates.json")
    assert_close(
        float(public_gate["architecture_macro_difference"]),
        0.02314185698827108,
    )
    assert_close(float(public_gate["rank_spearman"]), 1.0)
    if public_gate["cross_architecture_inflation_supported"] is not True:
        raise AssertionError("Public architecture inflation gate must pass.")
    if public_gate["ranking_instability_supported"] is not False:
        raise AssertionError("Public ranking-instability gate must fail.")

    public_rows = read_rows(public / "analysis" / "group_summary.csv")
    strict_hopfield = find_row(
        public_rows,
        protocol="strict_function",
        model="symbolic_hopfield",
    )
    strict_deepsets = find_row(
        public_rows,
        protocol="strict_function",
        model="symbolic_deepsets",
    )
    assert_close(
        float(strict_hopfield["primary_accuracy_mean"]),
        0.7293487310409545,
    )
    assert_close(
        float(strict_deepsets["primary_accuracy_mean"]),
        0.3886177003383636,
    )

    transfer_rows = read_rows(
        public / "family_transfer" / "group_summary.csv"
    )
    transfer_bayes = find_row(
        transfer_rows,
        family="heldout_macro",
        model="family_conditioned_bayes",
    )
    transfer_transformer = find_row(
        transfer_rows,
        family="heldout_macro",
        model="symbolic_transformer",
    )
    assert_close(float(transfer_bayes["accuracy_mean"]), 0.9916799902915955)
    assert_close(
        float(transfer_transformer["accuracy_mean"]),
        0.11533999890089035,
    )

    if read_json(phase2_9 / "gate.json")["passed"] is not True:
        raise AssertionError("Phase 2.9 information-sufficiency gate must be Go.")

    phase2_9_gate = read_json(phase2_9 / "gate.json")
    assert_close(
        float(
            phase2_9_gate["values"][
                "all_heldout_family_balanced_expected_accuracy"
            ]
        ),
        0.9892987124526501,
    )


def validate_anonymity(root: Path) -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".pyc"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            raise AssertionError(f"Finder metadata present: {path}")
        if path.name == "episodes.csv":
            raise AssertionError(f"Episode-level output present: {path}")
        if path.suffix.lower() in forbidden_suffixes:
            raise AssertionError(f"Large/generated artifact present: {path}")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise AssertionError(f"Unexpected file larger than 10 MiB: {path}")
        if path.suffix.lower() in {
            ".md",
            ".py",
            ".tex",
            ".json",
            ".csv",
            ".txt",
            ".bib",
        }:
            if IDENTITY_PATTERN.search(path.read_bytes()):
                raise AssertionError(f"Identity or absolute path leak: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()
    root = args.archive_root.resolve()
    validate_results(root)
    validate_anonymity(root)
    print("COMPACT_ARCHIVE_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
