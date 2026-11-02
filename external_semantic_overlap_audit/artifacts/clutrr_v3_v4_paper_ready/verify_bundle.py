#!/usr/bin/env python3
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
