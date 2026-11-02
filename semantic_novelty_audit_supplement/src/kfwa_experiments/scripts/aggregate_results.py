from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.utils import load_json


DEFAULT_COLUMNS = [
    "task",
    "model",
    "seed",
    "parameters",
    "best_epoch",
    "best_val_loss",
    "best_selection_metric",
    "best_selection_score",
    "test_loss",
    "test_ce",
    "test_accuracy",
    "test_seen_accuracy",
    "test_unseen_accuracy",
    "test_seen_function_accuracy",
    "test_unseen_function_accuracy",
    "test_in_support_accuracy",
    "test_out_support_accuracy",
    "test_ece",
    "test_brier",
    "run_dir",
]


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    """Collect metrics.json rows."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        row = load_json(metrics_path)
        row.setdefault("run_dir", str(metrics_path.parent))
        rows.append(row)
    return rows


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Use standard columns first and append extras."""

    columns = [column for column in DEFAULT_COLUMNS if any(column in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def write_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write CSV, falling back if summary is locked."""

    columns = ordered_columns(rows)
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with fallback.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return fallback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate KFWA benchmark metrics.")
    parser.add_argument("--runs-dir", default="kfwa_experiments/runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out) if args.out else runs_dir / "summary.csv"
    rows = collect_rows(runs_dir)
    if not rows:
        print(f"No metrics.json files found under {runs_dir}.")
        return 1
    written = write_summary(out_path, rows)
    print(f"Wrote {len(rows)} rows to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
