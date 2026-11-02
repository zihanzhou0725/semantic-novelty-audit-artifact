#!/usr/bin/env python3
"""Run or resume the frozen 60-job quality-enhancement P1 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quality_p1_common import (
    EXTERNAL_ROOT,
    P1_DIR,
    P1_SCHEMA_VERSION,
    sha256_file,
    write_json,
)


RUN_ROOT = EXTERNAL_ROOT / "runs" / "quality-enhancement-v1" / "p1"
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "run_quality_p1.py"
VALIDATE_SCRIPT = SCRIPT_DIR / "validate_quality_p1_protocol.py"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def load_rows() -> list[dict[str, str]]:
    with (P1_DIR / "run_manifest.csv").open(
        newline="",
        encoding="ascii",
    ) as handle:
        return list(csv.DictReader(handle))


def completed(run_id: str) -> bool:
    run_dir = RUN_ROOT / run_id
    required = (
        "config.json",
        "history.csv",
        "final_model.pt",
        "eval_predictions.npz",
        "metrics.json",
        "progress.json",
    )
    if not all((run_dir / filename).is_file() for filename in required):
        return False
    progress = json.loads(
        (run_dir / "progress.json").read_text(encoding="ascii")
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="ascii"))
    return (
        progress.get("status") == "completed"
        and metrics.get("epochs_completed") == 60
        and metrics.get("final_epoch") == 60
        and metrics.get("optimizer_updates") == 4740
    )


def status_payload(
    rows: list[dict[str, str]],
    *,
    started_at: str,
    current: str | None,
    failures: list[dict[str, Any]],
    state: str,
) -> dict[str, Any]:
    complete_ids = [row["run_id"] for row in rows if completed(row["run_id"])]
    return {
        "completed": len(complete_ids),
        "completed_run_ids": complete_ids,
        "current_run_id": current,
        "design_validation_sha256": sha256_file(
            P1_DIR / "validation_report.json"
        ),
        "driver_pid": os.getpid(),
        "failures": failures,
        "last_update": now_utc(),
        "protocol": P1_SCHEMA_VERSION,
        "started_at": started_at,
        "state": state,
        "total": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen P1 matrix.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Retry incomplete jobs from their frozen initialization and manifests.",
    )
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT)],
        cwd=str(EXTERNAL_ROOT.parent),
        check=True,
    )
    gate = json.loads(
        (P1_DIR / "validation_report.json").read_text(encoding="ascii")
    )
    if gate.get("design_gate") != "passed":
        raise SystemExit("P1 Design Gate has not passed.")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    logs_dir = RUN_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    rows = load_rows()
    started_at = now_utc()
    failures: list[dict[str, Any]] = []
    execution = {
        "design_checksum_manifest_sha256": gate[
            "p1_checksum_manifest_sha256"
        ],
        "design_validation_sha256": sha256_file(
            P1_DIR / "validation_report.json"
        ),
        "driver_pid": os.getpid(),
        "protocol": P1_SCHEMA_VERSION,
        "start_time": started_at,
        "training_started": True,
    }
    atomic_json(RUN_ROOT / "execution_state.json", execution)
    atomic_json(
        RUN_ROOT / "matrix_status.json",
        status_payload(
            rows,
            started_at=started_at,
            current=None,
            failures=failures,
            state="running",
        ),
    )

    driver_log = RUN_ROOT / "matrix_driver.log"
    with driver_log.open("a", encoding="ascii") as driver:
        driver.write(
            f"{now_utc()} start pid={os.getpid()} total={len(rows)} "
            f"device={args.device}\n"
        )
        driver.flush()
        for index, row in enumerate(rows, start=1):
            run_id = row["run_id"]
            if completed(run_id):
                driver.write(f"{now_utc()} skip completed {run_id}\n")
                driver.flush()
                continue
            run_dir = RUN_ROOT / run_id
            incomplete = run_dir.exists() and any(run_dir.iterdir())
            if incomplete and not args.retry_incomplete:
                failure = {
                    "returncode": None,
                    "run_id": run_id,
                    "time": now_utc(),
                    "type": "incomplete_run_requires_retry_flag",
                }
                failures.append(failure)
                atomic_json(
                    RUN_ROOT / "matrix_status.json",
                    status_payload(
                        rows,
                        started_at=started_at,
                        current=run_id,
                        failures=failures,
                        state="failed",
                    ),
                )
                driver.write(f"{now_utc()} fail incomplete {run_id}\n")
                return 2

            atomic_json(
                RUN_ROOT / "matrix_status.json",
                status_payload(
                    rows,
                    started_at=started_at,
                    current=run_id,
                    failures=failures,
                    state="running",
                ),
            )
            command = [
                sys.executable,
                str(RUN_SCRIPT),
                "--model",
                row["model"],
                "--seed",
                row["model_seed"],
                "--arm",
                row["arm"],
                "--device",
                args.device,
            ]
            if incomplete:
                command.append("--retry")
            run_log_path = logs_dir / f"{run_id}.log"
            driver.write(
                f"{now_utc()} launch {index}/{len(rows)} {run_id} "
                f"log={run_log_path}\n"
            )
            driver.flush()
            run_start = time.time()
            with run_log_path.open("a", encoding="ascii") as run_log:
                process = subprocess.run(
                    command,
                    cwd=str(EXTERNAL_ROOT.parent),
                    stdout=run_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.time() - run_start
            if process.returncode != 0 or not completed(run_id):
                failure = {
                    "elapsed_seconds": elapsed,
                    "log": str(run_log_path),
                    "returncode": process.returncode,
                    "run_id": run_id,
                    "time": now_utc(),
                    "type": "training_process_failed",
                }
                failures.append(failure)
                atomic_json(
                    RUN_ROOT / "matrix_status.json",
                    status_payload(
                        rows,
                        started_at=started_at,
                        current=run_id,
                        failures=failures,
                        state="failed",
                    ),
                )
                driver.write(
                    f"{now_utc()} fail {run_id} rc={process.returncode} "
                    f"seconds={elapsed:.1f}\n"
                )
                driver.flush()
                return process.returncode or 3
            driver.write(
                f"{now_utc()} complete {run_id} seconds={elapsed:.1f}\n"
            )
            driver.flush()

        atomic_json(
            RUN_ROOT / "matrix_status.json",
            status_payload(
                rows,
                started_at=started_at,
                current=None,
                failures=failures,
                state="completed",
            ),
        )
        driver.write(f"{now_utc()} matrix complete\n")
        driver.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
