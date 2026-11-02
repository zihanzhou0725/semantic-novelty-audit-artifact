"""Parallel driver for the 16 frozen CLUTRR V3 runs."""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .training_common import (
    EXECUTION_DIR,
    EXECUTION_ID,
    MAX_CONCURRENT_RUNS,
    PROJECT_ROOT,
    RUN_ROOT,
    artifact_hashes,
    read_csv,
    read_json,
    sha256_file,
    write_json,
)


RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_external_behavior_v3.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _completed(run_id: str) -> bool:
    run_dir = RUN_ROOT / run_id
    completed_path = run_dir / "COMPLETED.json"
    if not completed_path.is_file():
        return False
    completed = read_json(completed_path)
    if completed.get("status") != "completed":
        return False
    for filename, expected in completed["artifacts_sha256"].items():
        path = run_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            return False
    progress = read_json(run_dir / "progress.json")
    return (
        progress.get("status") == "completed"
        and progress.get("epoch") == 30
        and progress.get("optimizer_updates") == 5640
    )


def _status(
    rows: list[dict[str, str]],
    failures: list[dict[str, Any]],
    *,
    state: str,
    started_at: str,
) -> dict[str, Any]:
    completed = [row["run_id"] for row in rows if _completed(row["run_id"])]
    running = [
        row["run_id"]
        for row in rows
        if (RUN_ROOT / row["run_id"] / "progress.json").is_file()
        and not _completed(row["run_id"])
    ]
    return {
        "completed": len(completed),
        "completed_run_ids": completed,
        "execution_id": EXECUTION_ID,
        "failures": failures,
        "last_update_utc": _now(),
        "running_run_ids": running,
        "started_at_utc": started_at,
        "state": state,
        "total": len(rows),
    }


def _launch(row: dict[str, str], log_dir: Path) -> dict[str, Any]:
    run_id = row["run_id"]
    run_dir = RUN_ROOT / run_id
    if run_dir.exists() and not _completed(run_id):
        raise RuntimeError(f"Incomplete run directory exists: {run_dir}")
    if _completed(run_id):
        return {"returncode": 0, "run_id": run_id, "skipped": True}
    log_path = log_dir / f"{run_id}.log"
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--seed",
        row["model_seed"],
        "--arm",
        row["arm"],
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "MKL_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    with log_path.open("a", encoding="ascii") as log:
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "log": str(log_path),
        "returncode": process.returncode,
        "run_id": run_id,
        "skipped": False,
    }


def run_matrix() -> dict[str, Any]:
    execution = read_json(EXECUTION_DIR / "execution_manifest.json")
    if execution["status"] != "frozen_before_training":
        raise RuntimeError("Execution manifest is not frozen.")
    rows = read_csv(EXECUTION_DIR / "run_manifest.csv")
    if len(rows) != 16:
        raise RuntimeError("Frozen run count changed.")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    started_at = _now()
    failures: list[dict[str, Any]] = []
    state_path = RUN_ROOT / "execution_state.json"
    _atomic_json(
        state_path,
        {
            "execution_id": EXECUTION_ID,
            "execution_manifest_sha256": sha256_file(
                EXECUTION_DIR / "execution_manifest.json"
            ),
            "maximum_concurrent_runs": MAX_CONCURRENT_RUNS,
            "primary_effect_computed": False,
            "started_at_utc": started_at,
            "training_started": True,
        },
    )
    status_path = RUN_ROOT / "matrix_status.json"
    _atomic_json(
        status_path,
        _status(rows, failures, state="running", started_at=started_at),
    )
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_RUNS
    ) as executor:
        future_to_row = {
            executor.submit(_launch, row, log_dir): row for row in rows
        }
        for future in concurrent.futures.as_completed(future_to_row):
            row = future_to_row[future]
            try:
                result = future.result()
                if result["returncode"] != 0 or not _completed(row["run_id"]):
                    failures.append(
                        {
                            **result,
                            "time_utc": _now(),
                            "type": "run_failed_or_incomplete",
                        }
                    )
            except Exception as error:
                failures.append(
                    {
                        "error": repr(error),
                        "run_id": row["run_id"],
                        "time_utc": _now(),
                        "type": "driver_exception",
                    }
                )
            with lock:
                status = _status(
                    rows,
                    failures,
                    state="failed" if failures else "running",
                    started_at=started_at,
                )
                _atomic_json(status_path, status)
                print(
                    f"CLUTRR V3 runs {status['completed']}/{status['total']} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    final_state = "completed" if not failures and all(
        _completed(row["run_id"]) for row in rows
    ) else "failed"
    final = _status(
        rows,
        failures,
        state=final_state,
        started_at=started_at,
    )
    _atomic_json(status_path, final)
    if final_state != "completed":
        raise RuntimeError(f"Training matrix failed: {failures}")
    return final
