#!/usr/bin/env python3
"""Run the frozen 18-run Boolean behavioral matrix sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from freeze_boolean_behavioral_protocol import MODEL_SEEDS, SCHEMA_VERSION, TASKS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path) -> None:
    if path.exists():
        try:
            existing_pid = int(path.read_text().strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and process_is_alive(existing_pid):
            raise SystemExit(
                f"Matrix runner lock is held by live PID {existing_pid}: {path}"
            )
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{os.getpid()}\n")


def read_run_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    run_ids = [row["run_id"] for row in rows]
    expected = len(TASKS) * len(MODEL_SEEDS) * 2
    if len(run_ids) != expected or len(set(run_ids)) != expected:
        raise SystemExit(
            f"Frozen run manifest must contain {expected} unique rows, "
            f"found {len(run_ids)} rows and {len(set(run_ids))} unique IDs"
        )
    return run_ids


def completion_is_valid(path: Path, run_id: str) -> bool:
    if not path.exists():
        return False
    completed = json.loads(path.read_text())
    return (
        completed.get("status") == "complete"
        and completed.get("run_id") == run_id
        and completed.get("mode") == "full"
        and completed.get("eligible_for_gate") is True
        and completed.get("execution", {}).get("completed_steps") == 20_000
    )


def build_status(
    *,
    started_at: str,
    run_ids: list[str],
    complete: list[str],
    active: str | None,
    state: str,
    failure: dict | None = None,
) -> dict:
    completed_set = set(complete)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": utc_now(),
        "total_runs": len(run_ids),
        "completed_run_count": len(complete),
        "completed_run_ids": complete,
        "active_run_id": active,
        "pending_run_ids": [
            run_id
            for run_id in run_ids
            if run_id not in completed_set and run_id != active
        ],
        "effect_metrics_examined": False,
        "failure": failure,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "runs" / SCHEMA_VERSION,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_dir = root / "protocols" / SCHEMA_VERSION
    run_ids = read_run_ids(protocol_dir / "run_manifest.csv")
    full_dir = args.output_root / "full"
    matrix_dir = args.output_root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    status_path = matrix_dir / "full_matrix_status.json"
    lock_path = matrix_dir / "full_matrix.lock"

    existing_complete = [
        run_id
        for run_id in run_ids
        if completion_is_valid(full_dir / run_id / "COMPLETED.json", run_id)
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "total_runs": len(run_ids),
                    "already_complete": existing_complete,
                    "pending": [
                        run_id
                        for run_id in run_ids
                        if run_id not in set(existing_complete)
                    ],
                    "effect_metrics_examined": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    acquire_lock(lock_path)
    started_at = utc_now()
    complete = list(existing_complete)
    try:
        for run_id in run_ids:
            completed_path = full_dir / run_id / "COMPLETED.json"
            if completion_is_valid(completed_path, run_id):
                if run_id not in complete:
                    complete.append(run_id)
                print(f"SKIP complete {run_id}", flush=True)
                continue

            state_path = full_dir / run_id / "state.pt"
            command = [
                sys.executable,
                str(root / "scripts" / "run_boolean_behavioral.py"),
                "--run-id",
                run_id,
                "--mode",
                "full",
                "--device",
                args.device,
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--output-root",
                str(args.output_root),
            ]
            if state_path.exists():
                command.append("--resume")

            atomic_write_json(
                status_path,
                build_status(
                    started_at=started_at,
                    run_ids=run_ids,
                    complete=complete,
                    active=run_id,
                    state="running",
                ),
            )
            print(
                f"START {run_id} resume={state_path.exists()} at {utc_now()}",
                flush=True,
            )
            result = subprocess.run(command, cwd=root)
            if result.returncode != 0:
                failure = {
                    "run_id": run_id,
                    "returncode": result.returncode,
                    "failed_at": utc_now(),
                }
                atomic_write_json(
                    status_path,
                    build_status(
                        started_at=started_at,
                        run_ids=run_ids,
                        complete=complete,
                        active=None,
                        state="failed",
                        failure=failure,
                    ),
                )
                raise SystemExit(
                    f"Run {run_id} failed with exit code {result.returncode}"
                )
            if not completion_is_valid(completed_path, run_id):
                raise SystemExit(f"Run returned success without valid completion: {run_id}")
            complete.append(run_id)
            atomic_write_json(
                status_path,
                build_status(
                    started_at=started_at,
                    run_ids=run_ids,
                    complete=complete,
                    active=None,
                    state="running",
                ),
            )
            print(f"DONE {run_id} at {utc_now()}", flush=True)

        atomic_write_json(
            status_path,
            build_status(
                started_at=started_at,
                run_ids=run_ids,
                complete=complete,
                active=None,
                state="complete",
            ),
        )
        print(f"MATRIX COMPLETE at {utc_now()}", flush=True)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
