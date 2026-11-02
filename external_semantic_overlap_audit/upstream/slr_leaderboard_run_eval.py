"""
Eval worker: polls AIML-TUDA/slr-leaderboard-requests for pending models,
runs inference + IPT via the llms-gaming-verifiers scripts, saves results.

Run this on a GPU machine (not in the HF Space):
    python backend/run_eval.py --vllm-url http://localhost:8000/v1 --poll-interval 60
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

REQUESTS_DATASET = "AIML-TUDA/slr-leaderboard-requests"
RESULTS_DATASET = "AIML-TUDA/slr-leaderboard-results"
REPO_URL = "https://github.com/ml-research/llms-gaming-verifiers"
REPO_DIR = Path("/tmp/llms-gaming-verifiers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between queue polls")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--parallel-size", type=int, default=1)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    api = HfApi(token=args.hf_token)
    _ensure_repo_cloned()

    print(f"[worker] Polling {REQUESTS_DATASET} every {args.poll_interval}s")
    while True:
        try:
            requests = _fetch_pending(api)
            for req in requests:
                _process_request(req, api, args)
        except Exception as e:
            print(f"[worker] Error in poll loop: {e}", file=sys.stderr)
        time.sleep(args.poll_interval)


def _ensure_repo_cloned():
    if not REPO_DIR.exists():
        print(f"[worker] Cloning {REPO_URL}")
        subprocess.run(["git", "clone", "--recurse-submodules", REPO_URL, str(REPO_DIR)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REPO_DIR / "requirements.txt")], check=True)
    else:
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)


def _fetch_pending(api: HfApi) -> list[dict]:
    try:
        files = list(api.list_repo_files(REQUESTS_DATASET, repo_type="dataset"))
    except RepositoryNotFoundError:
        return []

    pending = []
    for fname in files:
        if not fname.startswith("pending/") or not fname.endswith(".json"):
            continue
        path = hf_hub_download(repo_id=REQUESTS_DATASET, filename=fname, repo_type="dataset")
        with open(path) as f:
            req = json.load(f)
        req["_filename"] = fname
        pending.append(req)
    return pending


def _process_request(req: dict, api: HfApi, args) -> None:
    model_id = req["model_id"]
    print(f"[worker] Evaluating {model_id}")

    _move_request(req, "pending", "running", api)

    with tempfile.TemporaryDirectory() as out_dir:
        try:
            # Step 1: inference
            _run_inference(model_id, out_dir, args)
            # Step 2: IPT shortcut detection
            _run_ipt(out_dir, args)
            # Step 3: parse and push results
            result = _parse_and_build_result(model_id, req, out_dir)
            _push_result(result, api)
            _move_request(req, "running", "finished", api)
            print(f"[worker] Done: {model_id}")
        except Exception as e:
            print(f"[worker] Failed {model_id}: {e}", file=sys.stderr)
            _move_request(req, "running", "failed", api)


def _run_inference(model_id: str, out_dir: str, args) -> None:
    cmd = [
        sys.executable,
        str(REPO_DIR / "evaluate_model_vllm.py"),
        "--model", model_id,
        "--out-path", out_dir,
        "--max-new-tokens", str(args.max_new_tokens),
        "--parallel-size", str(args.parallel_size),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_DIR))


def _run_ipt(out_dir: str, args) -> None:
    cmd = [
        sys.executable,
        str(REPO_DIR / "shortcuts.py"),
        "--output-dir", out_dir,
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_DIR))


def _parse_and_build_result(model_id: str, req: dict, out_dir: str) -> dict:
    import csv
    import re

    ipt_dir = Path(out_dir) / "ipt_results"
    perf_file = ipt_dir / "performance.csv"
    sc_file = ipt_dir / "shortcuts.csv"

    accuracy = {}
    shortcuts = {}
    syntax_pct = None
    tokens_int = None
    usd = None

    # performance.csv columns: model_name, Basic%, Easy%, Medium%, Hard%, Avg%, Syntax%, Tokens, Cost
    if perf_file.exists():
        with open(perf_file) as f:
            for row in csv.DictReader(f):
                if row.get("model_name", "").upper() == "SUM":
                    continue
                accuracy = {
                    "basic":  _parse_float(row.get("Basic%")),
                    "easy":   _parse_float(row.get("Easy%")),
                    "medium": _parse_float(row.get("Medium%")),
                    "hard":   _parse_float(row.get("Hard%")),
                }
                syntax_pct = _parse_float(row.get("Syntax%"))
                # "9.4M" → 9_400_000
                t = row.get("Tokens", "")
                m = re.match(r"([\d.]+)M", t)
                if m:
                    tokens_int = int(float(m.group(1)) * 1_000_000)
                # "$103.13" → 103.13
                c = row.get("Cost", "")
                m2 = re.match(r"\$([\d.]+)", c)
                if m2:
                    usd = float(m2.group(1))
                break  # first non-SUM row is the evaluated model

    # shortcuts.csv columns: model_name, Basic, Easy, Medium, Hard, Ns_total
    if sc_file.exists():
        with open(sc_file) as f:
            for row in csv.DictReader(f):
                if row.get("model_name", "").upper() == "SUM":
                    continue
                shortcuts = {
                    "basic":  int(row.get("Basic",  0) or 0),
                    "easy":   int(row.get("Easy",   0) or 0),
                    "medium": int(row.get("Medium", 0) or 0),
                    "hard":   int(row.get("Hard",   0) or 0),
                }
                break

    return {
        "model_name": model_id.split("/")[-1],
        "model_id": model_id,
        "model_type": "Unknown",
        "rlvr": None,
        "accuracy": accuracy,
        "shortcuts": shortcuts,
        "efficiency": {"syntax": syntax_pct, "tokens": tokens_int, "usd": usd},
        "eval_date": datetime.now(timezone.utc).date().isoformat(),
        "paper_result": False,
        "submitted_by": req.get("submitted_by", ""),
    }


def _parse_float(val) -> float | None:
    try:
        return round(float(val), 1) if val not in (None, "") else None
    except (ValueError, TypeError):
        return None


def _push_result(result: dict, api: HfApi) -> None:
    model_slug = result["model_id"].replace("/", "__")
    filename = f"{model_slug}_{result['eval_date']}.json"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(result, f, indent=2)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=RESULTS_DATASET,
            repo_type="dataset",
            commit_message=f"Add results for {result['model_id']}",
        )
    finally:
        os.unlink(tmp_path)


def _move_request(req: dict, from_status: str, to_status: str, api: HfApi) -> None:
    old_fname = req["_filename"]
    new_fname = old_fname.replace(f"{from_status}/", f"{to_status}/")
    req["status"] = to_status

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({k: v for k, v in req.items() if k != "_filename"}, f, indent=2)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=new_fname,
            repo_id=REQUESTS_DATASET,
            repo_type="dataset",
            commit_message=f"Move {req['model_id']} to {to_status}",
        )
        api.delete_file(
            path_in_repo=old_fname,
            repo_id=REQUESTS_DATASET,
            repo_type="dataset",
            commit_message=f"Remove old {from_status} request for {req['model_id']}",
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
