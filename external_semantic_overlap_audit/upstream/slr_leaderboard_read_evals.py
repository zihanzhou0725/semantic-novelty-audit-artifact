import json
import os
from pathlib import Path

from src.about import RESULTS_DATASET

_INITIAL_RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "initial_results.json"


def load_results() -> list[dict]:
    """Load results: always start with paper results, then overlay HF Dataset results."""
    results = _load_initial_results()
    hf_results = _load_hf_results()
    # HF results keyed by model_id + reasoning_effort to avoid duplicates with paper seeds
    existing_keys = {_result_key(r) for r in results}
    for r in hf_results:
        key = _result_key(r)
        if key not in existing_keys:
            results.append(r)
            existing_keys.add(key)
    return results


def _result_key(r: dict) -> str:
    effort = r.get("reasoning_effort", "")
    return f"{r.get('model_id', '')}|{effort}"


def _load_initial_results() -> list[dict]:
    with open(_INITIAL_RESULTS_PATH) as f:
        return json.load(f)


def _load_hf_results() -> list[dict]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        import tempfile

        api = HfApi()
        try:
            files = api.list_repo_files(RESULTS_DATASET, repo_type="dataset")
        except RepositoryNotFoundError:
            return []

        results = []
        for fname in files:
            if not fname.endswith(".json"):
                continue
            try:
                path = hf_hub_download(
                    repo_id=RESULTS_DATASET,
                    filename=fname,
                    repo_type="dataset",
                )
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results.extend(data)
                    else:
                        results.append(data)
            except EntryNotFoundError:
                continue
        return results
    except Exception:
        return []
