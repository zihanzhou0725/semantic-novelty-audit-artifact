from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common pseudo-random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    """Return a torch device from a user request."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_run_dir(out_dir: str | Path, task: str, model: str, seed: int) -> Path:
    """Create a timestamped run directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"{task}__{model}__seed{seed}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def as_serializable(value: Any) -> Any:
    """Convert common non-JSON values into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [as_serializable(item) for item in value]
    if isinstance(value, list):
        return [as_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_serializable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
