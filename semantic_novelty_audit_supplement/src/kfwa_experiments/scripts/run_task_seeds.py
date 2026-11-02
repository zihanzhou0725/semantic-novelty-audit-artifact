from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONTEXT_MODELS = [
    "mlp",
    "cnn",
    "oracle_remap",
    "film_cnn",
    "hyper_cnn",
    "lora_cnn",
    "vit",
    "kope_lite",
    "kfwa",
    "kfwa_no_kuramoto",
    "kfwa_no_sync_bias",
    "kfwa_no_adapter",
]

DEFAULT_SYMBOLIC_MODELS = [
    "symbolic_oracle",
    "symbolic_mlp",
    "symbolic_deepsets",
    "symbolic_gru",
    "symbolic_transformer",
    "symbolic_kfwa",
    "symbolic_kfwa_no_kuramoto",
    "symbolic_kfwa_no_sync_bias",
    "symbolic_kfwa_no_adapter",
    "symbolic_kfwa_srin",
    "symbolic_lop_ri_m64",
    "symbolic_lop_ssr_m64",
]


def parse_seeds(raw: str) -> list[int]:
    """Parse comma/range seed strings, e.g. 1-3,7."""

    seeds: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a KFWA task over models and seeds.")
    parser.add_argument("--task", default="context_permutation")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", default="1-3")
    parser.add_argument("--out-dir", default="kfwa_experiments/runs_context_permutation")
    args, extra = parser.parse_known_args(argv)

    models = args.models
    if models is None:
        models = DEFAULT_SYMBOLIC_MODELS if args.task == "symbolic_rule_mapping" else DEFAULT_CONTEXT_MODELS

    script = Path(__file__).with_name("run_experiment.py")
    for model in models:
        for seed in parse_seeds(args.seeds):
            cmd = [
                sys.executable,
                str(script),
                "--task",
                args.task,
                "--model",
                model,
                "--seed",
                str(seed),
                "--out-dir",
                args.out_dir,
                *extra,
            ]
            print("\n==>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
