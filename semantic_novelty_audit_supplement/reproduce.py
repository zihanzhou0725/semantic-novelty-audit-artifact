#!/usr/bin/env python3
"""Single entry point for validating and rebuilding the anonymous archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPTS = SRC / "kfwa_experiments" / "scripts"
PAPER = ROOT / "paper_sources"
BUILD = ROOT / "build"


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def python_command(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *args]


def validate() -> None:
    clean()
    run(
        python_command(
            SCRIPTS / "validate_compact_archive.py",
            "--archive-root",
            str(ROOT),
        )
    )
    run(
        python_command(
            SCRIPTS / "validate_symbolic_protocol.py",
            "--seeds",
            "0-4",
            "--sample-count",
            "256",
        ),
        cwd=SRC,
    )
    run(
        python_command(SCRIPTS / "validate_symbolic_public_baselines.py"),
        cwd=SRC,
    )
    run(
        python_command(SCRIPTS / "validate_symbolic_oracles.py"),
        cwd=SRC,
    )
    run(
        python_command(SCRIPTS / "validate_symbolic_attribution_models.py"),
        cwd=SRC,
    )
    run(
        python_command(
            SCRIPTS / "validate_symbolic_family_agnostic_ceiling.py"
        ),
        cwd=SRC,
    )


def smoke() -> None:
    smoke_dir = BUILD / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    run(
        python_command(
            SCRIPTS / "run_symbolic_fixed_k_sweep.py",
            "--out-dir",
            str(smoke_dir),
            "--models",
            "symbolic_deepsets,symbolic_transformer",
            "--seeds",
            "0",
            "--k-values",
            "4",
            "--device",
            "cpu",
            "--support-noise",
            "0.10",
            "--train-limit",
            "128",
            "--val-limit",
            "64",
            "--test-limit",
            "64",
            "--batch-size",
            "64",
            "--epochs",
            "1",
            "--patience",
            "1",
            "--no-progress",
        ),
        cwd=SRC,
    )


def figures() -> None:
    figure_dir = PAPER / "figures"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BUILD / "mplconfig")
    run(
        python_command(
            SRC / "paper_scripts" / "make_audit_figures.py",
            "--output-dir",
            str(figure_dir),
        ),
        env=env,
    )


def paper() -> None:
    if shutil.which("latexmk") is None:
        raise RuntimeError("latexmk is required for paper compilation.")
    paper_build = BUILD / "paper"
    paper_build.mkdir(parents=True, exist_ok=True)
    common = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={paper_build}",
    ]
    run([*common, "main.tex"], cwd=PAPER)
    run([*common, "supplementary.tex"], cwd=PAPER)


def clean() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    for cache_dir in list(ROOT.rglob("__pycache__")):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for pattern in ("*.pyc", "*.pyo", ".DS_Store"):
        for path in list(ROOT.rglob(pattern)):
            if path.is_file() or path.is_symlink():
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        choices=("validate", "smoke", "figures", "paper", "all", "clean"),
    )
    args = parser.parse_args()
    if args.target == "clean":
        clean()
        return 0
    BUILD.mkdir(parents=True, exist_ok=True)
    if args.target in {"validate", "all"}:
        validate()
    if args.target in {"smoke", "all"}:
        smoke()
    if args.target in {"figures", "all"}:
        figures()
    if args.target in {"paper", "all"}:
        paper()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
