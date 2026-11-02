from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import get_task_loaders
from kfwa.metrics import classification_loss
from kfwa.models import build_model
from kfwa.training import move_batch, unpack_model_output
from kfwa.utils import count_parameters, get_device, set_seed

from run_experiment import add_args

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    model: str
    lop_oscillators: int | None = None


DEFAULT_SPECS = [
    BenchmarkSpec("transformer", "symbolic_transformer"),
    BenchmarkSpec("kfwa_no_adapter", "symbolic_kfwa_no_adapter"),
    BenchmarkSpec("lop_no_kuramoto_m32", "symbolic_kfwa_lop_features_no_kuramoto", 32),
    BenchmarkSpec("lop_no_kuramoto_m64", "symbolic_kfwa_lop_features_no_kuramoto", 64),
    BenchmarkSpec("lop_no_kuramoto_m96", "symbolic_kfwa_lop_features_no_kuramoto", 96),
    BenchmarkSpec(
        "lop_no_kuramoto_order_only_m64",
        "symbolic_kfwa_lop_features_no_kuramoto_global_order_only",
        64,
    ),
]


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--forward-iters", type=int, default=100)
    parser.add_argument("--train-warmup", type=int, default=5)
    parser.add_argument("--train-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="kfwa_experiments/runtime_benchmarks")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--spec",
        action="append",
        choices=[spec.label for spec in DEFAULT_SPECS],
        help="Benchmark only selected spec labels. Repeat for multiple labels.",
    )


def make_model_args(spec: BenchmarkSpec, base_args: argparse.Namespace) -> argparse.Namespace:
    args = deepcopy(base_args)
    args.model = spec.model
    if spec.lop_oscillators is not None:
        args.lop_oscillators = spec.lop_oscillators
    return args


def parse_base_model_args(seed: int, batch_size: int, device: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_args(parser)
    args = parser.parse_args(
        [
            "--task",
            "symbolic_rule_mapping",
            "--model",
            "symbolic_transformer",
            "--seed",
            str(seed),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ]
    )
    args.model_dim = 64
    args.layers = 2
    args.heads = 4
    args.ff_dim = 128
    args.oscillator_dim = 8
    args.kuramoto_steps = 2
    args.lop_steps = 16
    args.lop_oscillators = 64
    args.train_limit = max(batch_size * 4, 1024)
    args.val_limit = max(batch_size, 256)
    args.test_limit = max(batch_size, 256)
    args.num_workers = 0
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def current_device_memory_mb(device: torch.device) -> float:
    try:
        if device.type == "cuda":
            return float(torch.cuda.memory_allocated(device)) / (1024.0**2)
        if device.type == "mps" and hasattr(torch, "mps"):
            return float(torch.mps.current_allocated_memory()) / (1024.0**2)
    except Exception:
        return float("nan")
    return float("nan")


def cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def timed_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            logits, _ = unpack_model_output(model(batch))
            _ = classification_loss(logits, target)
    synchronize(device)

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            logits, _ = unpack_model_output(model(batch))
            _ = classification_loss(logits, target)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed, elapsed / max(iters, 1)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    logits, aux_loss = unpack_model_output(model(batch))
    loss = classification_loss(logits, target)
    if aux_loss is not None:
        loss = loss + aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.detach()


def timed_train(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(warmup):
        _ = train_step(model, optimizer, batch, target)
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = train_step(model, optimizer, batch, target)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed, elapsed / max(iters, 1)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for label, group in grouped.items():
        forward_ms = [float(row["forward_ms_per_batch"]) for row in group]
        train_ms = [float(row["train_ms_per_batch"]) for row in group]
        forward_sps = [float(row["forward_samples_per_sec"]) for row in group]
        train_sps = [float(row["train_samples_per_sec"]) for row in group]
        memory_mb = [float(row["device_allocated_after_bench_mb"]) for row in group]
        first = group[0]
        summaries.append(
            {
                "label": label,
                "model": first["model"],
                "lop_oscillators": first["lop_oscillators"],
                "parameters": first["parameters"],
                "param_mb_fp32": first["param_mb_fp32"],
                "batch_size": first["batch_size"],
                "device": first["device"],
                "repeats": len(group),
                "forward_ms_mean": mean(forward_ms),
                "forward_ms_std": stdev(forward_ms) if len(forward_ms) > 1 else 0.0,
                "forward_samples_per_sec_mean": mean(forward_sps),
                "train_ms_mean": mean(train_ms),
                "train_ms_std": stdev(train_ms) if len(train_ms) > 1 else 0.0,
                "train_samples_per_sec_mean": mean(train_sps),
                "device_allocated_after_bench_mb_mean": mean(memory_mb)
                if all(math.isfinite(value) for value in memory_mb)
                else float("nan"),
            }
        )

    baseline = next((row for row in summaries if row["label"] == "transformer"), None)
    if baseline is not None:
        for row in summaries:
            row["forward_time_vs_transformer"] = row["forward_ms_mean"] / baseline["forward_ms_mean"]
            row["train_time_vs_transformer"] = row["train_ms_mean"] / baseline["train_ms_mean"]
            row["param_vs_transformer"] = int(row["parameters"]) / int(baseline["parameters"])
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                safe_row[key] = None
            else:
                safe_row[key] = value
        safe_rows.append(safe_row)
    return safe_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark symbolic model runtime on a fixed batch.")
    add_benchmark_args(parser)
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.forward_iters < 1 or args.train_iters < 1:
        raise ValueError("--forward-iters and --train-iters must be positive.")

    set_seed(args.seed)
    device = get_device(args.device)
    base_model_args = parse_base_model_args(args.seed, args.batch_size, str(device))
    bundle = get_task_loaders(
        task="symbolic_rule_mapping",
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        train_limit=base_model_args.train_limit,
        val_limit=base_model_args.val_limit,
        test_limit=base_model_args.test_limit,
        num_symbols=base_model_args.num_symbols,
        support_min_k=base_model_args.support_min_k,
        support_max_k=base_model_args.support_max_k,
        rule_families=base_model_args.rule_families,
        holdout_rule_frac=base_model_args.holdout_rule_frac,
        query_out_of_support_prob=base_model_args.query_out_of_support_prob,
        support_noise=base_model_args.support_noise,
    )
    batch_cpu, target_cpu = next(iter(bundle.train))
    batch = move_batch(batch_cpu, device)
    target = target_cpu.to(device)

    selected_labels = set(args.spec or [spec.label for spec in DEFAULT_SPECS])
    specs = [spec for spec in DEFAULT_SPECS if spec.label in selected_labels]
    out_dir = Path(args.out_dir)
    run_dir = out_dir / datetime.now().strftime("symbolic_runtime_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    iterator = specs
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(specs, desc="benchmark specs", unit="model", dynamic_ncols=True, file=sys.stdout)

    for spec in iterator:
        repeat_iter = range(args.repeats)
        if not args.no_progress and tqdm is not None:
            repeat_iter = tqdm(
                repeat_iter,
                desc=spec.label,
                unit="repeat",
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout,
            )
        for repeat in repeat_iter:
            cleanup_device(device)
            model_args = make_model_args(spec, base_model_args)
            set_seed(args.seed + repeat)
            model = build_model(model_args, context_dim=bundle.context_dim, target_dim=bundle.target_dim).to(device)
            parameters = count_parameters(model)
            synchronize(device)
            allocated_after_model = current_device_memory_mb(device)

            forward_elapsed, forward_per_batch = timed_forward(
                model,
                batch,
                target,
                device,
                warmup=args.warmup,
                iters=args.forward_iters,
            )
            train_elapsed, train_per_batch = timed_train(
                model,
                batch,
                target,
                device,
                warmup=args.train_warmup,
                iters=args.train_iters,
            )
            allocated_after_bench = current_device_memory_mb(device)
            rows.append(
                {
                    "label": spec.label,
                    "model": spec.model,
                    "lop_oscillators": spec.lop_oscillators if spec.lop_oscillators is not None else "",
                    "repeat": repeat,
                    "device": str(device),
                    "batch_size": args.batch_size,
                    "parameters": parameters,
                    "param_mb_fp32": parameters * 4.0 / (1024.0**2),
                    "forward_total_sec": forward_elapsed,
                    "forward_ms_per_batch": forward_per_batch * 1000.0,
                    "forward_samples_per_sec": args.batch_size / forward_per_batch,
                    "train_total_sec": train_elapsed,
                    "train_ms_per_batch": train_per_batch * 1000.0,
                    "train_samples_per_sec": args.batch_size / train_per_batch,
                    "device_allocated_after_model_mb": allocated_after_model,
                    "device_allocated_after_bench_mb": allocated_after_bench,
                }
            )
            del model
            cleanup_device(device)

    summary_rows = summarize(rows)
    raw_path = run_dir / "raw_runtime.csv"
    summary_path = run_dir / "summary.csv"
    json_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary_rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary_rows), handle, indent=2, sort_keys=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "batch_size": args.batch_size,
                "device": str(device),
                "forward_iters": args.forward_iters,
                "train_iters": args.train_iters,
                "repeats": args.repeats,
                "seed": args.seed,
                "warmup": args.warmup,
                "train_warmup": args.train_warmup,
                "task": "symbolic_rule_mapping",
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote raw runtime to {raw_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
