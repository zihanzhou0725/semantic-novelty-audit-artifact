#!/usr/bin/env python3
"""Train and evaluate one frozen Boolean ICL behavioral run.

The adapter consumes boolean-behavioral-v1 manifests. It does not use the
upstream training entry point because that entry point does not expose exact
function identities, fixed evaluation prompts, or a function-disjoint sampler.
The model and task equations mirror the released implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from freeze_boolean_behavioral_protocol import (
    EVAL_FUNCTION_DRAWS,
    N_POINTS,
    SCHEMA_VERSION,
    TAIL_START_ZERO_INDEXED,
    TASKS,
    THRESHOLD_70,
    TRAIN_BATCH_SIZE,
    TRAIN_FUNCTION_DRAWS,
    TRAIN_STEPS,
    decode_parameter,
    ids_sha256,
    replace_excluded_slots,
    sample_ids,
    semantic_id,
    sha256_file,
)
from validate_boolean_behavioral_protocol import (
    load_eval_rows,
    validate_checksums,
    validate_source_hashes,
)


MODE_CONFIG = {
    "smoke": {"steps": 2, "eval_examples": 8, "eligible_for_gate": False},
    "canary": {"steps": 200, "eval_examples": 128, "eligible_for_gate": False},
    "full": {
        "steps": TRAIN_STEPS,
        "eval_examples": EVAL_FUNCTION_DRAWS,
        "eligible_for_gate": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def step_seed(base_seed: int, step: int) -> int:
    return splitmix64(base_seed ^ splitmix64(step))


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    return device


def seed_model(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenTransformer(nn.Module):
    """Released TransformerModel architecture with a lazy Transformers import."""

    def __init__(
        self,
        *,
        n_dims: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
    ) -> None:
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2Model
        except ImportError as error:
            raise SystemExit(
                "Missing dependency 'transformers'. Install the adapter requirements "
                "before running model modes."
            ) from error

        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        batch_size, points, dimensions = xs.shape
        ys_wide = torch.cat(
            (
                ys.reshape(batch_size, points, 1),
                torch.zeros(
                    batch_size,
                    points,
                    dimensions - 1,
                    device=ys.device,
                    dtype=ys.dtype,
                ),
            ),
            dim=2,
        )
        return torch.stack((xs, ys_wide), dim=2).reshape(
            batch_size, 2 * points, dimensions
        )

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        embeddings = self._read_in(self._combine(xs, ys))
        hidden = self._backbone(inputs_embeds=embeddings).last_hidden_state
        return self._read_out(hidden)[:, ::2, 0]


def decode_weights(task_name: str, ids: np.ndarray) -> torch.Tensor:
    n_dims = TASKS[task_name]["n_dims"]
    weights = np.zeros((len(ids), n_dims), dtype=np.float32)
    if task_name == "conjunction_d28":
        remaining = ids.astype(np.uint64).copy()
        mapping = np.asarray((0.0, 1.0, -1.0), dtype=np.float32)
        for index in range(n_dims):
            weights[:, index] = mapping[(remaining % 3).astype(np.int8)]
            remaining //= 3
    else:
        for index in range(n_dims):
            weights[:, index] = ((ids >> np.uint64(index)) & np.uint64(1))
    return torch.from_numpy(weights)


def generate_episode_batch(
    task_name: str,
    ids: np.ndarray,
    *,
    n_points: int,
    numpy_seed: int,
    torch_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = decode_weights(task_name, ids)
    batch_size, n_dims = weights.shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(torch_seed)
    xs = (
        torch.randint(
            0,
            2,
            (batch_size, n_points, n_dims),
            generator=generator,
            dtype=torch.int8,
        ).float()
        * 2
        - 1
    )

    if task_name in {
        "conjunction_d28",
        "sparse_disjunction_d50_k4_positive",
    }:
        rng = np.random.Generator(np.random.PCG64(numpy_seed))
        balance = rng.bit_generator.random_raw((batch_size, n_points)) >= THRESHOLD_70
        for batch_index in range(batch_size):
            selected_points = np.flatnonzero(balance[batch_index])
            active = torch.nonzero(weights[batch_index], as_tuple=False).flatten()
            if not len(selected_points) or not len(active):
                continue
            point_index = torch.as_tensor(selected_points, dtype=torch.long)
            if task_name == "conjunction_d28":
                xs[batch_index, point_index[:, None], active[None, :]] = weights[
                    batch_index, active
                ]
            else:
                xs[batch_index, point_index[:, None], active[None, :]] = -1.0

    scores = torch.einsum("bpd,bd->bp", xs, weights)
    if task_name == "conjunction_d28":
        threshold = weights.abs().sum(dim=1, keepdim=True) - 1
        ys = torch.sign(scores - threshold)
    elif task_name == "majority_d28":
        ys = torch.sign(scores - 0.5)
    elif task_name == "sparse_disjunction_d50_k4_positive":
        threshold = weights.abs().sum(dim=1, keepdim=True) - 1
        ys = torch.sign(scores + threshold)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    if not torch.all((ys == -1) | (ys == 1)):
        raise AssertionError(f"{task_name} generated a non-binary label")
    return xs, ys


def generate_fixed_eval_batch(
    task_name: str, rows: list[dict]
) -> tuple[torch.Tensor, torch.Tensor]:
    xs_parts = []
    ys_parts = []
    for row in rows:
        ids = np.asarray([row["encoded_semantic_id"]], dtype=np.uint64)
        xs, ys = generate_episode_batch(
            task_name,
            ids,
            n_points=N_POINTS,
            numpy_seed=int(row["numpy_prompt_seed"]),
            torch_seed=int(row["torch_prompt_seed"]),
        )
        xs_parts.append(xs)
        ys_parts.append(ys)
    return torch.cat(xs_parts), torch.cat(ys_parts)


def load_run_row(run_manifest: Path, run_id: str) -> dict:
    with run_manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run_id"] == run_id]
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one manifest row for run_id={run_id}")
    return rows[0]


def reconstruct_function_stream(run_row: dict, eval_ids: np.ndarray) -> np.ndarray:
    task_name = run_row["task"]
    original = sample_ids(
        task_name,
        TRAIN_FUNCTION_DRAWS,
        int(run_row["train_function_seed"]),
    )
    if run_row["protocol"] == "original":
        result = original
    elif run_row["protocol"] == "strict_function_disjoint":
        result, replacement_draws = replace_excluded_slots(
            task_name,
            original,
            np.unique(eval_ids),
            int(run_row["replacement_seed"]),
        )
        if replacement_draws != int(run_row["replacement_candidate_draw_count"]):
            raise SystemExit("Strict replacement draw count differs from run manifest")
        if np.any(np.isin(result, np.unique(eval_ids))):
            raise SystemExit("Strict function stream overlaps frozen eval semantics")
    else:
        raise SystemExit(f"Unknown protocol: {run_row['protocol']}")

    observed = ids_sha256(result)
    if observed != run_row["function_stream_sha256"]:
        raise SystemExit(
            f"Function stream checksum mismatch: {observed} != "
            f"{run_row['function_stream_sha256']}"
        )
    return result


def load_protocol(root: Path, run_id: str) -> tuple[dict, dict, list[dict], np.ndarray]:
    protocol_dir = root / "protocols" / SCHEMA_VERSION
    validate_checksums(protocol_dir)
    validate_source_hashes(root)
    preregistration = json.loads(
        (protocol_dir / "preregistration.json").read_text()
    )
    split_manifest = json.loads((protocol_dir / "split_manifest.json").read_text())
    if preregistration["status"] != "frozen_before_training":
        raise SystemExit("Protocol is not marked frozen_before_training")
    if split_manifest["protocol_generator_sha256"] != sha256_file(
        root / "scripts" / "freeze_boolean_behavioral_protocol.py"
    ):
        raise SystemExit("Protocol generator checksum does not match split manifest")
    if split_manifest["training_adapter_sha256"] != sha256_file(
        Path(__file__).resolve()
    ):
        raise SystemExit("Training adapter checksum does not match split manifest")
    if split_manifest["behavioral_requirements_sha256"] != sha256_file(
        root / "behavioral_requirements.txt"
    ):
        raise SystemExit("Behavioral requirements checksum does not match split manifest")

    run_row = load_run_row(protocol_dir / "run_manifest.csv", run_id)
    rows_by_task = load_eval_rows(protocol_dir / "eval_functions.jsonl")
    eval_rows = rows_by_task[run_row["task"]]
    eval_ids = np.asarray(
        [row["encoded_semantic_id"] for row in eval_rows], dtype=np.uint64
    )
    eval_summary = split_manifest["task_eval_summaries"][run_row["task"]]
    if ids_sha256(eval_ids) != eval_summary["ordered_semantic_ids_sha256"]:
        raise SystemExit("Frozen evaluation semantic IDs do not match split manifest")
    for row in eval_rows:
        encoded = int(row["encoded_semantic_id"])
        if row["semantic_id"] != semantic_id(run_row["task"], encoded):
            raise SystemExit("Evaluation semantic ID text does not match encoding")
        if row["parameter"] != decode_parameter(run_row["task"], encoded):
            raise SystemExit("Evaluation parameter does not match semantic ID")

    return preregistration, run_row, eval_rows, eval_ids


def build_model(preregistration: dict, n_dims: int) -> nn.Module:
    model_spec = preregistration["model"]
    if model_spec["family"] != "san":
        raise SystemExit(f"Unsupported frozen model family: {model_spec['family']}")
    return FrozenTransformer(
        n_dims=n_dims,
        n_positions=model_spec["n_positions"],
        n_embd=model_spec["n_embd"],
        n_layer=model_spec["n_layer"],
        n_head=model_spec["n_head"],
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    run_id: str,
    mode: str,
    frozen_hashes: dict,
) -> None:
    atomic_torch_save(
        path,
        {
            "run_id": run_id,
            "mode": mode,
            "completed_steps": completed_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "frozen_hashes": frozen_hashes,
        },
    )


def train(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_row: dict,
    function_stream: np.ndarray,
    device: torch.device,
    target_steps: int,
    output_dir: Path,
    checkpoint_every: int,
    resume: bool,
    frozen_hashes: dict,
    mode: str,
) -> tuple[int, float]:
    state_path = output_dir / "state.pt"
    starting_step = 0
    if resume:
        if not state_path.exists():
            raise SystemExit(f"Cannot resume; checkpoint is missing: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["run_id"] != run_row["run_id"] or state["mode"] != mode:
            raise SystemExit("Checkpoint run identity does not match requested run")
        if state["frozen_hashes"] != frozen_hashes:
            raise SystemExit("Checkpoint was created from different frozen artifacts")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(next(model.parameters()).device)
        starting_step = int(state["completed_steps"])
    elif state_path.exists():
        raise SystemExit(
            f"Checkpoint already exists at {state_path}; pass --resume or use a "
            "different output root"
        )

    log_path = output_dir / "training_log.jsonl"
    log_mode = "a" if resume else "w"
    start_time = time.monotonic()
    model.train()
    with log_path.open(log_mode) as log_handle:
        for step in range(starting_step, target_steps):
            start = step * TRAIN_BATCH_SIZE
            stop = start + TRAIN_BATCH_SIZE
            ids = function_stream[start:stop]
            xs, ys = generate_episode_batch(
                run_row["task"],
                ids,
                n_points=N_POINTS,
                numpy_seed=step_seed(int(run_row["train_numpy_input_seed"]), step),
                torch_seed=step_seed(int(run_row["train_torch_input_seed"]), step),
            )
            xs = xs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xs, ys)
            targets = (ys + 1) / 2
            loss = F.binary_cross_entropy(torch.sigmoid(logits), targets)
            loss.backward()
            optimizer.step()

            completed_steps = step + 1
            if completed_steps == 1 or completed_steps % 25 == 0:
                elapsed = time.monotonic() - start_time
                record = {
                    "completed_steps": completed_steps,
                    "loss": float(loss.detach().cpu()),
                    "elapsed_seconds_since_start": elapsed,
                    "steps_per_second_since_start": (
                        (completed_steps - starting_step) / elapsed
                    ),
                }
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                log_handle.flush()
                print(
                    f"{run_row['run_id']} {completed_steps}/{target_steps} "
                    f"loss={record['loss']:.6f} "
                    f"steps/s={record['steps_per_second_since_start']:.2f}",
                    flush=True,
                )
            if completed_steps % checkpoint_every == 0:
                save_checkpoint(
                    state_path,
                    model=model,
                    optimizer=optimizer,
                    completed_steps=completed_steps,
                    run_id=run_row["run_id"],
                    mode=mode,
                    frozen_hashes=frozen_hashes,
                )

    elapsed = time.monotonic() - start_time
    save_checkpoint(
        state_path,
        model=model,
        optimizer=optimizer,
        completed_steps=target_steps,
        run_id=run_row["run_id"],
        mode=mode,
        frozen_hashes=frozen_hashes,
    )
    return starting_step, elapsed


def evaluate(
    *,
    model: nn.Module,
    task_name: str,
    eval_rows: list[dict],
    eval_ids: np.ndarray,
    original_train_ids: np.ndarray,
    protocol: str,
    device: torch.device,
    batch_size: int,
    output_path: Path,
) -> dict:
    original_seen = set(int(value) for value in np.unique(original_train_ids))
    totals = {
        "tail_accuracy": 0.0,
        "final_position_accuracy": 0.0,
        "all_positions_1_through_39_accuracy": 0.0,
        "tail_binary_cross_entropy": 0.0,
    }
    per_episode = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_rows), batch_size):
            batch_rows = eval_rows[start : start + batch_size]
            xs, ys = generate_fixed_eval_batch(task_name, batch_rows)
            logits = model(xs.to(device), ys.to(device)).cpu()
            correct = (torch.sign(logits) == ys).float()
            targets = (ys + 1) / 2
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            for local_index, row in enumerate(batch_rows):
                encoded = int(row["encoded_semantic_id"])
                tail_accuracy = float(
                    correct[local_index, TAIL_START_ZERO_INDEXED:].mean()
                )
                final_accuracy = float(correct[local_index, -1])
                all_accuracy = float(correct[local_index, 1:].mean())
                tail_bce = float(
                    bce[local_index, TAIL_START_ZERO_INDEXED:].mean()
                )
                record = {
                    "run_protocol": protocol,
                    "task": task_name,
                    "eval_index": row["eval_index"],
                    "semantic_id": row["semantic_id"],
                    "seen_in_original_training_stream": encoded in original_seen,
                    "seen_in_actual_training_stream": (
                        encoded in original_seen if protocol == "original" else False
                    ),
                    "tail_accuracy": tail_accuracy,
                    "final_position_accuracy": final_accuracy,
                    "all_positions_1_through_39_accuracy": all_accuracy,
                    "tail_binary_cross_entropy": tail_bce,
                    "labels": [int(value) for value in ys[local_index].tolist()],
                    "logits": [float(value) for value in logits[local_index].tolist()],
                }
                per_episode.append(record)
                for metric in totals:
                    totals[metric] += record[metric]

    with output_path.open("w") as handle:
        for row in per_episode:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metrics = {metric: total / len(per_episode) for metric, total in totals.items()}
    metrics.update(
        {
            "eval_episode_count": len(per_episode),
            "eval_unique_semantic_id_count": len(np.unique(eval_ids[: len(per_episode)])),
            "original_exposed_eval_episode_count": sum(
                row["seen_in_original_training_stream"] for row in per_episode
            ),
            "actual_exposed_eval_episode_count": sum(
                row["seen_in_actual_training_stream"] for row in per_episode
            ),
        }
    )
    return metrics


def package_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "not-installed"
    return versions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "runs" / SCHEMA_VERSION,
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive")

    preregistration, run_row, eval_rows, eval_ids = load_protocol(
        root, args.run_id
    )
    function_stream = reconstruct_function_stream(run_row, eval_ids)
    original_train_ids = sample_ids(
        run_row["task"],
        TRAIN_FUNCTION_DRAWS,
        int(run_row["train_function_seed"]),
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "run_id": args.run_id,
                    "protocol": run_row["protocol"],
                    "function_stream_sha256": ids_sha256(function_stream),
                    "strict_eval_overlap_count": (
                        int(np.isin(function_stream, np.unique(eval_ids)).sum())
                        if run_row["protocol"] == "strict_function_disjoint"
                        else None
                    ),
                    "training_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    mode_config = MODE_CONFIG[args.mode]
    target_steps = mode_config["steps"]
    selected_eval_rows = eval_rows[: mode_config["eval_examples"]]
    selected_eval_ids = eval_ids[: mode_config["eval_examples"]]
    device = resolve_device(args.device)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    model_seed = int(run_row["model_seed"])
    seed_model(model_seed)
    model = build_model(
        preregistration, TASKS[run_row["task"]]["n_dims"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=preregistration["training"]["learning_rate"],
        betas=tuple(preregistration["training"]["betas"]),
        eps=preregistration["training"]["epsilon"],
        weight_decay=preregistration["training"]["weight_decay"],
    )

    protocol_dir = root / "protocols" / SCHEMA_VERSION
    frozen_hashes = {
        filename: digest
        for digest, filename in (
            line.split("  ", maxsplit=1)
            for line in (protocol_dir / "SHA256SUMS").read_text().splitlines()
        )
    }
    output_dir = args.output_root / args.mode / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "COMPLETED.json"
    if completed_path.exists():
        raise SystemExit(f"Run is already complete: {completed_path}")

    started_at = utc_now()
    start_step, train_seconds = train(
        model=model,
        optimizer=optimizer,
        run_row=run_row,
        function_stream=function_stream,
        device=device,
        target_steps=target_steps,
        output_dir=output_dir,
        checkpoint_every=min(args.checkpoint_every, target_steps),
        resume=args.resume,
        frozen_hashes=frozen_hashes,
        mode=args.mode,
    )
    predictions_path = output_dir / "eval_predictions.jsonl"
    eval_started = time.monotonic()
    metrics = evaluate(
        model=model,
        task_name=run_row["task"],
        eval_rows=selected_eval_rows,
        eval_ids=selected_eval_ids,
        original_train_ids=original_train_ids,
        protocol=run_row["protocol"],
        device=device,
        batch_size=args.eval_batch_size,
        output_path=predictions_path,
    )
    eval_seconds = time.monotonic() - eval_started
    state_path = output_dir / "state.pt"
    metadata = {
        "status": "complete",
        "eligible_for_gate": mode_config["eligible_for_gate"],
        "run_id": args.run_id,
        "mode": args.mode,
        "started_at": started_at,
        "completed_at": utc_now(),
        "device": str(device),
        "package_versions": package_versions(),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_hashes": frozen_hashes,
        "run_manifest_row": run_row,
        "execution": {
            "starting_step": start_step,
            "completed_steps": target_steps,
            "training_seconds_this_invocation": train_seconds,
            "evaluation_seconds": eval_seconds,
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "artifacts": {
            "checkpoint": state_path.name,
            "checkpoint_sha256": sha256_file(state_path),
            "predictions": predictions_path.name,
            "predictions_sha256": sha256_file(predictions_path),
        },
        "metrics": metrics,
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    atomic_write_json(completed_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
