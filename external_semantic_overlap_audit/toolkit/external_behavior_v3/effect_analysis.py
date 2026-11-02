"""Frozen CLUTRR V3 exposure-effect analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ID = "external-behavior-v3-effect-analysis-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v3"
    / "effect-analysis-v1"
)
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v3"
    / "effect-analysis-v1"
)
DESIGN_DIR = PROJECT_ROOT / "manifests" / "external-behavior-v3"
DATA_DIR = DESIGN_DIR / "data"
EXECUTION_DIR = DESIGN_DIR / "execution-v1"
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "external-behavior-v3"
    / "execution-v1"
)
MODEL_SEEDS = tuple(range(8))
ARMS = ("A", "B")
PAIR_COUNT = 20
PATTERN_COUNT = 40
GROUPS_PER_PATTERN = 34
ROWS_PER_PATTERN = 200
EVAL_ROWS = 8_000
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_261_012
BOOTSTRAP_BATCH_SIZE = 128

IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v3_effect_analysis.py",
    "scripts/analyze_external_behavior_v3_effect.py",
    "scripts/validate_external_behavior_v3_effect.py",
    "tests/test_external_behavior_v3_effect_analysis.py",
    "toolkit/external_behavior_v3/effect_analysis.py",
    "protocols/external-behavior-v3/effect-analysis-v1/validate.py",
)

OUTPUT_NAMES = (
    "bootstrap.npz",
    "pair_effects.csv",
    "pattern_effects.csv",
    "relation_effects.csv",
    "report_cn.md",
    "results.json",
    "seed_aligned_metrics.csv",
    "seed_effects.csv",
    "seed_pattern_effects.csv",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            list(value.shape),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                yield value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deterministic_npz_bytes(arrays))


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _input_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT
        / "protocols"
        / "external-behavior-v3"
        / "preregistration.json",
        PROJECT_ROOT
        / "protocols"
        / "external-behavior-v3"
        / "execution-v1"
        / "preregistration.json",
        DESIGN_DIR / "design_validation_report.json",
        EXECUTION_DIR / "execution_manifest.json",
        EXECUTION_DIR / "execution_validation_report.json",
        EXECUTION_DIR / "EXECUTION_SHA256SUMS",
        EXECUTION_DIR / "run_artifact_manifest.json",
        EXECUTION_DIR / "run_manifest.csv",
        EXECUTION_DIR / "evaluation_manifest.npz",
        DATA_DIR / "clutrr_v3_pairs.jsonl",
        DATA_DIR / "clutrr_v3_evaluation.jsonl",
        DATA_DIR / "clutrr_v3_exposure.jsonl",
    ]
    run_rows = read_csv(EXECUTION_DIR / "run_manifest.csv")
    for row in run_rows:
        paths.append(DATA_DIR / row["train_manifest"])
        paths.append(
            RUN_ROOT / row["run_id"] / "eval_predictions.npz"
        )
    return sorted(set(paths), key=relative)


def _implementation_hashes() -> dict[str, str]:
    missing = [
        name
        for name in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing analysis implementation: {missing}")
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in IMPLEMENTATION_PATHS
    }


def _verify_run_prediction_hashes() -> None:
    manifest = read_json(EXECUTION_DIR / "run_artifact_manifest.json")
    records = manifest.get("runs")
    if not isinstance(records, list) or len(records) != 16:
        raise RuntimeError("Run artifact manifest is incomplete.")
    for record in records:
        run_id = record["run_id"]
        expected = record["artifacts_sha256"]["eval_predictions.npz"]
        path = RUN_ROOT / run_id / "eval_predictions.npz"
        if sha256_file(path) != expected:
            raise RuntimeError(f"Prediction hash mismatch: {run_id}")


def _checksum_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        digest, name = line.split("  ", 1)
        entries.append((digest, path.parent / name))
    return entries


def verify_protocol_bundle() -> None:
    checksum_path = PROTOCOL_DIR / "SHA256SUMS"
    for expected, path in _checksum_entries(checksum_path):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Protocol checksum mismatch: {path}")
    report = read_json(PROTOCOL_DIR / "validation_report.json")
    if (
        report.get("status") != "pass"
        or report.get("stage") != "pre_unblinding"
        or report.get("failed_count") != 0
    ):
        raise RuntimeError("Pre-unblinding validation did not pass.")


def verify_frozen_inputs() -> None:
    manifest = read_json(PROTOCOL_DIR / "input_manifest.json")
    for name, expected in manifest["input_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen analysis input changed: {name}")
    for name, expected in manifest["implementation_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen analysis code changed: {name}")


def validate_protocol(*, require_outcomes_absent: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    execution = read_json(EXECUTION_DIR / "execution_validation_report.json")
    input_manifest_path = PROTOCOL_DIR / "input_manifest.json"
    receipt_path = PROTOCOL_DIR / "freeze_receipt.json"
    check(
        "execution_gate",
        execution.get("status") == "pass"
        and execution.get("execution_gate") == "pass"
        and execution.get("failed_count") == 0,
    )
    check(
        "execution_blinded",
        execution.get("primary_effect_computed") is False,
    )
    check(
        "frozen_parent_estimand",
        prereg["estimand"]["unit"] == "percentage_points"
        and prereg["estimand"]["orientation"] == "exposed_minus_unexposed"
        and prereg["estimand"]["seed_count"] == 8
        and prereg["estimand"]["pattern_count"] == 40,
    )
    bootstrap = prereg["bootstrap"]
    check(
        "bootstrap",
        bootstrap["replicates"] == BOOTSTRAP_REPLICATES
        and bootstrap["seed"] == BOOTSTRAP_SEED
        and bootstrap["model_seed_draws_per_replicate"] == 8
        and bootstrap["pattern_pair_draws_per_replicate"] == 20
        and bootstrap["source_group_draws_per_pattern"] == 34,
    )
    gate = prereg["decision_gate"]
    check(
        "gate",
        gate["sesoi_pp"] == 1.0
        and gate["equivalence_margin_pp"] == [-0.5, 0.5]
        and gate["maximum_interval_width_pp"] == 2.0,
    )
    check(
        "input_manifest_exists",
        input_manifest_path.is_file() and receipt_path.is_file(),
    )
    input_failures: list[str] = []
    implementation_failures: list[str] = []
    if input_manifest_path.is_file():
        manifest = read_json(input_manifest_path)
        for name, expected in manifest["input_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                input_failures.append(name)
        for name, expected in manifest["implementation_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                implementation_failures.append(name)
    check("input_hashes", not input_failures, input_failures)
    check(
        "implementation_hashes",
        not implementation_failures,
        implementation_failures,
    )
    artifacts = (
        list(ANALYSIS_DIR.rglob("*")) if ANALYSIS_DIR.exists() else []
    )
    if require_outcomes_absent:
        check(
            "outcomes_absent",
            not artifacts,
            [relative(path) for path in artifacts],
        )
    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256_file(PROTOCOL_DIR / name)
            for name in document_names
        },
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v3-effect-protocol-validation-v1"
        ),
        "stage": (
            "pre_unblinding"
            if require_outcomes_absent
            else "post_unblinding_recheck"
        ),
        "status": "pass" if not failed else "fail",
        "validated_at_utc": _now(),
    }


def freeze_protocol() -> dict[str, Any]:
    if (PROTOCOL_DIR / "input_manifest.json").exists():
        raise RuntimeError("Effect-analysis protocol is already frozen.")
    if ANALYSIS_DIR.exists() and any(ANALYSIS_DIR.rglob("*")):
        raise RuntimeError("Outcome artifacts exist before protocol freeze.")
    execution = read_json(EXECUTION_DIR / "execution_validation_report.json")
    if (
        execution.get("status") != "pass"
        or execution.get("failed_count") != 0
        or execution.get("primary_effect_computed") is not False
    ):
        raise RuntimeError("Execution is not eligible for unblinding.")
    _verify_run_prediction_hashes()
    inputs = _input_paths()
    missing = [relative(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing analysis inputs: {missing}")
    input_manifest = {
        "analysis_id": ANALYSIS_ID,
        "frozen_at_utc": _now(),
        "implementation_sha256": _implementation_hashes(),
        "input_sha256": {
            relative(path): sha256_file(path) for path in inputs
        },
        "prediction_file_count": 16,
        "schema_version": (
            "external-behavior-v3-effect-input-manifest-v1"
        ),
        "status": "frozen_before_outcome_contrast_computation",
    }
    write_json(PROTOCOL_DIR / "input_manifest.json", input_manifest)
    receipt = {
        "analysis_id": ANALYSIS_ID,
        "frozen_at_utc": _now(),
        "schema_version": (
            "external-behavior-v3-effect-freeze-receipt-v1"
        ),
        "state_at_freeze": {
            "analysis_output_directory_absent": not ANALYSIS_DIR.exists(),
            "execution_gate_passed": True,
            "outcome_contrast_computed": False,
            "outcome_values_loaded_by_freeze_process": False,
            "prediction_files_accessed_for_sha256_only": True,
            "run_artifact_hashes_validated": True,
        },
        "status": "frozen_before_unblinding",
    }
    write_json(PROTOCOL_DIR / "freeze_receipt.json", receipt)
    report = validate_protocol(require_outcomes_absent=True)
    if report["status"] != "pass":
        raise RuntimeError(f"Pre-unblinding validation failed: {report}")
    write_json(PROTOCOL_DIR / "validation_report.json", report)
    checksum_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
        "validation_report.json",
    )
    (PROTOCOL_DIR / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(PROTOCOL_DIR / name)}  {name}\n"
            for name in checksum_names
        ),
        encoding="ascii",
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "protocol_sha256s_sha256": sha256_file(
            PROTOCOL_DIR / "SHA256SUMS"
        ),
        "status": "frozen_before_unblinding",
        "validation_status": report["status"],
    }


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != targets.shape[0]:
        raise ValueError("Invalid logits or labels.")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    predictions = values.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    for index in range(15):
        lower = index / 15
        upper = (index + 1) / 15
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if index == 14
            else (confidence >= lower) & (confidence < upper)
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean())
                - float(confidence[selected].mean())
            )
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    accuracy = float(correct.mean())
    mean_confidence = float(confidence.mean())
    return {
        "accuracy": accuracy,
        "brier_score": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "cross_entropy": float(
            (
                -shifted[np.arange(len(targets)), targets]
                + np.log(exponentials.sum(axis=1))
            ).mean()
        ),
        "ece_15": ece,
        "mean_confidence": mean_confidence,
        "signed_calibration_gap": mean_confidence - accuracy,
    }


def classify_gate(
    point_pp: float,
    lower_pp: float,
    upper_pp: float,
) -> dict[str, Any]:
    width_pp = upper_pp - lower_pp
    precision_pass = width_pp <= 2.0
    if not precision_pass:
        decision = "imprecise_no_go"
    elif point_pp >= 1.0 and lower_pp > 0.0:
        decision = "positive_behavioral_go"
    elif point_pp <= -1.0 and upper_pp < 0.0:
        decision = "negative_behavioral_go"
    elif lower_pp >= -0.5 and upper_pp <= 0.5:
        decision = "null_equivalence_go"
    else:
        decision = "inconclusive_no_go"
    return {
        "classification": decision,
        "interval_width_pp": width_pp,
        "precision_gate_pass": precision_pass,
    }


def hierarchical_bootstrap(
    group_difference_sums: np.ndarray,
    group_sizes: np.ndarray,
    pair_pattern_indices: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, str]]:
    differences = np.asarray(group_difference_sums, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.float64)
    pairs = np.asarray(pair_pattern_indices, dtype=np.int16)
    if differences.shape != (8, 40, 34):
        raise ValueError("Expected group differences with shape (8,40,34).")
    if sizes.shape != (40, 34) or pairs.shape != (20, 2):
        raise ValueError("Invalid group sizes or pair mapping.")
    if (sizes <= 0).any():
        raise ValueError("Source groups must be nonempty.")
    rng = np.random.Generator(np.random.PCG64(seed))
    seed_draws = rng.integers(
        0,
        8,
        size=(replicates, 8),
        dtype=np.int16,
    )
    pair_draws = rng.integers(
        0,
        20,
        size=(replicates, 20),
        dtype=np.int16,
    )
    group_draws = rng.integers(
        0,
        34,
        size=(replicates, 20, 2, 34),
        dtype=np.int16,
    )
    effects = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        selected_patterns = pairs[pair_draws[start:stop]].reshape(
            stop - start,
            40,
        )
        selected_groups = group_draws[start:stop].reshape(
            stop - start,
            40,
            34,
        )
        denominators = sizes[
            selected_patterns[:, :, None],
            selected_groups,
        ].sum(axis=2)
        real_seed_effects = np.empty(
            (stop - start, 8, 40),
            dtype=np.float64,
        )
        for model_seed in MODEL_SEEDS:
            numerators = differences[model_seed][
                selected_patterns[:, :, None],
                selected_groups,
            ].sum(axis=2)
            real_seed_effects[:, model_seed, :] = (
                numerators / denominators
            )
        selected_seed_effects = np.take_along_axis(
            real_seed_effects,
            seed_draws[start:stop, :, None],
            axis=1,
        )
        effects[start:stop] = (
            100.0 * selected_seed_effects.mean(axis=(1, 2))
        )
    draw_hashes = {
        "group_draws_sha256": array_sha256(group_draws),
        "pair_draws_sha256": array_sha256(pair_draws),
        "seed_draws_sha256": array_sha256(seed_draws),
    }
    return effects, draw_hashes


def _patterns() -> tuple[list[dict[str, Any]], np.ndarray]:
    pairs = sorted(
        iter_jsonl(DATA_DIR / "clutrr_v3_pairs.jsonl"),
        key=lambda row: int(row["pair_index"]),
    )
    if len(pairs) != PAIR_COUNT:
        raise RuntimeError("Frozen pair count changed.")
    patterns: list[dict[str, Any]] = []
    pair_pattern_indices = np.empty((20, 2), dtype=np.int16)
    for expected_pair, pair in enumerate(pairs):
        if int(pair["pair_index"]) != expected_pair:
            raise RuntimeError("Frozen pair order changed.")
        for member in (0, 1):
            pattern_index = len(patterns)
            pair_pattern_indices[expected_pair, member] = pattern_index
            patterns.append(
                {
                    "pair_index": expected_pair,
                    "pair_member": member,
                    "pattern_index": pattern_index,
                    "semantic_id": pair[f"member_{member}"],
                    "target_relation": pair["target_relation"],
                }
            )
    return patterns, pair_pattern_indices


def _exposed_arms(patterns: list[dict[str, Any]]) -> np.ndarray:
    exposure_by_row = {
        row["surface_id"]: row["semantic_id"]
        for row in iter_jsonl(DATA_DIR / "clutrr_v3_exposure.jsonl")
    }
    semantic_to_pattern = {
        row["semantic_id"]: row["pattern_index"] for row in patterns
    }
    result = np.empty((8, 40), dtype=np.int8)
    for model_seed in MODEL_SEEDS:
        counts_by_arm: dict[str, Counter[int]] = {}
        for arm in ARMS:
            path = (
                DATA_DIR
                / f"clutrr_train_seed{model_seed}_arm{arm}.txt"
            )
            counts: Counter[int] = Counter()
            for row_id in path.read_text(encoding="ascii").splitlines():
                semantic_id = exposure_by_row.get(row_id)
                if semantic_id in semantic_to_pattern:
                    counts[semantic_to_pattern[semantic_id]] += 1
            counts_by_arm[arm] = counts
        for pattern_index in range(40):
            counts = (
                counts_by_arm["A"][pattern_index],
                counts_by_arm["B"][pattern_index],
            )
            if counts == (64, 0):
                result[model_seed, pattern_index] = 0
            elif counts == (0, 64):
                result[model_seed, pattern_index] = 1
            else:
                raise RuntimeError(
                    "Frozen exposure is not complementary: "
                    f"seed={model_seed}, pattern={pattern_index}, "
                    f"counts={counts}"
                )
    return result


def _evaluation_groups(
    evaluation: dict[str, np.ndarray],
    patterns: list[dict[str, Any]],
) -> tuple[np.ndarray, list[list[np.ndarray]], np.ndarray]:
    pair_index = evaluation["pair_index"].astype(np.int64)
    pair_member = evaluation["pair_member"].astype(np.int64)
    pattern_by_row = pair_index * 2 + pair_member
    if (
        pattern_by_row.shape != (EVAL_ROWS,)
        or pattern_by_row.min() != 0
        or pattern_by_row.max() != 39
    ):
        raise RuntimeError("Invalid evaluation pattern mapping.")
    semantic_ids = evaluation["semantic_ids"].astype("U64")
    source_group_ids = evaluation["source_group_ids"].astype("U64")
    groups: list[list[np.ndarray]] = []
    group_sizes = np.zeros((40, 34), dtype=np.int16)
    for pattern in patterns:
        pattern_index = pattern["pattern_index"]
        rows = np.flatnonzero(pattern_by_row == pattern_index)
        if (
            len(rows) != ROWS_PER_PATTERN
            or set(semantic_ids[rows]) != {pattern["semantic_id"]}
        ):
            raise RuntimeError("Evaluation pattern rows changed.")
        group_ids = sorted(set(source_group_ids[rows]))
        if len(group_ids) != GROUPS_PER_PATTERN:
            raise RuntimeError("Evaluation source-group count changed.")
        pattern_groups: list[np.ndarray] = []
        for group_index, group_id in enumerate(group_ids):
            group_rows = rows[source_group_ids[rows] == group_id]
            if len(group_rows) not in (2, 6):
                raise RuntimeError("Unexpected evaluation group size.")
            pattern_groups.append(group_rows)
            group_sizes[pattern_index, group_index] = len(group_rows)
        if sorted(group_sizes[pattern_index].tolist()) != [2] + [6] * 33:
            raise RuntimeError("Frozen 200-row allocation changed.")
        groups.append(pattern_groups)
    return pattern_by_row, groups, group_sizes


def _load_predictions(
    evaluation: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.empty((8, 2, 8000, 23), dtype=np.float32)
    correct = np.empty((8, 2, 8000), dtype=np.bool_)
    required = {
        "labels",
        "logits",
        "pair_index",
        "pair_member",
        "predictions",
        "row_ids",
        "semantic_ids",
        "source_group_ids",
    }
    metadata_names = (
        "labels",
        "pair_index",
        "pair_member",
        "row_ids",
        "semantic_ids",
        "source_group_ids",
    )
    labels = evaluation["labels"].astype(np.int64)
    for model_seed in MODEL_SEEDS:
        for arm_index, arm in enumerate(ARMS):
            path = (
                RUN_ROOT
                / f"seed{model_seed}_arm{arm}"
                / "eval_predictions.npz"
            )
            prediction = load_npz(path)
            if (
                set(prediction) != required
                or prediction["logits"].shape != (8000, 23)
                or not np.isfinite(prediction["logits"]).all()
            ):
                raise RuntimeError(f"Invalid prediction artifact: {path}")
            for name in metadata_names:
                if not np.array_equal(prediction[name], evaluation[name]):
                    raise RuntimeError(
                        f"Prediction metadata mismatch: {path}:{name}"
                    )
            if not np.array_equal(
                prediction["predictions"],
                prediction["logits"].argmax(axis=1),
            ):
                raise RuntimeError(f"Stored predictions changed: {path}")
            logits[model_seed, arm_index] = prediction["logits"]
            correct[model_seed, arm_index] = (
                prediction["predictions"].astype(np.int64) == labels
            )
    return logits, correct


def _effect_tables(
    patterns: list[dict[str, Any]],
    pair_pattern_indices: np.ndarray,
    pattern_by_row: np.ndarray,
    groups: list[list[np.ndarray]],
    group_sizes: np.ndarray,
    exposed_arms: np.ndarray,
    correct: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:
    seed_pattern_rows: list[dict[str, Any]] = []
    group_differences = np.zeros((8, 40, 34), dtype=np.int16)
    point_matrix = np.zeros((8, 40), dtype=np.float64)
    for model_seed in MODEL_SEEDS:
        for pattern in patterns:
            pattern_index = pattern["pattern_index"]
            rows = np.flatnonzero(pattern_by_row == pattern_index)
            exposed_arm_index = int(exposed_arms[model_seed, pattern_index])
            unexposed_arm_index = 1 - exposed_arm_index
            exposed_correct = int(
                correct[model_seed, exposed_arm_index, rows].sum()
            )
            unexposed_correct = int(
                correct[model_seed, unexposed_arm_index, rows].sum()
            )
            effect_pp = (
                100.0
                * (exposed_correct - unexposed_correct)
                / ROWS_PER_PATTERN
            )
            point_matrix[model_seed, pattern_index] = effect_pp
            for group_index, group_rows in enumerate(
                groups[pattern_index]
            ):
                group_differences[
                    model_seed,
                    pattern_index,
                    group_index,
                ] = int(
                    correct[
                        model_seed,
                        exposed_arm_index,
                        group_rows,
                    ].sum()
                    - correct[
                        model_seed,
                        unexposed_arm_index,
                        group_rows,
                    ].sum()
                )
            seed_pattern_rows.append(
                {
                    "effect_pp": effect_pp,
                    "evaluation_rows": len(rows),
                    "exposed_accuracy": exposed_correct / len(rows),
                    "exposed_arm": ARMS[exposed_arm_index],
                    "exposed_correct": exposed_correct,
                    "model_seed": model_seed,
                    "pair_index": pattern["pair_index"],
                    "pair_member": pattern["pair_member"],
                    "pattern_index": pattern_index,
                    "semantic_id": pattern["semantic_id"],
                    "target_relation": pattern["target_relation"],
                    "unexposed_accuracy": unexposed_correct / len(rows),
                    "unexposed_arm": ARMS[unexposed_arm_index],
                    "unexposed_correct": unexposed_correct,
                }
            )
    pattern_rows: list[dict[str, Any]] = []
    for pattern in patterns:
        index = pattern["pattern_index"]
        pattern_rows.append(
            {
                "effect_pp": float(point_matrix[:, index].mean()),
                "pair_index": pattern["pair_index"],
                "pair_member": pattern["pair_member"],
                "pattern_index": index,
                "semantic_id": pattern["semantic_id"],
                "target_relation": pattern["target_relation"],
            }
        )
    seed_rows = [
        {
            "effect_pp": float(point_matrix[model_seed].mean()),
            "model_seed": model_seed,
        }
        for model_seed in MODEL_SEEDS
    ]
    pair_rows: list[dict[str, Any]] = []
    for pair_index in range(20):
        pattern_indices = pair_pattern_indices[pair_index]
        pair_rows.append(
            {
                "effect_pp": float(
                    point_matrix[:, pattern_indices].mean()
                ),
                "member_0_pattern_index": int(pattern_indices[0]),
                "member_1_pattern_index": int(pattern_indices[1]),
                "pair_index": pair_index,
                "target_relation": patterns[
                    int(pattern_indices[0])
                ]["target_relation"],
            }
        )
    relation_to_indices: dict[str, list[int]] = defaultdict(list)
    for pattern in patterns:
        relation_to_indices[pattern["target_relation"]].append(
            pattern["pattern_index"]
        )
    relation_rows = [
        {
            "effect_pp": float(point_matrix[:, indices].mean()),
            "pattern_count": len(indices),
            "target_relation": relation,
        }
        for relation, indices in sorted(relation_to_indices.items())
    ]
    if not np.array_equal(
        group_sizes.sum(axis=1),
        np.full(40, 200),
    ):
        raise RuntimeError("Group sizes do not recover pattern rows.")
    return (
        seed_pattern_rows,
        pattern_rows,
        seed_rows,
        pair_rows,
        relation_rows,
        group_differences,
    )


def _aligned_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    pattern_by_row: np.ndarray,
    exposed_arms: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    metric_names = (
        "accuracy",
        "brier_score",
        "cross_entropy",
        "ece_15",
        "mean_confidence",
        "signed_calibration_gap",
    )
    rows: list[dict[str, Any]] = []
    differences: dict[str, list[float]] = {
        name: [] for name in metric_names
    }
    for model_seed in MODEL_SEEDS:
        exposed_logits = np.empty((8000, 23), dtype=np.float32)
        unexposed_logits = np.empty((8000, 23), dtype=np.float32)
        for pattern_index in range(40):
            selected = pattern_by_row == pattern_index
            exposed_arm = int(exposed_arms[model_seed, pattern_index])
            exposed_logits[selected] = logits[
                model_seed,
                exposed_arm,
                selected,
            ]
            unexposed_logits[selected] = logits[
                model_seed,
                1 - exposed_arm,
                selected,
            ]
        exposed = classification_metrics(exposed_logits, labels)
        unexposed = classification_metrics(unexposed_logits, labels)
        row: dict[str, Any] = {"model_seed": model_seed}
        for name in metric_names:
            difference = exposed[name] - unexposed[name]
            differences[name].append(difference)
            row[f"exposed_{name}"] = exposed[name]
            row[f"unexposed_{name}"] = unexposed[name]
            row[f"difference_{name}"] = difference
        rows.append(row)
    aggregate = {
        f"mean_exposed_minus_unexposed_{name}": float(
            np.mean(differences[name])
        )
        for name in metric_names
    }
    return rows, aggregate


def _report(results: dict[str, Any]) -> str:
    primary = results["primary"]
    interval = primary["confidence_interval_95"]
    decision = results["decision"]
    secondary = results["secondary"]
    return "\n".join(
        [
            "# CLUTRR V3 暴露效应解盲报告",
            "",
            "## 主结果",
            "",
            (
                f"- 暴露减未暴露准确率效应："
                f"{primary['effect_pp']:.4f} pp。"
            ),
            (
                f"- 95% 层级 bootstrap 区间："
                f"[{interval['lower_pp']:.4f}, "
                f"{interval['upper_pp']:.4f}] pp；"
                f"宽度 {interval['width_pp']:.4f} pp。"
            ),
            (
                f"- 预注册判定：`{decision['classification']}`；"
                f"精度门通过："
                f"`{str(decision['precision_gate_pass']).lower()}`。"
            ),
            "",
            "## 描述性结果",
            "",
            (
                "- Cross-entropy 暴露减未暴露："
                f"{secondary['mean_exposed_minus_unexposed_cross_entropy']:.6f}。"
            ),
            (
                "- Brier 暴露减未暴露："
                f"{secondary['mean_exposed_minus_unexposed_brier_score']:.6f}。"
            ),
            (
                "- ECE-15 暴露减未暴露："
                f"{secondary['mean_exposed_minus_unexposed_ece_15']:.6f}。"
            ),
            "",
            (
                "主结论只使用预注册准确率 estimand、20,000 次层级 "
                "bootstrap 和冻结阈值。其他指标与分组结果均为描述性，"
                "不参与 Gate。"
            ),
            "",
        ]
    )


def run_analysis() -> dict[str, Any]:
    verify_protocol_bundle()
    verify_frozen_inputs()
    if ANALYSIS_DIR.exists():
        raise RuntimeError("Analysis output directory already exists.")
    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    evaluation = load_npz(EXECUTION_DIR / "evaluation_manifest.npz")
    patterns, pair_pattern_indices = _patterns()
    pattern_by_row, groups, group_sizes = _evaluation_groups(
        evaluation,
        patterns,
    )
    exposed_arms = _exposed_arms(patterns)
    logits, correct = _load_predictions(evaluation)
    (
        seed_pattern_rows,
        pattern_rows,
        seed_rows,
        pair_rows,
        relation_rows,
        group_differences,
    ) = _effect_tables(
        patterns,
        pair_pattern_indices,
        pattern_by_row,
        groups,
        group_sizes,
        exposed_arms,
        correct,
    )
    point_pp = float(
        np.mean([float(row["effect_pp"]) for row in seed_pattern_rows])
    )
    bootstrap_effects, draw_hashes = hierarchical_bootstrap(
        group_differences,
        group_sizes,
        pair_pattern_indices,
        replicates=prereg["bootstrap"]["replicates"],
        seed=prereg["bootstrap"]["seed"],
        batch_size=prereg["bootstrap"]["implementation_batch_size"],
    )
    lower_pp, upper_pp = np.quantile(
        bootstrap_effects,
        [0.025, 0.975],
        method="linear",
    )
    decision = classify_gate(
        point_pp,
        float(lower_pp),
        float(upper_pp),
    )
    labels = evaluation["labels"].astype(np.int64)
    aligned_metric_rows, secondary = _aligned_metrics(
        logits,
        labels,
        pattern_by_row,
        exposed_arms,
    )
    if not math.isclose(
        100.0
        * secondary["mean_exposed_minus_unexposed_accuracy"],
        point_pp,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Primary and aligned accuracy effects disagree.")

    ANALYSIS_DIR.mkdir(parents=True)
    write_csv(
        ANALYSIS_DIR / "seed_pattern_effects.csv",
        seed_pattern_rows,
    )
    write_csv(ANALYSIS_DIR / "pattern_effects.csv", pattern_rows)
    write_csv(ANALYSIS_DIR / "seed_effects.csv", seed_rows)
    write_csv(ANALYSIS_DIR / "pair_effects.csv", pair_rows)
    write_csv(ANALYSIS_DIR / "relation_effects.csv", relation_rows)
    write_csv(
        ANALYSIS_DIR / "seed_aligned_metrics.csv",
        aligned_metric_rows,
    )
    write_deterministic_npz(
        ANALYSIS_DIR / "bootstrap.npz",
        {"effects_pp": bootstrap_effects},
    )
    results = {
        "analysis_id": ANALYSIS_ID,
        "analyzed_at_utc": _now(),
        "decision": decision,
        "integrity": {
            "evaluation_rows_per_run": EVAL_ROWS,
            "model_seeds": 8,
            "pattern_pairs": 20,
            "patterns": 40,
            "prediction_files": 16,
            "source_groups_per_pattern": 34,
        },
        "primary": {
            "bootstrap": {
                "draw_sha256": draw_hashes,
                "effects_sha256": array_sha256(bootstrap_effects),
                "replicates": len(bootstrap_effects),
                "seed": BOOTSTRAP_SEED,
            },
            "confidence_interval_95": {
                "lower_pp": float(lower_pp),
                "upper_pp": float(upper_pp),
                "width_pp": float(upper_pp - lower_pp),
            },
            "effect_pp": point_pp,
            "estimand": (
                "equal-seed equal-pattern row-weighted "
                "exposed-minus-unexposed accuracy"
            ),
        },
        "protocol_input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "protocol_sha256s_sha256": sha256_file(
            PROTOCOL_DIR / "SHA256SUMS"
        ),
        "schema_version": (
            "external-behavior-v3-effect-analysis-results-v1"
        ),
        "secondary": secondary,
        "status": "unblinded_analysis_complete",
    }
    write_json(ANALYSIS_DIR / "results.json", results)
    (ANALYSIS_DIR / "report_cn.md").write_text(
        _report(results),
        encoding="utf-8",
    )
    output_hashes = {
        name: sha256_file(ANALYSIS_DIR / name)
        for name in OUTPUT_NAMES
    }
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_implementation_sha256": sha256_file(
            PROJECT_ROOT
            / "toolkit"
            / "external_behavior_v3"
            / "effect_analysis.py"
        ),
        "completed_at_utc": _now(),
        "output_sha256": output_hashes,
        "protocol_input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "schema_version": (
            "external-behavior-v3-effect-analysis-manifest-v1"
        ),
        "status": "analysis_complete_pending_independent_validation",
    }
    write_json(ANALYSIS_DIR / "analysis_manifest.json", manifest)
    return results


def _direct_primary_recompute() -> float:
    evaluation = load_npz(EXECUTION_DIR / "evaluation_manifest.npz")
    patterns, _ = _patterns()
    pattern_by_row, _, _ = _evaluation_groups(evaluation, patterns)
    exposed_arms = _exposed_arms(patterns)
    _, correct = _load_predictions(evaluation)
    differences: list[float] = []
    for model_seed in MODEL_SEEDS:
        for pattern_index in range(40):
            rows = np.flatnonzero(pattern_by_row == pattern_index)
            exposed_arm = int(exposed_arms[model_seed, pattern_index])
            exposed = float(
                correct[model_seed, exposed_arm, rows].mean()
            )
            unexposed = float(
                correct[model_seed, 1 - exposed_arm, rows].mean()
            )
            differences.append(100.0 * (exposed - unexposed))
    return float(np.mean(differences))


def _validation_checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    labels = (
        ("protocol_bundle", "Validate the frozen protocol bundle."),
        ("input_hashes", "Validate every frozen input and implementation hash."),
        ("output_hashes", "Validate all declared analysis output hashes."),
        ("primary_recompute", "Independently recompute the primary point estimate."),
        ("bootstrap", "Reproduce all 20,000 hierarchical bootstrap draws."),
        ("interval", "Recompute the registered percentile interval."),
        ("decision", "Reapply the frozen decision precedence and thresholds."),
        ("tables", "Validate registered table row counts and identities."),
        ("finite", "Confirm all numeric analysis outputs are finite."),
    )
    lines = ["# CLUTRR V3 Effect Analysis Integrity", ""]
    for name, label in labels:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_analysis() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    try:
        verify_protocol_bundle()
        protocol_ok = True
        protocol_detail = None
    except Exception as error:
        protocol_ok = False
        protocol_detail = repr(error)
    check("protocol_bundle", protocol_ok, protocol_detail)
    try:
        verify_frozen_inputs()
        inputs_ok = True
        input_detail = None
    except Exception as error:
        inputs_ok = False
        input_detail = repr(error)
    check("input_hashes", inputs_ok, input_detail)
    manifest = read_json(ANALYSIS_DIR / "analysis_manifest.json")
    output_failures: list[str] = []
    for name, expected in manifest["output_sha256"].items():
        path = ANALYSIS_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            output_failures.append(name)
    check("output_hashes", not output_failures, output_failures)
    results = read_json(ANALYSIS_DIR / "results.json")
    direct_point = _direct_primary_recompute()
    reported_point = float(results["primary"]["effect_pp"])
    check(
        "primary_recompute",
        math.isclose(
            direct_point,
            reported_point,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        {"direct": direct_point, "reported": reported_point},
    )

    evaluation = load_npz(EXECUTION_DIR / "evaluation_manifest.npz")
    patterns, pair_pattern_indices = _patterns()
    pattern_by_row, groups, group_sizes = _evaluation_groups(
        evaluation,
        patterns,
    )
    exposed_arms = _exposed_arms(patterns)
    _, correct = _load_predictions(evaluation)
    *_, group_differences = _effect_tables(
        patterns,
        pair_pattern_indices,
        pattern_by_row,
        groups,
        group_sizes,
        exposed_arms,
        correct,
    )
    reproduced, reproduced_draw_hashes = hierarchical_bootstrap(
        group_differences,
        group_sizes,
        pair_pattern_indices,
    )
    stored = load_npz(ANALYSIS_DIR / "bootstrap.npz")["effects_pp"]
    expected_bootstrap = results["primary"]["bootstrap"]
    bootstrap_ok = (
        np.array_equal(reproduced, stored)
        and array_sha256(stored) == expected_bootstrap["effects_sha256"]
        and reproduced_draw_hashes == expected_bootstrap["draw_sha256"]
    )
    check("bootstrap", bootstrap_ok)
    lower_pp, upper_pp = np.quantile(
        stored,
        [0.025, 0.975],
        method="linear",
    )
    reported_interval = results["primary"]["confidence_interval_95"]
    interval_ok = all(
        math.isclose(float(observed), float(expected), abs_tol=1e-12)
        for observed, expected in (
            (lower_pp, reported_interval["lower_pp"]),
            (upper_pp, reported_interval["upper_pp"]),
            (
                upper_pp - lower_pp,
                reported_interval["width_pp"],
            ),
        )
    )
    check("interval", interval_ok)
    reproduced_decision = classify_gate(
        reported_point,
        float(lower_pp),
        float(upper_pp),
    )
    check(
        "decision",
        reproduced_decision == results["decision"],
        {
            "reported": results["decision"],
            "reproduced": reproduced_decision,
        },
    )
    expected_rows = {
        "pair_effects.csv": 20,
        "pattern_effects.csv": 40,
        "relation_effects.csv": len(
            {pattern["target_relation"] for pattern in patterns}
        ),
        "seed_aligned_metrics.csv": 8,
        "seed_effects.csv": 8,
        "seed_pattern_effects.csv": 320,
    }
    table_detail = {
        name: len(read_csv(ANALYSIS_DIR / name))
        for name in expected_rows
    }
    check(
        "tables",
        table_detail == expected_rows,
        {"expected": expected_rows, "observed": table_detail},
    )

    def finite_tree(value: Any) -> bool:
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True

    check(
        "finite",
        finite_tree(results) and np.isfinite(stored).all(),
    )
    failed = [item for item in checks if not item["pass"]]
    report = {
        "analysis_gate": "pass" if not failed else "no_go",
        "analysis_id": ANALYSIS_ID,
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "primary_decision": results["decision"]["classification"],
        "schema_version": (
            "external-behavior-v3-effect-analysis-validation-v1"
        ),
        "status": "pass" if not failed else "no_go",
        "validated_at_utc": _now(),
    }
    write_json(ANALYSIS_DIR / "analysis_validation_report.json", report)
    (ANALYSIS_DIR / "analysis_checklist.md").write_text(
        _validation_checklist(checks),
        encoding="ascii",
    )
    checksum_names = (
        *OUTPUT_NAMES,
        "analysis_manifest.json",
        "analysis_checklist.md",
        "analysis_validation_report.json",
    )
    (ANALYSIS_DIR / "ANALYSIS_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(ANALYSIS_DIR / name)}  {name}\n"
            for name in checksum_names
        ),
        encoding="ascii",
    )
    return report
