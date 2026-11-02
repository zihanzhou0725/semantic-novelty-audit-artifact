"""Prospective directional V4 power simulation and block design."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from toolkit.external_behavior_v3.effect_analysis import (
    DATA_DIR,
    PROJECT_ROOT,
    array_sha256,
    read_csv,
    read_json,
    relative,
    sha256_file,
    write_csv,
    write_deterministic_npz,
    write_json,
)
from toolkit.external_behavior_v3.variance_design import (
    ANALYSIS_DIR as VARIANCE_ANALYSIS_DIR,
    PROTOCOL_DIR as VARIANCE_PROTOCOL_DIR,
    all_candidate_pairs,
    verify_variance_inputs,
    verify_variance_protocol,
)


DESIGN_ID = "external-behavior-v4-directional-design-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v4"
    / "directional-design-v1"
)
ANALYSIS_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "external-behavior-v4"
    / "directional-design-v1"
)
MANIFEST_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "directional-design-v1"
)
SIMULATION_REPLICATES = 100_000
SIMULATION_SEED = 202_610_131
MODEL_SEEDS = tuple(range(8))
ARMS = ("A", "B")
PAIRS_PER_BLOCK = 20
CANDIDATE_BLOCK_COUNTS = (1, 2, 3, 4)
MINIMUM_POWER = 0.90
MINIMUM_WILSON_LOWER = 0.90
SESOI_PP = 1.0
Z_975 = 1.959963984540054
REMAINING_PAIR_CAPACITY = 139
V3_POINT_PP = 15.3375
V3_LOWER_PP = 9.876873975196792
DESIGN_EFFECT_PP = min(0.5 * V3_POINT_PP, V3_LOWER_PP)
SENSITIVITY_EFFECTS_PP = (
    0.0,
    1.0,
    5.0,
    DESIGN_EFFECT_PP,
    V3_LOWER_PP,
    V3_POINT_PP,
)

IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v4_directional_design.py",
    "scripts/analyze_external_behavior_v4_directional_design.py",
    "scripts/validate_external_behavior_v4_directional_design.py",
    "tests/test_external_behavior_v4_directional_design.py",
    "toolkit/external_behavior_v4/__init__.py",
    "toolkit/external_behavior_v4/directional_design.py",
    "protocols/external-behavior-v4/directional-design-v1/validate.py",
)

ANALYSIS_OUTPUT_NAMES = (
    "block_gate_report.json",
    "power_results.json",
    "power_simulation.csv",
    "report_cn.md",
    "simulation_draws.npz",
)
DESIGN_OUTPUT_NAMES = (
    "design_manifest.json",
    "selected_pairs.csv",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line:
            digest, name = line.split("  ", 1)
            entries.append((digest, path.parent / name))
    return entries


def _input_paths() -> list[Path]:
    return sorted(
        [
            VARIANCE_PROTOCOL_DIR / "SHA256SUMS",
            VARIANCE_PROTOCOL_DIR / "input_manifest.json",
            VARIANCE_PROTOCOL_DIR / "preregistration.json",
            VARIANCE_PROTOCOL_DIR / "validation_report.json",
            VARIANCE_ANALYSIS_DIR / "ANALYSIS_SHA256SUMS",
            VARIANCE_ANALYSIS_DIR / "analysis_validation_report.json",
            VARIANCE_ANALYSIS_DIR / "remaining_pair_capacity.csv",
            VARIANCE_ANALYSIS_DIR / "v4_design_gate_report.json",
            VARIANCE_ANALYSIS_DIR / "variance_results.json",
            DATA_DIR / "clutrr_v3_eligible_patterns.jsonl",
            DATA_DIR / "clutrr_v3_pairs.jsonl",
        ],
        key=relative,
    )


def _implementation_hashes() -> dict[str, str]:
    missing = [
        name
        for name in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing V4 design implementation: {missing}")
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in IMPLEMENTATION_PATHS
    }


def verify_protocol() -> None:
    for expected, path in _checksum_entries(PROTOCOL_DIR / "SHA256SUMS"):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V4 protocol checksum mismatch: {path}")
    report = read_json(PROTOCOL_DIR / "validation_report.json")
    if (
        report.get("status") != "pass"
        or report.get("stage") != "pre_simulation"
        or report.get("failed_count") != 0
    ):
        raise RuntimeError("V4 pre-simulation validation did not pass.")


def verify_inputs() -> None:
    manifest = read_json(PROTOCOL_DIR / "input_manifest.json")
    for name, expected in manifest["input_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen V4 input changed: {name}")
    for name, expected in manifest["implementation_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen V4 code changed: {name}")


def validate_protocol(*, require_outputs_absent: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    variance_validation = read_json(
        VARIANCE_ANALYSIS_DIR / "analysis_validation_report.json"
    )
    strict_gate = read_json(
        VARIANCE_ANALYSIS_DIR / "v4_design_gate_report.json"
    )
    check(
        "upstream_integrity",
        variance_validation.get("status") == "pass"
        and variance_validation.get("failed_count") == 0,
    )
    check(
        "strict_gate_closed",
        strict_gate["gate"]["decision"]
        == "strict_v3_precision_replication_no_go"
        and strict_gate["gate"]["v4_training_released"] is False,
    )
    check(
        "directional_estimand",
        prereg["primary_gate"]["sesoi_pp"] == SESOI_PP
        and prereg["primary_gate"]["positive_rule"]
        == "point_estimate_pp >= 1.0 and two_sided_95_percent_ci_lower_pp > 0",
    )
    power = prereg["power_simulation"]
    check(
        "power_rule",
        power["replicates"] == SIMULATION_REPLICATES
        and power["seed"] == SIMULATION_SEED
        and power["minimum_power"] == MINIMUM_POWER
        and power["minimum_wilson_lower"] == MINIMUM_WILSON_LOWER
        and math.isclose(
            power["design_effect_pp"],
            DESIGN_EFFECT_PP,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    )
    check(
        "block_rule",
        prereg["block_design"]["candidate_block_counts"]
        == list(CANDIDATE_BLOCK_COUNTS)
        and prereg["block_design"]["pairs_per_block"]
        == PAIRS_PER_BLOCK
        and prereg["block_design"]["runs_per_block"] == 16,
    )
    check(
        "independence",
        prereg["target_selection"][
            "v3_target_semantic_ids_excluded"
        ]
        is True
        and prereg["target_selection"]["selection_uses_v3_outcomes"]
        is False,
    )
    manifest_path = PROTOCOL_DIR / "input_manifest.json"
    receipt_path = PROTOCOL_DIR / "freeze_receipt.json"
    check(
        "freeze_artifacts",
        manifest_path.is_file() and receipt_path.is_file(),
    )
    input_failures: list[str] = []
    code_failures: list[str] = []
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for name, expected in manifest["input_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                input_failures.append(name)
        for name, expected in manifest["implementation_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                code_failures.append(name)
    check("input_hashes", not input_failures, input_failures)
    check("implementation_hashes", not code_failures, code_failures)
    artifacts = []
    for root in (ANALYSIS_DIR, MANIFEST_DIR):
        if root.exists():
            artifacts.extend(root.rglob("*"))
    if require_outputs_absent:
        check(
            "outputs_absent",
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
        "check_count": len(checks),
        "checks": checks,
        "design_id": DESIGN_ID,
        "document_sha256": {
            name: sha256_file(PROTOCOL_DIR / name)
            for name in document_names
        },
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v4-directional-protocol-validation-v1"
        ),
        "stage": (
            "pre_simulation"
            if require_outputs_absent
            else "post_simulation_recheck"
        ),
        "status": "pass" if not failed else "fail",
        "validated_at_utc": _now(),
    }


def freeze_protocol() -> dict[str, Any]:
    if (PROTOCOL_DIR / "input_manifest.json").exists():
        raise RuntimeError("Directional V4 protocol is already frozen.")
    for root in (ANALYSIS_DIR, MANIFEST_DIR):
        if root.exists() and any(root.rglob("*")):
            raise RuntimeError("V4 output exists before protocol freeze.")
    verify_variance_protocol()
    verify_variance_inputs()
    validation = read_json(
        VARIANCE_ANALYSIS_DIR / "analysis_validation_report.json"
    )
    if validation.get("status") != "pass":
        raise RuntimeError("Variance-design integrity did not pass.")
    paths = _input_paths()
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V4 design inputs: {missing}")
    manifest = {
        "design_id": DESIGN_ID,
        "frozen_at_utc": _now(),
        "implementation_sha256": _implementation_hashes(),
        "input_sha256": {
            relative(path): sha256_file(path) for path in paths
        },
        "schema_version": (
            "external-behavior-v4-directional-input-manifest-v1"
        ),
        "status": "frozen_before_power_simulation",
    }
    write_json(PROTOCOL_DIR / "input_manifest.json", manifest)
    receipt = {
        "design_id": DESIGN_ID,
        "frozen_at_utc": _now(),
        "schema_version": (
            "external-behavior-v4-directional-freeze-receipt-v1"
        ),
        "state_at_freeze": {
            "block_count_selected": False,
            "pair_manifest_materialized": False,
            "power_simulation_executed": False,
            "simulation_outputs_absent": True,
            "training_started": False,
            "v3_pilot_results_known": True,
        },
        "status": "frozen_before_power_simulation",
    }
    write_json(PROTOCOL_DIR / "freeze_receipt.json", receipt)
    report = validate_protocol(require_outputs_absent=True)
    if report["status"] != "pass":
        raise RuntimeError(f"V4 protocol validation failed: {report}")
    write_json(PROTOCOL_DIR / "validation_report.json", report)
    names = (
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
            for name in names
        ),
        encoding="ascii",
    )
    return {
        "design_id": DESIGN_ID,
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "status": receipt["status"],
        "validation_status": report["status"],
    }


def calibrated_standard_error(pair_count: int) -> float:
    variance = read_json(
        VARIANCE_ANALYSIS_DIR / "variance_results.json"
    )
    anova = variance["anova"]
    projection = variance["prospective_projection"]
    raw_variance = (
        anova["model_seed_variance_pp2"] / 8
        + anova["pattern_pair_variance_pp2"] / pair_count
        + anova["interaction_variance_pp2"] / (8 * pair_count)
    )
    return (
        math.sqrt(raw_variance)
        * projection["anova_to_registered_width_calibration_factor"]
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = Z_975,
) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Invalid binomial count.")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def analytic_power(effect_pp: float, standard_error_pp: float) -> float:
    threshold = max(SESOI_PP, Z_975 * standard_error_pp)
    standardized = (threshold - effect_pp) / standard_error_pp
    return 1.0 - NormalDist().cdf(standardized)


def select_block_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate_candidates = [
        row for row in rows if row["design_scenario"]
    ]
    passing = [
        row
        for row in gate_candidates
        if row["monte_carlo_power"] >= MINIMUM_POWER
        and row["wilson_95_lower"] >= MINIMUM_WILSON_LOWER
    ]
    selected = (
        min(passing, key=lambda row: row["block_count"])
        if passing
        else None
    )
    selected_pair_count = (
        int(selected["pair_count"]) if selected is not None else None
    )
    capacity_pass = (
        selected_pair_count is None
        or selected_pair_count <= REMAINING_PAIR_CAPACITY
    )
    if not capacity_pass:
        selected = None
        selected_pair_count = None
    return {
        "capacity_gate_pass": capacity_pass,
        "candidate_block_counts": list(CANDIDATE_BLOCK_COUNTS),
        "decision": (
            "directional_v4_design_go"
            if selected is not None
            else "directional_v4_design_no_go"
        ),
        "design_effect_pp": DESIGN_EFFECT_PP,
        "minimum_power": MINIMUM_POWER,
        "minimum_wilson_lower": MINIMUM_WILSON_LOWER,
        "selected_block_count": (
            int(selected["block_count"]) if selected is not None else None
        ),
        "selected_pair_count": (
            selected_pair_count
        ),
        "selected_run_count": (
            int(selected["run_count"]) if selected is not None else None
        ),
        "training_released": False,
    }


def simulate_power() -> tuple[
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    rng = np.random.Generator(np.random.PCG64(SIMULATION_SEED))
    draws = rng.standard_normal(
        (SIMULATION_REPLICATES, len(CANDIDATE_BLOCK_COUNTS))
    )
    rows: list[dict[str, Any]] = []
    for column, block_count in enumerate(CANDIDATE_BLOCK_COUNTS):
        pair_count = block_count * PAIRS_PER_BLOCK
        standard_error = calibrated_standard_error(pair_count)
        half_width = Z_975 * standard_error
        for effect_pp in SENSITIVITY_EFFECTS_PP:
            estimates = effect_pp + standard_error * draws[:, column]
            success = (estimates >= SESOI_PP) & (
                estimates - half_width > 0.0
            )
            successes = int(success.sum())
            power = successes / SIMULATION_REPLICATES
            lower, upper = wilson_interval(
                successes,
                SIMULATION_REPLICATES,
            )
            row = {
                "analytic_power": analytic_power(
                    effect_pp,
                    standard_error,
                ),
                "block_count": block_count,
                "design_scenario": math.isclose(
                    effect_pp,
                    DESIGN_EFFECT_PP,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "effect_assumption_pp": effect_pp,
                "monte_carlo_power": power,
                "pair_count": pair_count,
                "run_count": block_count * 16,
                "standard_error_pp": standard_error,
                "successes": successes,
                "two_sided_95_half_width_pp": half_width,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
            }
            rows.append(row)
    gate = select_block_gate(rows)
    return draws, rows, gate


def remaining_pairs() -> list[dict[str, Any]]:
    selected_v3 = {
        row["pair_sha256"]
        for row in (
            json.loads(line)
            for line in (
                DATA_DIR / "clutrr_v3_pairs.jsonl"
            ).read_text(encoding="ascii").splitlines()
            if line
        )
    }
    result = [
        row
        for row in all_candidate_pairs()
        if row["pair_sha256"] not in selected_v3
    ]
    if len(result) != 139:
        raise RuntimeError("Remaining V4 pair capacity changed.")
    return result


def selected_pair_rows(pair_count: int) -> list[dict[str, Any]]:
    candidates = remaining_pairs()
    if pair_count > len(candidates) or pair_count % PAIRS_PER_BLOCK:
        raise ValueError("Invalid selected pair count.")
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(candidates[:pair_count]):
        rows.append(
            {
                "block_index": index // PAIRS_PER_BLOCK,
                "block_pair_index": index % PAIRS_PER_BLOCK,
                "global_v4_pair_index": index,
                "member_0": pair["member_0"],
                "member_0_complete_source_groups": pair[
                    "member_0_complete_source_groups"
                ],
                "member_1": pair["member_1"],
                "member_1_complete_source_groups": pair[
                    "member_1_complete_source_groups"
                ],
                "pair_sha256": pair["pair_sha256"],
                "target_relation": pair["target_relation"],
            }
        )
    return rows


def _selection_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [row["pair_sha256"] for row in rows],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _design_manifest(
    pair_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    power_results_sha256: str,
) -> dict[str, Any]:
    pair_count = len(pair_rows)
    block_count = pair_count // PAIRS_PER_BLOCK
    selected_semantics = sorted(
        {
            row[member]
            for row in pair_rows
            for member in ("member_0", "member_1")
        }
    )
    v3_semantics = sorted(
        {
            row[member]
            for row in (
                json.loads(line)
                for line in (
                    DATA_DIR / "clutrr_v3_pairs.jsonl"
                ).read_text(encoding="ascii").splitlines()
                if line
            )
            for member in ("member_0", "member_1")
        }
    )
    if set(selected_semantics) & set(v3_semantics):
        raise RuntimeError("V4 target semantics overlap V3.")
    return {
        "analysis": {
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 202_610_132,
            "confidence": 0.95,
            "decision_rule": (
                "point_estimate_pp >= 1.0 and "
                "two_sided_95_percent_ci_lower_pp > 0"
            ),
            "effect_unit": "model_seed_by_target_pattern",
            "orientation": "exposed_minus_unexposed",
            "pair_draws_per_replicate": pair_count,
            "seed_draws_per_replicate": 8,
            "source_group_draws_per_pattern": 34,
        },
        "arms": list(ARMS),
        "block_design": {
            "block_count": block_count,
            "evaluation_rows_per_block_run": 8_000,
            "exposure_rows_per_arm": 1_280,
            "pairs_per_block": PAIRS_PER_BLOCK,
            "runs_per_block": 16,
            "target_patterns_per_block": 40,
            "total_runs": block_count * 16,
            "training_rows_per_run": 24_000,
        },
        "common_base": {
            "rows_per_run": 22_720,
            "same_rows_across_all_blocks_and_arms": True,
            "semantic_exclusions": (
                "all V3 target semantic IDs and all selected V4 target "
                "semantic IDs"
            ),
        },
        "design_gate": gate,
        "design_id": DESIGN_ID,
        "evaluation_allocation": {
            "rows_per_pattern": 200,
            "source_groups_per_pattern": 34,
            "source_group_rule": "same deterministic rule as V3",
        },
        "exposure_allocation": {
            "rows_per_pattern": 64,
            "source_groups_per_pattern": 11,
            "source_group_rule": "same deterministic rule as V3",
        },
        "frozen_at_utc": _now(),
        "model_seeds": list(MODEL_SEEDS),
        "paired_randomness": {
            "arm_assignment_namespace": (
                "clutrr:v4:arm:{block_index}:{model_seed}:"
                "{block_pair_index}"
            ),
            "epoch_order_namespace": (
                "clutrr:epoch:{model_seed}:{epoch_1_based}"
            ),
            "model_initialization_namespace": "clutrr:model:{model_seed}",
            "shared_across_blocks": True,
        },
        "power_results_sha256": power_results_sha256,
        "schema_version": (
            "external-behavior-v4-directional-design-manifest-v1"
        ),
        "selected_pair_count": pair_count,
        "selected_pair_manifest_sha256": _selection_sha256(pair_rows),
        "selected_pattern_count": len(selected_semantics),
        "status": "design_go_pending_materialization_gate",
        "target_independence": {
            "selected_v4_semantic_ids_sha256": hashlib.sha256(
                json.dumps(
                    selected_semantics,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
            "v3_semantic_overlap_count": 0,
            "v3_target_semantic_ids_excluded": True,
        },
        "training_released": False,
    }


def _report(
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    selected = (
        next(
            row
            for row in rows
            if row["design_scenario"]
            and row["block_count"] == gate["selected_block_count"]
        )
        if gate["selected_block_count"] is not None
        else None
    )
    selected_lines = (
        [
            f"- 选择 blocks：{gate['selected_block_count']}。",
            f"- 选择 pairs：{gate['selected_pair_count']}。",
            f"- 计划 runs：{gate['selected_run_count']}。",
            (
                f"- 模拟 power：{selected['monte_carlo_power']:.4f}；"
                f"Wilson 下界：{selected['wilson_95_lower']:.4f}。"
            ),
            (
                f"- 投影 SE：{selected['standard_error_pp']:.4f} pp；"
                f"95% 半宽：{selected['two_sided_95_half_width_pp']:.4f} pp。"
            ),
        ]
        if selected is not None
        else ["- 没有候选 block 数通过冻结的功效门。"]
    )
    return "\n".join(
        [
            "# Directional V4 功效与 Block 门控",
            "",
            "## 冻结假设",
            "",
            f"- Pilot-shrunk 设计效应：{DESIGN_EFFECT_PP:.5f} pp。",
            "- Gate：效应点估计至少 1 pp，且双侧 95% CI 下界大于 0。",
            "- Power 与 95% Wilson 下界均须至少 90%。",
            "",
            "## 门控结果",
            "",
            f"- 决策：`{gate['decision']}`。",
            *selected_lines,
            "",
            "## 约束",
            "",
            "- 仅选择 V3 未使用的 deterministic pair prefix。",
            "- 每个 block 保持 V3 的 20-pair、24k-row 和 exposure dose。",
            "- 当前仅冻结设计；训练仍未放行，必须先通过 materialization gate。",
            "",
        ]
    )


def run_design() -> dict[str, Any]:
    verify_protocol()
    verify_inputs()
    for root in (ANALYSIS_DIR, MANIFEST_DIR):
        if root.exists():
            raise RuntimeError(f"V4 output directory already exists: {root}")
    draws, rows, gate = simulate_power()
    if gate["decision"] != "directional_v4_design_go":
        selected_rows: list[dict[str, Any]] = []
    else:
        selected_rows = selected_pair_rows(gate["selected_pair_count"])
    ANALYSIS_DIR.mkdir(parents=True)
    MANIFEST_DIR.mkdir(parents=True)
    write_csv(ANALYSIS_DIR / "power_simulation.csv", rows)
    write_deterministic_npz(
        ANALYSIS_DIR / "simulation_draws.npz",
        {"standard_normal_draws": draws},
    )
    power_results = {
        "analytic_agreement_max_abs": max(
            abs(row["analytic_power"] - row["monte_carlo_power"])
            for row in rows
        ),
        "design_id": DESIGN_ID,
        "draws_sha256": array_sha256(draws),
        "gate": gate,
        "replicates": SIMULATION_REPLICATES,
        "schema_version": (
            "external-behavior-v4-directional-power-results-v1"
        ),
        "seed": SIMULATION_SEED,
        "simulation_rows": rows,
        "status": "power_simulation_complete",
    }
    write_json(ANALYSIS_DIR / "power_results.json", power_results)
    write_json(
        ANALYSIS_DIR / "block_gate_report.json",
        {
            "design_id": DESIGN_ID,
            "gate": gate,
            "schema_version": (
                "external-behavior-v4-block-gate-report-v1"
            ),
            "status": gate["decision"],
        },
    )
    if selected_rows:
        write_csv(MANIFEST_DIR / "selected_pairs.csv", selected_rows)
        design_manifest = _design_manifest(
            selected_rows,
            gate,
            sha256_file(ANALYSIS_DIR / "power_results.json"),
        )
    else:
        write_csv(
            MANIFEST_DIR / "selected_pairs.csv",
            [{"status": "no_pairs_selected"}],
        )
        design_manifest = {
            "design_gate": gate,
            "design_id": DESIGN_ID,
            "schema_version": (
                "external-behavior-v4-directional-design-manifest-v1"
            ),
            "status": "design_no_go",
            "training_released": False,
        }
    write_json(MANIFEST_DIR / "design_manifest.json", design_manifest)
    (ANALYSIS_DIR / "report_cn.md").write_text(
        _report(gate, rows),
        encoding="utf-8",
    )
    output_hashes = {
        relative(path): sha256_file(path)
        for path in (
            *[ANALYSIS_DIR / name for name in ANALYSIS_OUTPUT_NAMES],
            *[MANIFEST_DIR / name for name in DESIGN_OUTPUT_NAMES],
        )
    }
    analysis_manifest = {
        "completed_at_utc": _now(),
        "design_id": DESIGN_ID,
        "output_sha256": output_hashes,
        "protocol_input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "schema_version": (
            "external-behavior-v4-directional-analysis-manifest-v1"
        ),
        "status": "complete_pending_independent_validation",
    }
    write_json(ANALYSIS_DIR / "analysis_manifest.json", analysis_manifest)
    return {
        "design_manifest": relative(
            MANIFEST_DIR / "design_manifest.json"
        ),
        "gate": gate,
        "power_results": relative(ANALYSIS_DIR / "power_results.json"),
    }


def _checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    labels = (
        ("protocol", "Validate the frozen pre-simulation protocol."),
        ("inputs", "Validate all input and implementation hashes."),
        ("outputs", "Validate every declared output hash."),
        ("draws", "Reproduce all 100,000 standard-normal draws."),
        ("power", "Recompute Monte Carlo, analytic, and Wilson power."),
        ("block_gate", "Reapply the frozen minimum-block rule."),
        ("pair_selection", "Reconstruct the deterministic disjoint pair prefix."),
        ("design_manifest", "Validate the selected block design semantics."),
        ("finite", "Confirm all numeric outputs are finite."),
    )
    lines = ["# Directional V4 Design Integrity", ""]
    for name, label in labels:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_design() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    try:
        verify_protocol()
        check("protocol", True)
    except Exception as error:
        check("protocol", False, repr(error))
    try:
        verify_inputs()
        check("inputs", True)
    except Exception as error:
        check("inputs", False, repr(error))
    manifest = read_json(ANALYSIS_DIR / "analysis_manifest.json")
    failures = [
        name
        for name, expected in manifest["output_sha256"].items()
        if not (PROJECT_ROOT / name).is_file()
        or sha256_file(PROJECT_ROOT / name) != expected
    ]
    check("outputs", not failures, failures)
    stored_draws = load_simulation_draws()
    reproduced_draws, rows, gate = simulate_power()
    check(
        "draws",
        np.array_equal(stored_draws, reproduced_draws)
        and array_sha256(stored_draws)
        == read_json(ANALYSIS_DIR / "power_results.json")[
            "draws_sha256"
        ],
    )
    stored_results = read_json(ANALYSIS_DIR / "power_results.json")
    check(
        "power",
        rows == stored_results["simulation_rows"]
        and stored_results["analytic_agreement_max_abs"] <= 0.005,
        stored_results["analytic_agreement_max_abs"],
    )
    stored_gate = read_json(
        ANALYSIS_DIR / "block_gate_report.json"
    )["gate"]
    check("block_gate", gate == stored_gate)
    expected_pairs = (
        selected_pair_rows(gate["selected_pair_count"])
        if gate["selected_pair_count"] is not None
        else []
    )
    observed_pairs = read_csv(MANIFEST_DIR / "selected_pairs.csv")
    pair_selection_ok = (
        len(expected_pairs) == len(observed_pairs)
        and all(
            str(expected[key]) == observed[key]
            for expected, observed in zip(expected_pairs, observed_pairs)
            for key in expected
        )
    ) if expected_pairs else (
        observed_pairs == [{"status": "no_pairs_selected"}]
    )
    check("pair_selection", pair_selection_ok)
    design = read_json(MANIFEST_DIR / "design_manifest.json")
    design_ok = (
        (
            design["selected_pair_count"] == gate["selected_pair_count"]
            and design["selected_pattern_count"]
            == 2 * gate["selected_pair_count"]
            and design["block_design"]["block_count"]
            == gate["selected_block_count"]
            and design["block_design"]["total_runs"]
            == gate["selected_run_count"]
            and design["target_independence"][
                "v3_semantic_overlap_count"
            ]
            == 0
        )
        if gate["decision"] == "directional_v4_design_go"
        else design["status"] == "design_no_go"
    ) and design["training_released"] is False
    check("design_manifest", design_ok)

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
        np.isfinite(stored_draws).all()
        and finite_tree(stored_results)
        and finite_tree(design),
    )
    failed = [item for item in checks if not item["pass"]]
    report = {
        "analysis_gate": "pass" if not failed else "no_go",
        "check_count": len(checks),
        "checks": checks,
        "design_decision": stored_gate["decision"],
        "design_id": DESIGN_ID,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "schema_version": (
            "external-behavior-v4-directional-analysis-validation-v1"
        ),
        "status": "pass" if not failed else "no_go",
        "training_released": False,
        "validated_at_utc": _now(),
    }
    write_json(ANALYSIS_DIR / "analysis_validation_report.json", report)
    (ANALYSIS_DIR / "analysis_checklist.md").write_text(
        _checklist(checks),
        encoding="ascii",
    )
    analysis_names = (
        *ANALYSIS_OUTPUT_NAMES,
        "analysis_manifest.json",
        "analysis_checklist.md",
        "analysis_validation_report.json",
    )
    (ANALYSIS_DIR / "ANALYSIS_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(ANALYSIS_DIR / name)}  {name}\n"
            for name in analysis_names
        ),
        encoding="ascii",
    )
    (MANIFEST_DIR / "DESIGN_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(MANIFEST_DIR / name)}  {name}\n"
            for name in DESIGN_OUTPUT_NAMES
        ),
        encoding="ascii",
    )
    return report


def load_simulation_draws() -> np.ndarray:
    with np.load(
        ANALYSIS_DIR / "simulation_draws.npz",
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != {"standard_normal_draws"}:
            raise RuntimeError("Unexpected simulation draw archive.")
        return archive["standard_normal_draws"]
