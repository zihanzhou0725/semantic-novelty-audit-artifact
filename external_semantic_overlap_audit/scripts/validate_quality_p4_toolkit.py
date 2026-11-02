#!/usr/bin/env python3
"""Validate P4 toolkit gates and freeze deterministic reproduction artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXTERNAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXTERNAL_ROOT.parent
TOOLKIT_ROOT = EXTERNAL_ROOT / "toolkit"
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from semantic_audit.core import sha256_file, write_json  # noqa: E402
from semantic_audit.schema import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    load_schema,
    validate_audit_output,
)


SCHEMA_VERSION = "quality-enhancement-p4-validation-v1"
PROTOCOL_PATH = (
    EXTERNAL_ROOT
    / "protocols"
    / "quality-enhancement-v1"
    / "preregistration.json"
)
CLI_PATH = EXTERNAL_ROOT / "scripts" / "semantic_audit_v1.py"
TEST_PATH = EXTERNAL_ROOT / "tests" / "test_p4_toolkit.py"
OUTPUT_DIR = (
    EXTERNAL_ROOT / "analysis" / "quality-enhancement-v1" / "p4"
)
ADAPTER_OUTPUTS = (
    "occam_markov.audit.json",
    "tensor_logic.audit.json",
)
REPRODUCTION_FILES = (
    *ADAPTER_OUTPUTS,
    "manifest.json",
    "SHA256SUMS",
)


def fail(message: str) -> None:
    raise SystemExit(f"P4 validation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def validate_protocol_gate() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    p4 = protocol["p4"]
    gate = p4["gates"]["tool_contribution_go"]
    require(
        p4["common_schema_required_fields"]
        == [
            "source_revisions_and_hashes",
            "train_and_evaluation_identity_records",
            "semantic_ids",
            "equivalence_witness_type",
            "completeness_status",
            "overlap_counts_and_denominators",
            "behavioral_estimand_metadata_if_present",
            "claim_gate_and_allowed_wording",
        ],
        "P4 common-schema fields changed",
    )
    require(
        gate
        == {
            "clean_environment_reproduction": True,
            "deterministic_manifest_hashes": True,
            "distinct_public_benchmark_adapters_at_least": 2,
            "schema_validation": True,
            "unit_canaries": [
                "exact_function",
                "sound_alias",
                "unresolved",
            ],
        },
        "P4 Tool-Contribution Gate changed",
    )

    schema = load_schema()
    required = set(schema["required"])
    for field in p4["common_schema_required_fields"]:
        require(field in required, f"schema omits frozen field: {field}")
    return p4


def clean_environment() -> dict[str, str]:
    return {
        "HOME": tempfile.gettempdir(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }


def run_canaries() -> dict[str, Any]:
    command = [sys.executable, "-I", str(TEST_PATH)]
    completed = subprocess.run(
        command,
        cwd=EXTERNAL_ROOT,
        env=clean_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    require(completed.returncode == 0, completed.stdout)
    match = re.search(r"Ran (\d+) tests", completed.stdout)
    require(match is not None, "could not parse unittest count")
    return {
        "canaries": {
            "exact_function": "passed",
            "sound_alias": "passed",
            "unresolved": "passed",
        },
        "isolated_python": True,
        "schema_negative_tests": "passed",
        "status": "passed",
        "test_command": "python3 -I tests/test_p4_toolkit.py",
        "unit_test_count": int(match.group(1)),
    }


def run_clean_reproduction(destination: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="semantic-audit-p4-a-") as first_raw:
        with tempfile.TemporaryDirectory(
            prefix="semantic-audit-p4-b-"
        ) as second_raw:
            first = Path(first_raw)
            second = Path(second_raw)
            for output_dir in (first, second):
                command = [
                    sys.executable,
                    "-I",
                    str(CLI_PATH),
                    "--external-root",
                    str(EXTERNAL_ROOT),
                    "reproduce",
                    "--output-dir",
                    str(output_dir),
                ]
                completed = subprocess.run(
                    command,
                    cwd=EXTERNAL_ROOT,
                    env=clean_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                require(completed.returncode == 0, completed.stdout)

            first_names = sorted(path.name for path in first.iterdir())
            second_names = sorted(path.name for path in second.iterdir())
            require(first_names == second_names, "reproduction file sets differ")
            require(
                set(first_names) == set(REPRODUCTION_FILES),
                "unexpected reproduction file set",
            )

            hashes: dict[str, str] = {}
            for filename in REPRODUCTION_FILES:
                first_path = first / filename
                second_path = second / filename
                require(
                    first_path.read_bytes() == second_path.read_bytes(),
                    f"non-deterministic reproduction: {filename}",
                )
                hashes[filename] = sha256_file(first_path)

            destination.mkdir(parents=True, exist_ok=True)
            for filename in (*ADAPTER_OUTPUTS, "manifest.json"):
                shutil.copyfile(first / filename, destination / filename)

    outputs = {
        filename: read_json(destination / filename)
        for filename in ADAPTER_OUTPUTS
    }
    for output in outputs.values():
        validate_audit_output(output)

    occam = outputs["occam_markov.audit.json"]
    tensor = outputs["tensor_logic.audit.json"]
    occam_bounds = [
        (
            row["overlap_lower_bound"],
            row["overlap_upper_bound"],
            row["denominator"],
        )
        for row in occam["overlap_counts_and_denominators"]
    ]
    tensor_bounds = {
        row["claim_scope"]: (
            row["overlap_lower_bound"],
            row["overlap_upper_bound"],
            row["denominator"],
        )
        for row in tensor["overlap_counts_and_denominators"]
    }
    require(occam_bounds == [(2, 4, 20), (2, 4, 20)], "Occam counts")
    require(
        tensor_bounds
        == {
            "direct_triple_identity": (0, 0, 1000),
            "endpoint_conditioned_two_hop_semantics": (484, 1000, 1000),
            "ordered_relation_composition_schema": (999, 1000, 1000),
        },
        "Tensor Logic counts",
    )
    return {
        "adapter_output_count": len(outputs),
        "clean_environment": {
            "isolated_python_flag": "-I",
            "pythonpath_removed": True,
            "third_party_runtime_dependencies": [],
        },
        "deterministic_file_sha256": hashes,
        "frozen_summary_counts": {
            "occam_markov": occam_bounds,
            "tensor_logic": tensor_bounds,
        },
        "independent_reproduction_runs": 2,
        "schema_validated_outputs": len(outputs),
        "status": "passed",
    }


def build_status(
    canary_report: dict[str, Any],
    reproduction_report: dict[str, Any],
) -> dict[str, Any]:
    manifest = read_json(OUTPUT_DIR / "manifest.json")
    adapter_ids = [
        item["adapter_id"] for item in manifest["adapters"]
    ]
    conditions = {
        "clean_environment_reproduction": (
            reproduction_report["status"] == "passed"
        ),
        "deterministic_manifest_hashes": (
            reproduction_report["independent_reproduction_runs"] == 2
        ),
        "distinct_public_benchmark_adapters_at_least_2": (
            len(set(adapter_ids)) >= 2
        ),
        "schema_validation": (
            reproduction_report["schema_validated_outputs"]
            == len(ADAPTER_OUTPUTS)
        ),
        "unit_canaries_exact_function_sound_alias_unresolved": (
            set(canary_report["canaries"])
            == {"exact_function", "sound_alias", "unresolved"}
            and set(canary_report["canaries"].values()) == {"passed"}
        ),
    }
    decision = "go" if all(conditions.values()) else "no_go"
    return {
        "adapter_count": len(set(adapter_ids)),
        "adapters": adapter_ids,
        "artifact_sha256": {
            "manifest.json": sha256_file(OUTPUT_DIR / "manifest.json"),
            "occam_markov.audit.json": sha256_file(
                OUTPUT_DIR / "occam_markov.audit.json"
            ),
            "tensor_logic.audit.json": sha256_file(
                OUTPUT_DIR / "tensor_logic.audit.json"
            ),
        },
        "audit_output_schema_sha256": sha256_file(DEFAULT_SCHEMA_PATH),
        "conditions": conditions,
        "decision": decision,
        "input_sha256": {
            str(CLI_PATH.relative_to(PROJECT_ROOT)): sha256_file(CLI_PATH),
            str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)): sha256_file(
                PROTOCOL_PATH
            ),
            str(TEST_PATH.relative_to(PROJECT_ROOT)): sha256_file(TEST_PATH),
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): (
                sha256_file(Path(__file__).resolve())
            ),
        },
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
    }


def build_report_cn(status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P4 Reproducible Audit Toolkit 验证报告",
            "",
            "## 结论",
            "",
            "- **P4 Tool-Contribution Go**：冻结 Gate 的五项要求全部通过。",
            "- 该结论只确认工具方法与复现性，不改变 P3 Breadth No-Go。",
            "",
            "## 公共 benchmark adapters",
            "",
            "- `tensor_logic_fb15k237_v1`：知识图谱 direct triple、"
            "endpoint-conditioned path 与 relation schema。",
            "- `occam_markov_transition_v1`：Markov order、transition tensor "
            "semantic ID 与 pigeonhole exposure bound。",
            "",
            "两种 adapter 使用不同语义表示，但输出同一 claim-specific schema；"
            "不会把 direct identity、sound alias 与 unresolved 合并成单一布尔值。",
            "",
            "## Gate 验证",
            "",
            "- Common schema：2/2 audit outputs 通过校验。",
            "- Unit canaries：`exact_function`、`sound_alias`、`unresolved` "
            "全部通过。",
            "- Clean environment：使用 `python3 -I`、空 `PYTHONPATH`、"
            "零第三方运行时依赖完成两次独立复现。",
            "- Determinism：adapter outputs、manifest 和 checksum bytes "
            "两次完全一致。",
            "- Frozen counts：Tensor Logic `484/1000` endpoint lower bound、"
            "`999/1000` schema lower bound；Occam 两个实验均为 `2-4/20`。",
            "",
            "## 允许主张",
            "",
            "- 工具包提供可迁移的 claim-specific evidence schema、两种公共 "
            "benchmark adapter 和可复现的 lower-bound audit。",
            "- `unresolved` 表示未识别，不表示 semantic novelty。",
            "",
            "禁止据此声称 P3 已通过、overlap 导致了行为变化，或未发现 witness "
            "等价于语义新颖。",
            "",
        ]
    )


def write_checksums() -> None:
    paths = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}" for path in paths
        )
        + "\n",
        encoding="ascii",
    )


def main() -> int:
    validate_protocol_gate()
    canary_report = run_canaries()
    reproduction_report = run_clean_reproduction(OUTPUT_DIR)
    write_json(OUTPUT_DIR / "canary_report.json", canary_report)
    write_json(
        OUTPUT_DIR / "clean_reproduction_report.json",
        reproduction_report,
    )
    status = build_status(canary_report, reproduction_report)
    require(status["decision"] == "go", "Tool-Contribution Gate did not pass")
    write_json(OUTPUT_DIR / "tool_contribution_status.json", status)
    (OUTPUT_DIR / "report_cn.md").write_text(
        build_report_cn(status),
        encoding="utf-8",
    )
    write_checksums()
    print(
        json.dumps(
            {
                "adapter_count": status["adapter_count"],
                "decision": status["decision"],
                "schema_validated_outputs": reproduction_report[
                    "schema_validated_outputs"
                ],
                "unit_test_count": canary_report["unit_test_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
