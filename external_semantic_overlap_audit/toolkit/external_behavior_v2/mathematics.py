"""V2 Mathematics materializer with narrow generator assertion recovery."""

from __future__ import annotations

import json
import random
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from toolkit.external_behavior.common import (
    canonical_json_bytes,
    derive_seed,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from toolkit.external_behavior.mathematics import (
    MAX_ANSWER_LENGTH,
    MAX_QUESTION_LENGTH,
    MODULES,
    REGIMES,
    _module_callable,
    canonical_polynomial,
)


RECOVERABLE_RELATIVE_PATH = "mathematics_dataset/sample/polynomials.py"
RECOVERABLE_LINE = 107
MAX_ASSERTION_DROPS = 1000


def _relative_frame_path(filename: str) -> str:
    marker = "mathematics_dataset/"
    normalized = filename.replace("\\", "/")
    if marker in normalized:
        return marker + normalized.split(marker, 1)[1]
    return Path(normalized).name


def assertion_record(
    error: AssertionError,
    *,
    regime: str,
    module_name: str,
    next_retained_row_index: int,
    anomaly_index: int,
) -> dict[str, Any]:
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        raise RuntimeError("Recoverable assertion has no traceback.")
    normalized_frames = [
        {
            "function": frame.name,
            "line": frame.lineno,
            "path": _relative_frame_path(frame.filename),
        }
        for frame in frames
    ]
    terminal = normalized_frames[-1]
    message = str(error)
    if (
        terminal["path"] != RECOVERABLE_RELATIVE_PATH
        or terminal["line"] != RECOVERABLE_LINE
        or message != ""
    ):
        raise error
    return {
        "anomaly_index": anomaly_index,
        "exception_class": type(error).__name__,
        "exception_message": message,
        "module": module_name,
        "next_retained_row_index": next_retained_row_index,
        "regime": regime,
        "terminal_frame_function": terminal["function"],
        "terminal_frame_line": terminal["line"],
        "terminal_frame_relative_path": terminal["path"],
        "traceback_sha256": sha256_bytes(
            canonical_json_bytes(normalized_frames)
        ),
    }


def sample_retained(
    module: Callable[[], Any],
    *,
    regime: str,
    module_name: str,
    next_retained_row_index: int,
    anomaly_start_index: int,
) -> tuple[Any, int, list[dict[str, Any]]]:
    length_drops = 0
    anomalies: list[dict[str, Any]] = []
    while True:
        try:
            problem = module()
        except AssertionError as error:
            anomaly_index = anomaly_start_index + len(anomalies)
            anomalies.append(
                assertion_record(
                    error,
                    regime=regime,
                    module_name=module_name,
                    next_retained_row_index=next_retained_row_index,
                    anomaly_index=anomaly_index,
                )
            )
            if anomaly_index + 1 > MAX_ASSERTION_DROPS:
                raise RuntimeError(
                    f"Assertion-drop cap exceeded for {regime}/{module_name}."
                ) from error
            continue
        question = str(problem.question)
        answer = str(problem.answer)
        if len(question) > MAX_QUESTION_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
            length_drops += 1
            continue
        return problem, length_drops, anomalies


def materialize_stream(
    output_path: Path,
    anomaly_path: Path,
    *,
    regime: str,
    module_name: str,
    retained_count: int,
    progress: Callable[[str, str, int, int], None] | None = None,
) -> dict[str, Any]:
    module = _module_callable(regime, module_name)
    seed = derive_seed(f"mathematics:{regime}:{module_name}")
    random.seed(seed)
    numpy_seed = seed % (2**32)
    np.random.seed(numpy_seed)
    length_drops = 0
    anomalies: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as handle:
        for row_index in range(retained_count):
            problem, dropped, new_anomalies = sample_retained(
                module,
                regime=regime,
                module_name=module_name,
                next_retained_row_index=row_index,
                anomaly_start_index=len(anomalies),
            )
            length_drops += dropped
            anomalies.extend(new_anomalies)
            question = str(problem.question)
            answer = str(problem.answer)
            canonical_srepr, semantic_id = canonical_polynomial(
                module_name,
                question,
            )
            row = {
                "answer": answer,
                "canonical_srepr": canonical_srepr,
                "module": module_name,
                "question": question,
                "regime": regime,
                "row_id": sha256_bytes(
                    json.dumps(
                        [
                            regime,
                            module_name,
                            row_index,
                            question,
                            answer,
                        ],
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("ascii")
                ),
                "row_index": row_index,
                "semantic_id": semantic_id,
            }
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            )
            completed = row_index + 1
            if progress is not None and (
                completed % 10_000 == 0 or completed == retained_count
            ):
                progress(regime, module_name, completed, retained_count)
    temporary.replace(output_path)
    write_jsonl(anomaly_path, anomalies)
    return {
        "anomaly_log": anomaly_path.name,
        "anomaly_log_sha256": sha256_file(anomaly_path),
        "generator_assertion_drops": len(anomalies),
        "length_drops": length_drops,
        "manifest": output_path.name,
        "manifest_sha256": sha256_file(output_path),
        "module": module_name,
        "numpy_registered_seed": seed,
        "numpy_seed_modulo_2_32": numpy_seed,
        "python_seed": seed,
        "regime": regime,
        "retained_rows": retained_count,
    }


def _progress(
    regime: str,
    module_name: str,
    completed: int,
    total: int,
) -> None:
    print(f"Mathematics v2 {regime}/{module_name} {completed}/{total}", flush=True)


def _job(
    args: tuple[str, str, str, str, int],
) -> dict[str, Any]:
    output, anomalies, regime, module_name, retained_count = args
    return materialize_stream(
        Path(output),
        Path(anomalies),
        regime=regime,
        module_name=module_name,
        retained_count=retained_count,
        progress=_progress,
    )


def materialize_all_streams(data_dir: Path) -> dict[str, Any]:
    jobs: list[tuple[str, str, str, str, int]] = []
    for regime, retained_count in REGIMES.items():
        for module_name in MODULES:
            stem = f"mathematics_{regime}_{module_name}"
            jobs.append(
                (
                    str(data_dir / f"{stem}.jsonl"),
                    str(data_dir / f"{stem}.anomalies.jsonl"),
                    regime,
                    module_name,
                    retained_count,
                )
            )
    print("Mathematics v2 launching 4 frozen streams", flush=True)
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        streams = list(executor.map(_job, jobs))
    report = {
        "aggregate_overlap_released": False,
        "generator_assertion_drops": sum(
            stream["generator_assertion_drops"] for stream in streams
        ),
        "join_or_collision_census_executed": False,
        "retained_rows": sum(stream["retained_rows"] for stream in streams),
        "status": "materialized_blind_pending_validation",
        "streams": streams,
    }
    write_json(data_dir / "mathematics_materialization_report.json", report)
    return report
