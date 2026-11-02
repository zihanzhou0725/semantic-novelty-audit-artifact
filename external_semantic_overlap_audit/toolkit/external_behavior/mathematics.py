"""DeepMind Mathematics stream materializer with exact polynomial IDs."""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy

from .common import (
    MATHEMATICS_ROOT,
    derive_seed,
    sha256_bytes,
    sha256_file,
    write_json,
)


MODULES = ("polynomials__expand", "polynomials__collect")
REGIMES = {"train": 100_000, "interpolate": 10_000}
MAX_QUESTION_LENGTH = 160
MAX_ANSWER_LENGTH = 30


def _ensure_upstream_imports() -> tuple[Any, Any]:
    sys.dont_write_bytecode = True
    root = str(MATHEMATICS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from mathematics_dataset import generate
    from mathematics_dataset.modules import polynomials

    return generate, polynomials


def parse_expression(module_name: str, question: str) -> sympy.Expr:
    if module_name == "polynomials__expand":
        prefix = "Expand "
    elif module_name == "polynomials__collect":
        prefix = "Collect the terms in "
    else:
        raise ValueError(f"Unsupported Mathematics module: {module_name}")
    if not question.startswith(prefix) or not question.endswith("."):
        raise ValueError(
            f"Question does not match the frozen parser for {module_name}: "
            f"{question!r}"
        )
    expression_text = question[len(prefix) : -1]
    expression = sympy.sympify(expression_text, evaluate=False)
    if not isinstance(expression, sympy.Expr):
        raise ValueError("Parsed Mathematics question is not an expression.")
    return expression


def alpha_normalize(expression: sympy.Expr) -> tuple[sympy.Expr, list[sympy.Symbol]]:
    source_symbols = sorted(expression.free_symbols, key=str)
    normalized_symbols = [
        sympy.Symbol(f"x{index}") for index in range(len(source_symbols))
    ]
    replacements = dict(zip(source_symbols, normalized_symbols))
    return expression.xreplace(replacements), normalized_symbols


def canonical_polynomial(
    module_name: str,
    question: str,
) -> tuple[str, str]:
    expression = parse_expression(module_name, question)
    normalized, symbols = alpha_normalize(expression)
    if not symbols:
        raise ValueError("Frozen polynomial modules must contain a free symbol.")
    polynomial = sympy.Poly(sympy.expand(normalized), *symbols)
    canonical_object = sympy.Tuple(
        *[
            sympy.Tuple(
                sympy.Tuple(*[sympy.Integer(power) for power in monomial]),
                sympy.Rational(coefficient),
            )
            for monomial, coefficient in polynomial.terms()
        ]
    )
    canonical_srepr = sympy.srepr(canonical_object)
    payload = f"{module_name}\x1f{canonical_srepr}"
    return canonical_srepr, sha256_bytes(payload.encode("ascii"))


def _sample_retained(module: Callable[[], Any]) -> tuple[Any, int]:
    dropped = 0
    while True:
        problem = module()
        question = str(problem.question)
        answer = str(problem.answer)
        if len(question) > MAX_QUESTION_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
            dropped += 1
            continue
        return problem, dropped


def _module_callable(regime: str, module_name: str) -> Callable[[], Any]:
    generate, polynomials = _ensure_upstream_imports()
    short_name = module_name.removeprefix("polynomials__")
    if regime == "train":
        available = polynomials.train(generate._make_entropy_fn(0, 1))
    elif regime == "interpolate":
        available = polynomials.test()
    else:
        raise ValueError(f"Unsupported Mathematics regime: {regime}")
    return available[short_name]


def materialize_stream(
    path: Path,
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
    dropped_total = 0
    parser_failures = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as handle:
        for row_index in range(retained_count):
            problem, dropped = _sample_retained(module)
            dropped_total += dropped
            question = str(problem.question)
            answer = str(problem.answer)
            try:
                canonical_srepr, semantic_id = canonical_polynomial(
                    module_name,
                    question,
                )
            except Exception:
                parser_failures += 1
                raise
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
    temporary.replace(path)
    return {
        "dropped_samples": dropped_total,
        "manifest": path.name,
        "manifest_sha256": sha256_file(path),
        "module": module_name,
        "numpy_registered_seed": seed,
        "numpy_seed": numpy_seed,
        "parser_failures": parser_failures,
        "python_seed": seed,
        "regime": regime,
        "retained_rows": retained_count,
    }


def _parallel_progress(
    regime: str,
    module_name: str,
    completed: int,
    total: int,
) -> None:
    print(
        f"Mathematics {regime}/{module_name} {completed}/{total}",
        flush=True,
    )


def _materialize_job(
    job: tuple[str, str, str, int],
) -> dict[str, Any]:
    path, regime, module_name, retained_count = job
    return materialize_stream(
        Path(path),
        regime=regime,
        module_name=module_name,
        retained_count=retained_count,
        progress=_parallel_progress,
    )


def materialize_mathematics(
    data_dir: Path,
    *,
    progress: Callable[[str, str, int, int], None] | None = None,
) -> dict[str, Any]:
    jobs: list[tuple[str, str, str, int]] = []
    for regime, retained_count in REGIMES.items():
        for module_name in MODULES:
            path = data_dir / f"mathematics_{regime}_{module_name}.jsonl"
            jobs.append(
                (str(path), regime, module_name, retained_count)
            )
    if progress is not None:
        print("Mathematics materialization launching 4 frozen streams", flush=True)
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        streams = list(executor.map(_materialize_job, jobs))
    report = {
        "aggregate_overlap_released": False,
        "join_or_collision_census_executed": False,
        "parser_coverage": 1.0,
        "retained_rows": sum(item["retained_rows"] for item in streams),
        "status": "materialized_blind",
        "streams": streams,
    }
    write_json(data_dir / "mathematics_materialization_report.json", report)
    return report
