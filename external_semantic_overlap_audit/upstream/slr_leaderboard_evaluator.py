"""
Main evaluation runner: loads SLR-Bench tasks, runs a model via vLLM,
applies IPT verification, and returns aggregated metrics.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from src.about import SLR_BENCH_DATASET, TIERS
from src.evaluation.verifier import verify_hypothesis, VerificationResult


@dataclass
class TaskResult:
    task_id: int
    tier: str
    level: int
    hypothesis: str
    ext_pass: bool
    iso_pass: bool
    is_shortcut: bool
    tokens_used: int = 0
    error: str = ""


@dataclass
class EvalResult:
    model_id: str
    model_type: str = "Unknown"
    accuracy: dict = field(default_factory=dict)
    shortcuts: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)
    per_task: list = field(default_factory=list)
    eval_date: str = ""
    paper_result: bool = False


def run_evaluation(
    model_id: str,
    vllm_url: str = "http://localhost:8000/v1",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    num_tasks: Optional[int] = None,
) -> EvalResult:
    """
    Full evaluation pipeline:
    1. Load 1000 test tasks from SLR-Bench (250 per tier)
    2. Run model via vLLM OpenAI-compatible API
    3. Apply IPT verification
    4. Aggregate metrics
    """
    tasks = _load_tasks(num_tasks)
    task_results = []
    total_tokens = 0

    from openai import OpenAI
    client = OpenAI(base_url=vllm_url, api_key="EMPTY")

    for task in tasks:
        hypothesis, tokens = _query_model(client, task["prompt"], model_id, max_tokens, temperature)
        total_tokens += tokens

        if hypothesis:
            vr = verify_hypothesis(hypothesis, task["validation program"])
            task_results.append(TaskResult(
                task_id=task["id"],
                tier=task["curriculum tier"],
                level=task["curriculum level"],
                hypothesis=hypothesis,
                ext_pass=vr.extensional_pass,
                iso_pass=vr.isomorphic_pass,
                is_shortcut=vr.is_shortcut,
                tokens_used=tokens,
                error=vr.error,
            ))
        else:
            task_results.append(TaskResult(
                task_id=task["id"],
                tier=task["curriculum tier"],
                level=task["curriculum level"],
                hypothesis="",
                ext_pass=False,
                iso_pass=False,
                is_shortcut=False,
                tokens_used=tokens,
                error="No hypothesis generated",
            ))

    return _aggregate(model_id, task_results, total_tokens)


def _load_tasks(num_tasks: Optional[int]) -> list[dict]:
    ds = load_dataset(SLR_BENCH_DATASET, "v1-All", split="test")
    tasks = list(ds)
    if num_tasks:
        tasks = tasks[:num_tasks]
    return tasks


def _query_model(client, prompt: str, model_id: str, max_tokens: int, temperature: float) -> tuple[str, int]:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        hypothesis = _extract_hypothesis(content)
        return hypothesis, tokens
    except Exception as e:
        return "", 0


def _extract_hypothesis(response: str) -> str:
    """Extract the Prolog hypothesis from model response."""
    # Look for eastbound(...) :- ... pattern
    match = re.search(r"(eastbound\([^)]*\)\s*:-[^\n.]+\.)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: last non-empty line that looks like Prolog
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if "eastbound" in line.lower() and (":-" in line or "." in line):
            return line
    return response.strip()


def _aggregate(model_id: str, task_results: list[TaskResult], total_tokens: int) -> EvalResult:
    from datetime import datetime, timezone

    accuracy = {}
    shortcuts = {}
    for tier in TIERS:
        tier_tasks = [t for t in task_results if t.tier == tier]
        n = len(tier_tasks)
        if n == 0:
            continue
        acc = sum(1 for t in tier_tasks if t.ext_pass) / n * 100
        sc = sum(1 for t in tier_tasks if t.is_shortcut)
        accuracy[tier] = round(acc, 1)
        shortcuts[tier] = sc

    syntax_ok = sum(1 for t in task_results if t.hypothesis) / len(task_results) * 100 if task_results else 0

    return EvalResult(
        model_id=model_id,
        accuracy=accuracy,
        shortcuts=shortcuts,
        efficiency={
            "syntax": round(syntax_ok, 1),
            "tokens": total_tokens,
            "usd": None,
        },
        per_task=[
            {
                "id": t.task_id, "tier": t.tier, "level": t.level,
                "ext_pass": t.ext_pass, "iso_pass": t.iso_pass,
                "is_shortcut": t.is_shortcut, "tokens": t.tokens_used,
            }
            for t in task_results
        ],
        eval_date=datetime.now(timezone.utc).date().isoformat(),
    )
