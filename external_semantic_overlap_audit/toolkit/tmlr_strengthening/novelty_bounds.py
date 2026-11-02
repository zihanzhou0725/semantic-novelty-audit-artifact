"""Novelty-conditioned score intervals under bounded exposure."""

from __future__ import annotations

import math
from typing import Sequence


def novelty_score_interval(
    scores: Sequence[float],
    U: int,
) -> tuple[float, float]:
    """Return attained lower and upper novel-item mean scores."""

    item_count = len(scores)
    if not 0 <= U < item_count:
        raise ValueError("require 0 <= U < M")
    values = sorted(float(score) for score in scores)
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in values):
        raise ValueError("scores must be finite values in [0, 1]")
    novel_count = item_count - U
    return (
        sum(values[:novel_count]) / novel_count,
        sum(values[U:]) / novel_count,
    )


def binary_closed_form(
    M: int,
    k: int,
    U: int,
) -> tuple[float, float]:
    """Closed form for M binary scores with k successes."""

    if M <= 0 or not 0 <= k <= M or not 0 <= U < M:
        raise ValueError("require M > 0, 0 <= k <= M, and 0 <= U < M")
    novel_count = M - U
    return (
        max(0, k - U) / novel_count,
        min(k, novel_count) / novel_count,
    )

