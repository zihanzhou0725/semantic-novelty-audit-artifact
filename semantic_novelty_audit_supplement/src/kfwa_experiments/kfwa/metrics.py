from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class MetricBundle:
    """Classification metrics collected over an epoch."""

    loss: float
    ce: float
    accuracy: float
    ece: float
    brier: float
    seen_accuracy: float | None
    unseen_accuracy: float | None
    seen_function_accuracy: float | None
    unseen_function_accuracy: float | None
    in_support_accuracy: float | None
    out_support_accuracy: float | None
    per_context_accuracy: dict[int, float]

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten metrics into CSV/JSON-friendly keys."""

        output: dict[str, float] = {
            f"{prefix}loss": self.loss,
            f"{prefix}ce": self.ce,
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}ece": self.ece,
            f"{prefix}brier": self.brier,
        }
        if self.seen_accuracy is not None:
            output[f"{prefix}seen_accuracy"] = self.seen_accuracy
        if self.unseen_accuracy is not None:
            output[f"{prefix}unseen_accuracy"] = self.unseen_accuracy
        if self.seen_function_accuracy is not None:
            output[f"{prefix}seen_function_accuracy"] = self.seen_function_accuracy
        if self.unseen_function_accuracy is not None:
            output[f"{prefix}unseen_function_accuracy"] = self.unseen_function_accuracy
        if self.in_support_accuracy is not None:
            output[f"{prefix}in_support_accuracy"] = self.in_support_accuracy
        if self.out_support_accuracy is not None:
            output[f"{prefix}out_support_accuracy"] = self.out_support_accuracy
        for context_id, value in sorted(self.per_context_accuracy.items()):
            output[f"{prefix}context_{context_id}_accuracy"] = value
        return output


def classification_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy classification loss."""

    return F.cross_entropy(logits, target)


def expected_calibration_error(
    probabilities: Tensor,
    target: Tensor,
    bins: int = 15,
) -> float:
    """Compute top-label expected calibration error."""

    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(target).float()
    ece = probabilities.new_tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if lower.item() == 0.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_weight = mask.float().mean()
            ece = ece + bin_weight * (confidence[mask].mean() - correct[mask].mean()).abs()
    return float(ece.item())


def brier_score(probabilities: Tensor, target: Tensor, num_classes: int) -> float:
    """Compute multiclass Brier score."""

    one_hot = F.one_hot(target, num_classes=num_classes).to(probabilities.dtype)
    return float((probabilities - one_hot).square().sum(dim=-1).mean().item())


def classification_metrics(
    logits: Tensor,
    target: Tensor,
    context_id: Tensor | None = None,
    seen_context: Tensor | None = None,
    seen_function: Tensor | None = None,
    query_in_support: Tensor | None = None,
    loss: float | None = None,
) -> MetricBundle:
    """Compute overall, calibration, seen/unseen, and per-context metrics."""

    logits = logits.detach()
    target = target.detach().long()
    ce = float(F.cross_entropy(logits, target).item())
    probabilities = logits.softmax(dim=-1)
    prediction = probabilities.argmax(dim=-1)
    correct = prediction.eq(target)
    accuracy = float(correct.float().mean().item())
    per_context: dict[int, float] = {}
    if context_id is not None:
        context_id = context_id.detach().long()
        for raw_id in context_id.unique(sorted=True).tolist():
            mask = context_id.eq(int(raw_id))
            per_context[int(raw_id)] = float(correct[mask].float().mean().item())

    seen_accuracy: float | None = None
    unseen_accuracy: float | None = None
    if seen_context is not None:
        seen_context = seen_context.detach().bool()
        if seen_context.any():
            seen_accuracy = float(correct[seen_context].float().mean().item())
        if (~seen_context).any():
            unseen_accuracy = float(correct[~seen_context].float().mean().item())

    seen_function_accuracy: float | None = None
    unseen_function_accuracy: float | None = None
    if seen_function is not None:
        seen_function = seen_function.detach().bool()
        if seen_function.any():
            seen_function_accuracy = float(correct[seen_function].float().mean().item())
        if (~seen_function).any():
            unseen_function_accuracy = float(correct[~seen_function].float().mean().item())

    in_support_accuracy: float | None = None
    out_support_accuracy: float | None = None
    if query_in_support is not None:
        query_in_support = query_in_support.detach().bool()
        if query_in_support.any():
            in_support_accuracy = float(correct[query_in_support].float().mean().item())
        if (~query_in_support).any():
            out_support_accuracy = float(correct[~query_in_support].float().mean().item())

    return MetricBundle(
        loss=ce if loss is None else loss,
        ce=ce,
        accuracy=accuracy,
        ece=expected_calibration_error(probabilities, target),
        brier=brier_score(probabilities, target, num_classes=logits.shape[-1]),
        seen_accuracy=seen_accuracy,
        unseen_accuracy=unseen_accuracy,
        seen_function_accuracy=seen_function_accuracy,
        unseen_function_accuracy=unseen_function_accuracy,
        in_support_accuracy=in_support_accuracy,
        out_support_accuracy=out_support_accuracy,
        per_context_accuracy=per_context,
    )
