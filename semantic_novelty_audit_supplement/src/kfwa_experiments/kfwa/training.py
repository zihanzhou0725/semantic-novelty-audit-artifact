from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .metrics import MetricBundle, classification_loss, classification_metrics

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class TrainResult:
    """Training result with history and test metrics."""

    best_epoch: int
    best_val_loss: float
    best_selection_metric: str
    best_selection_score: float
    history: list[dict[str, float]]
    test_metrics: MetricBundle


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """Move collated batch dictionaries to a device."""

    return {key: value.to(device) for key, value in batch.items()}


def unpack_model_output(output: Tensor | dict[str, Tensor] | tuple[Tensor, Tensor | None]) -> tuple[Tensor, Tensor | None]:
    """Return logits and an optional auxiliary loss from a model output."""

    if isinstance(output, dict):
        logits = output["logits"]
        aux_loss = output.get("aux_loss")
        return logits, aux_loss
    if isinstance(output, tuple):
        logits, aux_loss = output
        return logits, aux_loss
    return output, None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    phase: str = "train",
    show_progress: bool = True,
) -> MetricBundle:
    """Run one classification epoch."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    logits_all: list[Tensor] = []
    target_all: list[Tensor] = []
    context_all: list[Tensor] = []
    seen_all: list[Tensor] = []
    seen_function_all: list[Tensor] = []
    query_in_support_all: list[Tensor] = []

    iterator = loader
    progress_bar = None
    if show_progress and tqdm is not None:
        epoch_text = f"{epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "-"
        progress_bar = tqdm(
            loader,
            desc=f"epoch {epoch_text} {phase}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        iterator = progress_bar

    for batch, target in iterator:
        batch = move_batch(batch, device)
        target = target.to(device)
        with torch.set_grad_enabled(is_train):
            logits, aux_loss = unpack_model_output(model(batch))
            loss = classification_loss(logits, target)
            if aux_loss is not None and is_train:
                loss = loss + aux_loss
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        logits_all.append(logits.detach().cpu())
        target_all.append(target.detach().cpu())
        context_all.append(batch["context_id"].detach().cpu())
        seen_all.append(batch["seen_context"].detach().cpu())
        if "seen_function" in batch:
            seen_function_all.append(batch["seen_function"].detach().cpu())
        if "query_in_support" in batch:
            query_in_support_all.append(batch["query_in_support"].detach().cpu())
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    logits_cat = torch.cat(logits_all, dim=0)
    target_cat = torch.cat(target_all, dim=0)
    context_cat = torch.cat(context_all, dim=0)
    seen_cat = torch.cat(seen_all, dim=0)
    seen_function_cat = torch.cat(seen_function_all, dim=0) if seen_function_all else None
    query_in_support_cat = torch.cat(query_in_support_all, dim=0) if query_in_support_all else None
    return classification_metrics(
        logits_cat,
        target_cat,
        context_id=context_cat,
        seen_context=seen_cat,
        seen_function=seen_function_cat,
        query_in_support=query_in_support_cat,
        loss=total_loss / max(total_samples, 1),
    )


def format_seconds(seconds: float) -> str:
    """Format seconds as compact ETA text."""

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def metric_for_selection(metrics: MetricBundle, selection_metric: str) -> float:
    """Return the validation metric used for checkpoint selection."""

    name = selection_metric.removeprefix("val_")
    if name == "loss":
        return metrics.loss
    if name == "ce":
        return metrics.ce
    if name == "accuracy":
        return metrics.accuracy
    if name == "seen_accuracy":
        if metrics.seen_accuracy is None:
            raise ValueError("val_seen_accuracy is unavailable for this task.")
        return metrics.seen_accuracy
    if name == "unseen_accuracy":
        if metrics.unseen_accuracy is None:
            raise ValueError("val_unseen_accuracy is unavailable for this task.")
        return metrics.unseen_accuracy
    if name == "seen_function_accuracy":
        if metrics.seen_function_accuracy is None:
            raise ValueError("val_seen_function_accuracy is unavailable for this task.")
        return metrics.seen_function_accuracy
    if name == "unseen_function_accuracy":
        if metrics.unseen_function_accuracy is None:
            raise ValueError("val_unseen_function_accuracy is unavailable for this task.")
        return metrics.unseen_function_accuracy
    if name == "in_support_accuracy":
        if metrics.in_support_accuracy is None:
            raise ValueError("val_in_support_accuracy is unavailable for this task.")
        return metrics.in_support_accuracy
    if name == "out_support_accuracy":
        if metrics.out_support_accuracy is None:
            raise ValueError("val_out_support_accuracy is unavailable for this task.")
        return metrics.out_support_accuracy
    if name == "ece":
        return metrics.ece
    if name == "brier":
        return metrics.brier
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def selection_is_better(candidate: float, best: float, selection_metric: str) -> bool:
    """Compare checkpoint scores with the right optimization direction."""

    maximize = selection_metric.endswith("accuracy")
    return candidate > best if maximize else candidate < best


def initial_best_score(selection_metric: str) -> float:
    """Return an initial sentinel for a selection metric."""

    return float("-inf") if selection_metric.endswith("accuracy") else float("inf")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path | None = None,
    show_progress: bool = True,
    selection_metric: str = "val_unseen_accuracy",
) -> TrainResult:
    """Train with early stopping and evaluate the best checkpoint."""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")
    best_selection_score = initial_best_score(selection_metric)
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            phase="train",
            show_progress=show_progress,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            epoch=epoch,
            total_epochs=epochs,
            phase="val",
            show_progress=show_progress,
        )

        row = {"epoch": float(epoch)}
        row.update(train_metrics.as_dict(prefix="train_"))
        row.update(val_metrics.as_dict(prefix="val_"))
        history.append(row)

        selection_score = metric_for_selection(val_metrics, selection_metric)
        if selection_is_better(selection_score, best_selection_score, selection_metric):
            best_selection_score = selection_score
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if show_progress:
            elapsed = time.time() - start_time
            eta = elapsed / epoch * max(epochs - epoch, 0)
            print(
                f"epoch {epoch:03d}/{epochs} "
                f"train_loss={train_metrics.loss:.6f} train_acc={train_metrics.accuracy:.4f} "
                f"val_loss={val_metrics.loss:.6f} val_acc={val_metrics.accuracy:.4f} "
                f"val_seen={val_metrics.seen_accuracy if val_metrics.seen_accuracy is not None else float('nan'):.4f} "
                f"val_unseen={val_metrics.unseen_accuracy if val_metrics.unseen_accuracy is not None else float('nan'):.4f} "
                f"val_unseen_function={val_metrics.unseen_function_accuracy if val_metrics.unseen_function_accuracy is not None else float('nan'):.4f} "
                f"val_out_support={val_metrics.out_support_accuracy if val_metrics.out_support_accuracy is not None else float('nan'):.4f} "
                f"select={selection_metric}:{selection_score:.4f} "
                f"best_epoch={best_epoch} patience={stale_epochs}/{patience} eta={format_seconds(eta)}",
                flush=True,
            )

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None and best_state is not None:
        torch.save(best_state, checkpoint_path)

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        phase="test",
        show_progress=show_progress,
    )
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_selection_metric=selection_metric,
        best_selection_score=best_selection_score,
        history=history,
        test_metrics=test_metrics,
    )
