from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.utils import count_parameters, get_device, set_seed


MODEL_CHOICES = [
    "label_propagation",
    "deep_sets",
    "gcn",
    "graph_transformer",
    "lop_graph",
    "lop_graph_no_kuramoto",
]


@dataclass(frozen=True)
class SBMConfig:
    num_nodes: int
    num_communities: int
    p_in: float
    p_out: float
    anchors_per_community: int
    seed: int


class SyntheticSBMCommunityDataset(Dataset):
    """Few-anchor SBM query-node community prediction.

    Each graph is node-permuted. A small number of anchor nodes reveal their
    community label. The model must infer the community of a non-anchor query
    node from graph connectivity and anchor labels.
    """

    def __init__(self, size: int, config: SBMConfig) -> None:
        if config.num_nodes % config.num_communities != 0:
            raise ValueError("num_nodes must be divisible by num_communities.")
        if config.anchors_per_community < 1:
            raise ValueError("anchors_per_community must be positive.")
        if config.anchors_per_community >= config.num_nodes // config.num_communities:
            raise ValueError("anchors_per_community must leave non-anchor query candidates.")
        self.size = int(size)
        self.config = config

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        cfg = self.config
        generator = torch.Generator()
        generator.manual_seed(cfg.seed + index)

        community_size = cfg.num_nodes // cfg.num_communities
        labels = torch.arange(cfg.num_communities).repeat_interleave(community_size)
        perm = torch.randperm(cfg.num_nodes, generator=generator)
        labels = labels[perm]

        same = labels[:, None] == labels[None, :]
        probs = torch.where(
            same,
            torch.full((cfg.num_nodes, cfg.num_nodes), cfg.p_in),
            torch.full((cfg.num_nodes, cfg.num_nodes), cfg.p_out),
        )
        upper = torch.rand((cfg.num_nodes, cfg.num_nodes), generator=generator) < probs
        upper = torch.triu(upper, diagonal=1)
        adj = (upper | upper.T).float()

        anchor_mask = torch.zeros(cfg.num_nodes, dtype=torch.bool)
        anchor_labels = torch.zeros(cfg.num_nodes, cfg.num_communities)
        for community in range(cfg.num_communities):
            candidates = torch.nonzero(labels == community, as_tuple=False).flatten()
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[: cfg.anchors_per_community]]
            anchor_mask[chosen] = True
            anchor_labels[chosen, community] = 1.0

        query_candidates = torch.nonzero(~anchor_mask, as_tuple=False).flatten()
        query_index = query_candidates[torch.randint(len(query_candidates), (1,), generator=generator).item()]
        query_mask = torch.zeros(cfg.num_nodes)
        query_mask[query_index] = 1.0

        degree = adj.sum(dim=-1, keepdim=True) / max(cfg.num_nodes - 1, 1)
        features = torch.cat(
            [
                torch.ones(cfg.num_nodes, 1),
                degree,
                query_mask[:, None],
                anchor_mask.float()[:, None],
                anchor_labels,
            ],
            dim=-1,
        )
        batch = {
            "features": features.float(),
            "adj": adj.float(),
            "query_index": query_index.clone().long(),
            "labels": labels.long(),
            "anchor_mask": anchor_mask,
            "anchor_labels": anchor_labels.float(),
        }
        return batch, labels[query_index].long()


def collate_graphs(items: list[tuple[dict[str, Tensor], Tensor]]) -> tuple[dict[str, Tensor], Tensor]:
    keys = items[0][0].keys()
    batch = {key: torch.stack([item[0][key] for item in items], dim=0) for key in keys}
    targets = torch.stack([item[1] for item in items], dim=0)
    return batch, targets


def normalize_adj(adj: Tensor) -> Tensor:
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
    adj_hat = adj + eye[None, :, :]
    deg = adj_hat.sum(dim=-1).clamp_min(1.0)
    deg_inv_sqrt = deg.rsqrt()
    return deg_inv_sqrt[:, :, None] * adj_hat * deg_inv_sqrt[:, None, :]


def gather_query(nodes: Tensor, query_index: Tensor) -> Tensor:
    batch_index = torch.arange(nodes.size(0), device=nodes.device)
    return nodes[batch_index, query_index]


class DeepSetsGraphClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.node = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.node(batch["features"])
        query = gather_query(h, batch["query_index"])
        pooled = h.mean(dim=1)
        return self.out(torch.cat([query, pooled], dim=-1))


class GCNGraphClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(layers - 1, 0)))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        norm = normalize_adj(batch["adj"])
        h = F.gelu(norm @ self.input(batch["features"]))
        for layer in self.layers:
            h = F.gelu(norm @ self.dropout(layer(h)))
        return self.out(gather_query(h, batch["query_index"]))


class LabelPropagationClassifier(nn.Module):
    """Anchor-label diffusion baseline for the few-anchor SBM task."""

    def __init__(self, steps: int, num_classes: int) -> None:
        super().__init__()
        self.steps = steps
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        adj = batch["adj"]
        anchor_labels = batch["anchor_labels"]
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj_hat = adj + eye[None, :, :]
        norm = adj_hat / adj_hat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchor_mask = anchor_labels.sum(dim=-1, keepdim=True).clamp_max(1.0)
        belief = anchor_labels
        for _ in range(max(self.steps, 1)):
            belief = norm @ belief
            belief = belief * (1.0 - anchor_mask) + anchor_labels * anchor_mask
        return self.logit_scale.clamp_min(0.0) * gather_query(belief, batch["query_index"]) + self.bias


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.edge_bias = nn.Parameter(torch.tensor(0.5))
        self.non_edge_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, h: Tensor, adj: Tensor) -> Tensor:
        residual = h
        h_norm = self.norm1(h)
        qkv = self.qkv(h_norm).view(h.size(0), h.size(1), 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        edge = (adj + eye[None, :, :]).clamp_max(1.0)[:, None, :, :]
        scores = scores + edge * self.edge_bias + (1.0 - edge) * self.non_edge_bias
        scores = scores.masked_fill(edge <= 0.0, -1.0e4)
        attn = scores.softmax(dim=-1)
        mixed = (self.dropout(attn) @ v).transpose(1, 2).reshape(h.size(0), h.size(1), -1)
        h = residual + self.dropout(self.out(mixed))
        h = h + self.dropout(self.ff(self.norm2(h)))
        return h


class GraphTransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        ff_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.input(batch["features"])
        for layer in self.layers:
            h = layer(h, batch["adj"])
        return self.out(gather_query(self.norm(h), batch["query_index"]))


class LOPGraphClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        oscillator_dim: int,
        steps: int,
        dt: float,
        num_classes: int,
        dropout: float,
        use_kuramoto: bool,
    ) -> None:
        super().__init__()
        self.oscillator_dim = oscillator_dim
        self.steps = steps
        self.dt = dt
        self.use_kuramoto = use_kuramoto
        base_codes = torch.linspace(-math.pi, math.pi, steps=num_classes + 1)[:-1]
        self.register_buffer("anchor_phase_codes", base_codes[:, None].repeat(1, oscillator_dim))
        self.node = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.phase_init = nn.Linear(hidden_dim, oscillator_dim)
        self.frequency = nn.Linear(hidden_dim, oscillator_dim)
        self.coupling = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, oscillator_dim))
        self.belief_logit_scale = nn.Parameter(torch.tensor(2.0))
        readout_dim = hidden_dim + (2 * oscillator_dim) + oscillator_dim + num_classes + num_classes
        self.out = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def apply_anchor_boundary(self, theta: Tensor, anchor_labels: Tensor) -> Tensor:
        anchor_mask = anchor_labels.sum(dim=-1, keepdim=True).clamp_max(1.0)
        anchor_theta = torch.einsum("bnc,cd->bnd", anchor_labels, self.anchor_phase_codes)
        return theta * (1.0 - anchor_mask) + anchor_theta * anchor_mask

    def evolve(self, theta: Tensor, omega: Tensor, adj: Tensor, coupling: Tensor, anchor_labels: Tensor) -> Tensor:
        if not self.use_kuramoto or self.steps <= 0:
            return theta
        norm = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        for _ in range(self.steps):
            diff = theta[:, None, :, :] - theta[:, :, None, :]
            sync = (norm[:, :, :, None] * torch.sin(diff)).sum(dim=2)
            theta = theta + self.dt * (omega + coupling * sync)
            theta = theta.clamp(min=-math.pi, max=math.pi)
            theta = self.apply_anchor_boundary(theta, anchor_labels)
        return theta

    def anchor_sync_logits(self, theta: Tensor, query_index: Tensor, anchor_labels: Tensor) -> Tensor:
        query_theta = gather_query(theta, query_index)
        denom = anchor_labels.sum(dim=1).clamp_min(1.0)
        proto_cos = torch.einsum("bnc,bnd->bcd", anchor_labels, torch.cos(theta)) / denom[:, :, None]
        proto_sin = torch.einsum("bnc,bnd->bcd", anchor_labels, torch.sin(theta)) / denom[:, :, None]
        query_cos = torch.cos(query_theta)[:, None, :]
        query_sin = torch.sin(query_theta)[:, None, :]
        return (query_cos * proto_cos + query_sin * proto_sin).mean(dim=-1)

    def diffuse_anchor_beliefs(self, adj: Tensor, anchor_labels: Tensor) -> Tensor:
        if not self.use_kuramoto or self.steps <= 0:
            return anchor_labels
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj_hat = adj + eye[None, :, :]
        norm = adj_hat / adj_hat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchor_mask = anchor_labels.sum(dim=-1, keepdim=True).clamp_max(1.0)
        belief = anchor_labels
        for _ in range(self.steps):
            belief = norm @ belief
            belief = belief * (1.0 - anchor_mask) + anchor_labels * anchor_mask
        return belief

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.node(batch["features"])
        theta = math.pi * torch.tanh(self.phase_init(h))
        theta = self.apply_anchor_boundary(theta, batch["anchor_labels"])
        omega = 0.5 * torch.tanh(self.frequency(h))
        coupling = F.softplus(self.coupling(h))
        theta = self.evolve(theta, omega, batch["adj"], coupling, batch["anchor_labels"])

        cos_sin = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        norm = batch["adj"] / batch["adj"].sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_cos = norm @ torch.cos(theta)
        local_sin = norm @ torch.sin(theta)
        local_order = torch.sqrt(local_cos.square() + local_sin.square() + 1.0e-8).clamp_max(1.0)

        query_h = gather_query(h, batch["query_index"])
        query_phase = gather_query(cos_sin, batch["query_index"])
        query_order = gather_query(local_order, batch["query_index"])
        sync_logits = self.anchor_sync_logits(theta, batch["query_index"], batch["anchor_labels"])
        belief_logits = gather_query(self.diffuse_anchor_beliefs(batch["adj"], batch["anchor_labels"]), batch["query_index"])
        neural_logits = self.out(torch.cat([query_h, query_phase, query_order, sync_logits, belief_logits], dim=-1))
        return neural_logits + self.belief_logit_scale * belief_logits


def make_model(args: argparse.Namespace, model_name: str, feature_dim: int) -> nn.Module:
    if model_name == "label_propagation":
        return LabelPropagationClassifier(args.kuramoto_steps, args.num_communities)
    if model_name == "deep_sets":
        return DeepSetsGraphClassifier(feature_dim, args.hidden_dim, args.num_communities, args.dropout)
    if model_name == "gcn":
        return GCNGraphClassifier(feature_dim, args.hidden_dim, args.layers, args.num_communities, args.dropout)
    if model_name == "graph_transformer":
        return GraphTransformerClassifier(
            feature_dim,
            args.hidden_dim,
            args.layers,
            args.heads,
            args.ff_dim,
            args.num_communities,
            args.dropout,
        )
    if model_name == "lop_graph":
        return LOPGraphClassifier(
            feature_dim,
            args.hidden_dim,
            args.oscillator_dim,
            args.kuramoto_steps,
            args.dt,
            args.num_communities,
            args.dropout,
            use_kuramoto=True,
        )
    if model_name == "lop_graph_no_kuramoto":
        return LOPGraphClassifier(
            feature_dim,
            args.hidden_dim,
            args.oscillator_dim,
            args.kuramoto_steps,
            args.dt,
            args.num_communities,
            args.dropout,
            use_kuramoto=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    show_progress: bool,
    desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    iterator: Any = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, file=sys.stdout)
    for batch, targets in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.item()) * int(targets.numel())
        total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total_examples += int(targets.numel())
        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4f}", acc=f"{total_correct / max(total_examples, 1):.3f}")
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def build_loader(args: argparse.Namespace, size: int, num_nodes: int, p_out: float, seed: int, shuffle: bool) -> DataLoader:
    cfg = SBMConfig(
        num_nodes=num_nodes,
        num_communities=args.num_communities,
        p_in=args.p_in,
        p_out=p_out,
        anchors_per_community=args.anchors_per_community,
        seed=seed,
    )
    dataset = SyntheticSBMCommunityDataset(size=size, config=cfg)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
        generator=generator,
    )


def train_one(args: argparse.Namespace, model_name: str, seed: int, device: torch.device, out_dir: Path) -> dict[str, Any]:
    set_seed(seed)
    feature_dim = 4 + args.num_communities
    model = make_model(args, model_name, feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(args, args.train_graphs, args.train_nodes, args.p_out, seed + 10_000, shuffle=True)
    val_loader = build_loader(args, args.val_graphs, args.train_nodes, args.p_out, seed + 20_000, shuffle=False)
    test_loader = build_loader(args, args.test_graphs, args.train_nodes, args.p_out, seed + 30_000, shuffle=False)
    ood_size_loader = build_loader(args, args.test_graphs, args.test_nodes, args.p_out, seed + 40_000, shuffle=False)
    ood_noise_loader = build_loader(args, args.test_graphs, args.train_nodes, args.ood_p_out, seed + 50_000, shuffle=False)

    best_val = -1.0
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            show_progress=not args.no_progress,
            desc=f"{model_name} seed={seed} train {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            show_progress=not args.no_progress,
            desc=f"{model_name} seed={seed} val {epoch}/{args.epochs}",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        if not args.no_progress:
            print(
                f"{model_name} seed={seed} epoch={epoch:03d} "
                f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.perf_counter() - start
    eval_start = time.perf_counter()
    test_metrics = run_epoch(model, test_loader, device, optimizer=None, show_progress=False, desc="test")
    ood_size_metrics = run_epoch(model, ood_size_loader, device, optimizer=None, show_progress=False, desc="ood-size")
    ood_noise_metrics = run_epoch(model, ood_noise_loader, device, optimizer=None, show_progress=False, desc="ood-noise")
    eval_time = time.perf_counter() - eval_start

    run_name = f"sbm_phase_locking__{model_name}__seed{seed}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "history.csv", history)
    metrics = {
        "task": "sbm_phase_locking",
        "model": model_name,
        "seed": seed,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "ood_size_accuracy": ood_size_metrics["accuracy"],
        "ood_size_loss": ood_size_metrics["loss"],
        "ood_noise_accuracy": ood_noise_metrics["accuracy"],
        "ood_noise_loss": ood_noise_metrics["loss"],
        "train_wall_seconds": train_time,
        "eval_wall_seconds": eval_time,
        "seconds_per_epoch": train_time / max(len(history), 1),
        "device": str(device),
        "run_dir": str(run_dir),
        "num_nodes_train": args.train_nodes,
        "num_nodes_test": args.test_nodes,
        "num_communities": args.num_communities,
        "anchors_per_community": args.anchors_per_community,
        "p_in": args.p_in,
        "p_out": args.p_out,
        "ood_p_out": args.ood_p_out,
        "train_graphs": args.train_graphs,
        "val_graphs": args.val_graphs,
        "test_graphs": args.test_graphs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "ff_dim": args.ff_dim,
        "oscillator_dim": args.oscillator_dim,
        "kuramoto_steps": args.kuramoto_steps,
        "dt": args.dt,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for model, group in sorted(by_model.items()):
        out: dict[str, Any] = {"model": model, "n": len(group)}
        for metric in [
            "test_accuracy",
            "ood_size_accuracy",
            "ood_noise_accuracy",
            "best_val_accuracy",
            "train_wall_seconds",
            "seconds_per_epoch",
            "parameters",
        ]:
            values = [float(row[metric]) for row in group]
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_sd"] = math.sqrt(var)
        summary.append(out)
    return summary


def parse_models(raw: str) -> list[str]:
    models = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    unknown = [model for model in models if model not in MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choices={MODEL_CHOICES}")
    return models


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="kfwa_experiments/runs_sbm_phase_locking_demo")
    parser.add_argument(
        "--models",
        default="label_propagation,deep_sets,gcn,graph_transformer,lop_graph,lop_graph_no_kuramoto",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--train-graphs", type=int, default=1024)
    parser.add_argument("--val-graphs", type=int, default=256)
    parser.add_argument("--test-graphs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--train-nodes", type=int, default=32)
    parser.add_argument("--test-nodes", type=int, default=64)
    parser.add_argument("--num-communities", type=int, default=4)
    parser.add_argument("--anchors-per-community", type=int, default=1)
    parser.add_argument("--p-in", type=float, default=0.35)
    parser.add_argument("--p-out", type=float, default=0.05)
    parser.add_argument("--ood-p-out", type=float, default=0.12)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--oscillator-dim", type=int, default=16)
    parser.add_argument("--kuramoto-steps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")


def apply_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.train_graphs = min(args.train_graphs, 128)
    args.val_graphs = min(args.val_graphs, 64)
    args.test_graphs = min(args.test_graphs, 64)
    args.batch_size = min(args.batch_size, 32)
    args.epochs = min(args.epochs, 3)
    args.patience = min(args.patience, 2)
    args.hidden_dim = min(args.hidden_dim, 32)
    args.ff_dim = min(args.ff_dim, 64)
    args.oscillator_dim = min(args.oscillator_dim, 8)
    args.kuramoto_steps = min(args.kuramoto_steps, 4)


def write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# SBM phase-locking demo smoke analysis",
        "",
        "## Material Passport",
        "",
        "- Verification Status: ANALYZED",
        f"- Runs: {len(rows)}",
        f"- Train nodes: {args.train_nodes}; OOD size nodes: {args.test_nodes}",
        f"- Communities: {args.num_communities}; anchors/community: {args.anchors_per_community}",
        f"- Edge probabilities: p_in={args.p_in}, p_out={args.p_out}, ood_p_out={args.ood_p_out}",
        "",
        "## Aggregate",
        "",
        "| Model | n | Test acc | OOD size acc | OOD noise acc | sec/epoch | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {n} | {test:.4f} +/- {test_sd:.4f} | {size:.4f} +/- {size_sd:.4f} | "
            "{noise:.4f} +/- {noise_sd:.4f} | {sec:.3f} | {params:.0f} |".format(
                model=row["model"],
                n=int(row["n"]),
                test=row["test_accuracy_mean"],
                test_sd=row["test_accuracy_sd"],
                size=row["ood_size_accuracy_mean"],
                size_sd=row["ood_size_accuracy_sd"],
                noise=row["ood_noise_accuracy_mean"],
                noise_sd=row["ood_noise_accuracy_sd"],
                sec=row["seconds_per_epoch_mean"],
                params=row["parameters_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This first run is intended to verify task plumbing and rough compute cost, not to establish a final result.",
            "The key diagnostic is whether graph-aware and phase-locking models separate from `deep_sets`, which ignores adjacency.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic SBM community phase-locking demo.")
    add_args(parser)
    args = parser.parse_args(argv)
    apply_smoke(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = get_device(args.device)
    models = parse_models(args.models)
    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, Any]] = []
    run_specs = [(model, seed) for model in models for seed in seeds]
    iterator: Any = run_specs
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(run_specs, desc="SBM demo runs", unit="run", dynamic_ncols=True, file=sys.stdout)
    for model_name, seed in iterator:
        print(f"\n==> model={model_name} seed={seed} device={device}", flush=True)
        metrics = train_one(args, model_name, seed, device, out_dir)
        rows.append(metrics)
        write_csv(out_dir / "summary.csv", rows)
        summary = aggregate_by_model(rows)
        write_csv(out_dir / "group_summary.csv", summary)
        write_report(out_dir / "analysis.md", rows, summary, args)
        print(
            f"<== {model_name} seed={seed} test={metrics['test_accuracy']:.4f} "
            f"ood_size={metrics['ood_size_accuracy']:.4f} ood_noise={metrics['ood_noise_accuracy']:.4f} "
            f"sec/epoch={metrics['seconds_per_epoch']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
