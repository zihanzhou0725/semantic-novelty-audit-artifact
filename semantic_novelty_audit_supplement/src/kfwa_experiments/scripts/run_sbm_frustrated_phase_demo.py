from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kfwa.utils import count_parameters, get_device, set_seed
from run_sbm_phase_locking_demo import (
    DeepSetsGraphClassifier,
    GraphTransformerClassifier,
    aggregate_by_model,
    gather_query,
    parse_seeds,
    run_epoch,
    write_csv,
)


MODEL_CHOICES = [
    "label_propagation",
    "signed_label_propagation",
    "deep_sets",
    "gcn",
    "graph_transformer",
    "lop_graph",
    "lop_graph_traj_readout",
    "lop_graph_freq_traj_readout",
    "lop_graph_no_kuramoto",
]


@dataclass(frozen=True)
class FrustratedSBMConfig:
    num_nodes: int
    num_communities: int
    p_in: float
    p_out: float
    positive_strength: float
    negative_strength: float
    negative_fraction: float
    frequency_gap: float
    frequency_noise: float
    initial_phase_noise: float
    teacher_steps: int
    teacher_dt: float
    label_mode: str
    lock_threshold: float
    lock_window: int
    slip_threshold: float
    anchors_per_class: int
    seed: int


def wrap_phase(theta: Tensor) -> Tensor:
    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def circular_rank_labels(theta: Tensor, num_classes: int) -> Tensor:
    """Balanced deterministic labels from final phase order.

    We cut the circle at the largest angular gap, unwrap, then split nodes into
    equal-rank contiguous phase groups. This makes labels depend on the teacher
    dynamics while keeping every class represented in each graph.
    """

    theta = wrap_phase(theta.detach())
    sorted_angles, _ = torch.sort(theta)
    shifted = torch.roll(sorted_angles, shifts=-1)
    gaps = torch.remainder(shifted - sorted_angles, 2.0 * math.pi)
    cut = int(torch.argmax(gaps).item())
    start = sorted_angles[(cut + 1) % theta.numel()]
    unwrapped = torch.remainder(theta - start, 2.0 * math.pi)
    order = torch.argsort(unwrapped)
    labels = torch.empty(theta.numel(), dtype=torch.long)
    for rank, node in enumerate(order.tolist()):
        labels[node] = min(num_classes - 1, rank * num_classes // theta.numel())
    return labels


def balanced_rank_labels(score: Tensor, num_classes: int) -> Tensor:
    """Balanced labels from a scalar trajectory score."""

    order = torch.argsort(score.detach())
    labels = torch.empty(score.numel(), dtype=torch.long)
    for rank, node in enumerate(order.tolist()):
        labels[node] = min(num_classes - 1, rank * num_classes // score.numel())
    return labels


def simulate_teacher_kuramoto(theta: Tensor, omega: Tensor, signed_adj: Tensor, steps: int, dt: float) -> Tensor:
    abs_degree = signed_adj.abs().sum(dim=-1).clamp_min(1.0)
    norm_signed = signed_adj / abs_degree[:, None]
    for _ in range(max(steps, 1)):
        diff = theta[None, :] - theta[:, None]
        force = (norm_signed * torch.sin(diff)).sum(dim=-1)
        theta = wrap_phase(theta + dt * (omega + force))
    return theta


def simulate_teacher_kuramoto_history(theta: Tensor, omega: Tensor, signed_adj: Tensor, steps: int, dt: float) -> Tensor:
    history = [theta]
    abs_degree = signed_adj.abs().sum(dim=-1).clamp_min(1.0)
    norm_signed = signed_adj / abs_degree[:, None]
    for _ in range(max(steps, 1)):
        diff = theta[None, :] - theta[:, None]
        force = (norm_signed * torch.sin(diff)).sum(dim=-1)
        theta = wrap_phase(theta + dt * (omega + force))
        history.append(theta)
    return torch.stack(history, dim=0)


def local_order_history(history: Tensor, signed_adj: Tensor) -> Tensor:
    abs_norm = signed_adj.abs()
    abs_norm = abs_norm / abs_norm.sum(dim=-1, keepdim=True).clamp_min(1.0)
    local_cos = torch.einsum("ij,tj->ti", abs_norm, torch.cos(history))
    local_sin = torch.einsum("ij,tj->ti", abs_norm, torch.sin(history))
    return torch.sqrt(local_cos.square() + local_sin.square() + 1.0e-8).clamp_max(1.0)


def transient_lock_time_labels(
    history: Tensor,
    signed_adj: Tensor,
    num_classes: int,
    threshold: float,
    window: int,
) -> Tensor:
    """Labels from first sustained local synchronization time."""

    order = local_order_history(history, signed_adj)
    steps = order.size(0)
    window = max(1, min(window, steps))
    lock_time = torch.full((order.size(1),), float(steps + 1))
    for t in range(steps - window + 1):
        stable = (order[t : t + window] >= threshold).all(dim=0)
        first_hit = stable & (lock_time > steps)
        lock_time[first_hit] = float(t)
    fallback = (1.0 - order[-1]) + 0.01 * torch.arange(order.size(1), dtype=order.dtype)
    score = lock_time + fallback
    return balanced_rank_labels(score, num_classes)


def phase_slip_count_labels(history: Tensor, signed_adj: Tensor, num_classes: int, threshold: float) -> Tensor:
    """Labels from transient phase-slip count relative to local neighborhood phase."""

    abs_norm = signed_adj.abs()
    abs_norm = abs_norm / abs_norm.sum(dim=-1, keepdim=True).clamp_min(1.0)
    local_cos = torch.einsum("ij,tj->ti", abs_norm, torch.cos(history))
    local_sin = torch.einsum("ij,tj->ti", abs_norm, torch.sin(history))
    local_phase = torch.atan2(local_sin, local_cos)
    rel = wrap_phase(history - local_phase)
    jumps = wrap_phase(rel[1:] - rel[:-1]).abs()
    slip_count = (jumps > threshold).float().sum(dim=0)
    mean_jump = jumps.mean(dim=0)
    return balanced_rank_labels(slip_count + 0.1 * mean_jump, num_classes)


def trajectory_labels(
    history: Tensor,
    signed_adj: Tensor,
    num_classes: int,
    label_mode: str,
    lock_threshold: float,
    lock_window: int,
    slip_threshold: float,
) -> Tensor:
    if label_mode == "final_phase_rank":
        return circular_rank_labels(history[-1], num_classes)
    if label_mode == "transient_lock_time":
        return transient_lock_time_labels(history, signed_adj, num_classes, lock_threshold, lock_window)
    if label_mode == "phase_slip_count":
        return phase_slip_count_labels(history, signed_adj, num_classes, slip_threshold)
    raise ValueError(f"Unknown label_mode: {label_mode}")


class FrustratedSBMPhaseDataset(Dataset):
    """Signed SBM with labels generated by teacher Kuramoto phase locking.

    The graph contains attractive intra-community edges and a controlled amount
    of repulsive inter-community edges. The supervised class is not the planted
    SBM community id; it is derived from a teacher Kuramoto trajectory. Anchor
    nodes reveal a few trajectory labels.
    """

    def __init__(self, size: int, config: FrustratedSBMConfig) -> None:
        if config.num_nodes % config.num_communities != 0:
            raise ValueError("num_nodes must be divisible by num_communities.")
        if config.anchors_per_class < 1:
            raise ValueError("anchors_per_class must be positive.")
        if config.anchors_per_class >= config.num_nodes // config.num_communities:
            raise ValueError("anchors_per_class must leave non-anchor query candidates.")
        self.size = int(size)
        self.config = config

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        cfg = self.config
        generator = torch.Generator()
        generator.manual_seed(cfg.seed + index)

        community_size = cfg.num_nodes // cfg.num_communities
        planted = torch.arange(cfg.num_communities).repeat_interleave(community_size)
        perm = torch.randperm(cfg.num_nodes, generator=generator)
        planted = planted[perm]

        same = planted[:, None] == planted[None, :]
        probs = torch.where(
            same,
            torch.full((cfg.num_nodes, cfg.num_nodes), cfg.p_in),
            torch.full((cfg.num_nodes, cfg.num_nodes), cfg.p_out),
        )
        edges = torch.rand((cfg.num_nodes, cfg.num_nodes), generator=generator) < probs
        edges = torch.triu(edges, diagonal=1)
        edges = edges | edges.T

        cross_negative = torch.rand((cfg.num_nodes, cfg.num_nodes), generator=generator) < cfg.negative_fraction
        weights = torch.zeros(cfg.num_nodes, cfg.num_nodes)
        weights = torch.where(same, torch.full_like(weights, cfg.positive_strength), weights)
        cross_weight = torch.where(
            cross_negative,
            torch.full_like(weights, -cfg.negative_strength),
            torch.full_like(weights, 0.25 * cfg.positive_strength),
        )
        weights = torch.where(same, weights, cross_weight)
        signed_adj = torch.where(edges, weights, torch.zeros_like(weights)).float()
        signed_adj = 0.5 * (signed_adj + signed_adj.T)
        signed_adj.fill_diagonal_(0.0)

        phase_codes = torch.linspace(-math.pi, math.pi, steps=cfg.num_communities + 1)[:-1]
        omega_means = torch.linspace(-cfg.frequency_gap, cfg.frequency_gap, steps=cfg.num_communities)
        omega = omega_means[planted] + cfg.frequency_noise * torch.randn(cfg.num_nodes, generator=generator)
        theta0 = phase_codes[planted] + cfg.initial_phase_noise * torch.randn(cfg.num_nodes, generator=generator)
        history = simulate_teacher_kuramoto_history(
            wrap_phase(theta0),
            omega.float(),
            signed_adj,
            steps=cfg.teacher_steps,
            dt=cfg.teacher_dt,
        )
        theta = history[-1]
        labels = trajectory_labels(
            history,
            signed_adj,
            cfg.num_communities,
            cfg.label_mode,
            cfg.lock_threshold,
            cfg.lock_window,
            cfg.slip_threshold,
        )

        anchor_mask = torch.zeros(cfg.num_nodes, dtype=torch.bool)
        anchor_labels = torch.zeros(cfg.num_nodes, cfg.num_communities)
        for phase_class in range(cfg.num_communities):
            candidates = torch.nonzero(labels == phase_class, as_tuple=False).flatten()
            if candidates.numel() <= cfg.anchors_per_class:
                # The rank-label fallback should keep this unreachable for the
                # default balanced setup, but keep a clear error if parameters
                # are changed aggressively.
                raise RuntimeError("Not enough nodes in a phase class for anchors and query candidates.")
            chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[: cfg.anchors_per_class]]
            anchor_mask[chosen] = True
            anchor_labels[chosen, phase_class] = 1.0

        query_candidates = torch.nonzero(~anchor_mask, as_tuple=False).flatten()
        query_index = query_candidates[torch.randint(query_candidates.numel(), (1,), generator=generator).item()]
        query_mask = torch.zeros(cfg.num_nodes)
        query_mask[query_index] = 1.0

        abs_adj = signed_adj.abs()
        pos_adj = signed_adj.clamp_min(0.0)
        neg_adj = (-signed_adj).clamp_min(0.0)
        pos_degree = pos_adj.sum(dim=-1, keepdim=True) / max(cfg.num_nodes - 1, 1)
        neg_degree = neg_adj.sum(dim=-1, keepdim=True) / max(cfg.num_nodes - 1, 1)
        signed_degree = signed_adj.sum(dim=-1, keepdim=True) / max(cfg.num_nodes - 1, 1)
        omega_scale = max(abs(cfg.frequency_gap) + 3.0 * cfg.frequency_noise, 1.0e-6)
        omega_feature = (omega[:, None] / omega_scale).float()
        features = torch.cat(
            [
                torch.ones(cfg.num_nodes, 1),
                pos_degree,
                neg_degree,
                signed_degree,
                omega_feature,
                torch.cos(theta0)[:, None],
                torch.sin(theta0)[:, None],
                query_mask[:, None],
                anchor_mask.float()[:, None],
                anchor_labels,
            ],
            dim=-1,
        )
        batch = {
            "features": features.float(),
            "adj": (abs_adj > 0.0).float(),
            "pos_adj": pos_adj.float(),
            "neg_adj": neg_adj.float(),
            "signed_adj": signed_adj.float(),
            "omega": omega.float(),
            "initial_theta": wrap_phase(theta0).float(),
            "query_index": query_index.clone().long(),
            "labels": labels.long(),
            "planted": planted.long(),
            "teacher_theta": theta.float(),
            "anchor_mask": anchor_mask,
            "anchor_labels": anchor_labels.float(),
        }
        return batch, labels[query_index].long()


def collate_graphs(items: list[tuple[dict[str, Tensor], Tensor]]) -> tuple[dict[str, Tensor], Tensor]:
    keys = items[0][0].keys()
    batch = {key: torch.stack([item[0][key] for item in items], dim=0) for key in keys}
    targets = torch.stack([item[1] for item in items], dim=0)
    return batch, targets


def normalize_weighted_adj(adj: Tensor, add_eye: bool = False) -> Tensor:
    if add_eye:
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj = adj + eye[None, :, :]
    degree = adj.sum(dim=-1).clamp_min(1.0)
    deg_inv_sqrt = degree.rsqrt()
    return deg_inv_sqrt[:, :, None] * adj * deg_inv_sqrt[:, None, :]


class LabelPropagationClassifier(nn.Module):
    """Anchor-label diffusion baseline.

    The standard setting uses the unsigned graph topology, matching the Demo A
    baseline. The signed setting is kept as a stronger diagnostic baseline that
    can exploit repulsive coupling directly.
    """

    def __init__(self, steps: int, num_classes: int, use_signed: bool) -> None:
        super().__init__()
        self.steps = steps
        self.use_signed = use_signed
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        if self.use_signed:
            adj = batch["signed_adj"]
            denom_fn = Tensor.abs
        else:
            adj = batch["adj"]
            denom_fn = lambda value: value
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj_hat = adj + eye[None, :, :]
        norm = adj_hat / denom_fn(adj_hat).sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchor_labels = batch["anchor_labels"]
        anchor_mask = anchor_labels.sum(dim=-1, keepdim=True).clamp_max(1.0)
        belief = anchor_labels
        for _ in range(max(self.steps, 1)):
            belief = norm @ belief
            belief = belief * (1.0 - anchor_mask) + anchor_labels * anchor_mask
        return self.logit_scale.clamp_min(0.0) * gather_query(belief, batch["query_index"]) + self.bias


class SignedGCNGraphClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(layers, 1)))
        self.pos_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(layers, 1)))
        self.neg_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(max(layers, 1)))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        pos_norm = normalize_weighted_adj(batch["pos_adj"])
        neg_norm = normalize_weighted_adj(batch["neg_adj"])
        h = F.gelu(self.input(batch["features"]))
        for self_layer, pos_layer, neg_layer in zip(self.self_layers, self.pos_layers, self.neg_layers):
            dropped = self.dropout(h)
            h = F.gelu(self_layer(dropped) + pos_norm @ pos_layer(dropped) + neg_norm @ neg_layer(dropped))
        return self.out(gather_query(h, batch["query_index"]))


class SignedLOPGraphClassifier(nn.Module):
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
        use_trajectory_readout: bool = False,
        use_frequency_aware_readout: bool = False,
    ) -> None:
        super().__init__()
        if use_frequency_aware_readout and not use_trajectory_readout:
            raise ValueError("frequency-aware readout requires trajectory readout")
        self.oscillator_dim = oscillator_dim
        self.steps = steps
        self.dt = dt
        self.use_kuramoto = use_kuramoto
        self.use_trajectory_readout = use_trajectory_readout
        self.use_frequency_aware_readout = use_frequency_aware_readout
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
        self.omega_gain = nn.Parameter(torch.ones(oscillator_dim))
        self.coupling = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, oscillator_dim))
        self.belief_logit_scale = nn.Parameter(torch.tensor(2.0))
        trajectory_groups = 0
        if use_trajectory_readout:
            trajectory_groups += 6
        if use_frequency_aware_readout:
            trajectory_groups += 8
        trajectory_dim = trajectory_groups * oscillator_dim
        readout_dim = hidden_dim + (2 * oscillator_dim) + oscillator_dim + trajectory_dim + num_classes + num_classes
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

    def evolve(self, theta: Tensor, omega: Tensor, signed_adj: Tensor, coupling: Tensor, anchor_labels: Tensor) -> Tensor:
        if not self.use_kuramoto or self.steps <= 0:
            return theta
        norm = signed_adj / signed_adj.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        for _ in range(self.steps):
            diff = theta[:, None, :, :] - theta[:, :, None, :]
            sync = (norm[:, :, :, None] * torch.sin(diff)).sum(dim=2)
            theta = wrap_phase(theta + self.dt * (omega + coupling * sync))
            theta = self.apply_anchor_boundary(theta, anchor_labels)
        return theta

    def evolve_history(self, theta: Tensor, omega: Tensor, signed_adj: Tensor, coupling: Tensor, anchor_labels: Tensor) -> Tensor:
        history = [theta]
        if not self.use_kuramoto or self.steps <= 0:
            return torch.stack(history, dim=1)
        norm = signed_adj / signed_adj.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        for _ in range(self.steps):
            diff = theta[:, None, :, :] - theta[:, :, None, :]
            sync = (norm[:, :, :, None] * torch.sin(diff)).sum(dim=2)
            theta = wrap_phase(theta + self.dt * (omega + coupling * sync))
            theta = self.apply_anchor_boundary(theta, anchor_labels)
            history.append(theta)
        return torch.stack(history, dim=1)

    def local_order_from_history(self, history: Tensor, signed_adj: Tensor) -> Tensor:
        abs_norm = signed_adj.abs()
        abs_norm = abs_norm / abs_norm.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_cos = torch.einsum("bij,btjd->btid", abs_norm, torch.cos(history))
        local_sin = torch.einsum("bij,btjd->btid", abs_norm, torch.sin(history))
        return torch.sqrt(local_cos.square() + local_sin.square() + 1.0e-8).clamp_max(1.0)

    def trajectory_readout(self, history: Tensor, signed_adj: Tensor, query_index: Tensor, omega: Tensor) -> Tensor:
        local_order = self.local_order_from_history(history, signed_adj)
        mean_order = local_order.mean(dim=1)
        max_order = local_order.max(dim=1).values
        order_gain = local_order[:, -1] - local_order[:, 0]
        if history.size(1) > 1:
            phase_delta = wrap_phase(history[:, 1:] - history[:, :-1]).abs()
            phase_mobility = phase_delta.mean(dim=1)
            slip_proxy = torch.sigmoid(6.0 * (phase_delta - 1.0)).mean(dim=1)
        else:
            phase_mobility = torch.zeros_like(mean_order)
            slip_proxy = torch.zeros_like(mean_order)
        lock_exposure = torch.sigmoid(10.0 * (local_order - 0.55)).mean(dim=1)
        features = [
            gather_query(mean_order, query_index),
            gather_query(max_order, query_index),
            gather_query(order_gain, query_index),
            gather_query(phase_mobility, query_index),
            gather_query(slip_proxy, query_index),
            gather_query(lock_exposure, query_index),
        ]
        if self.use_frequency_aware_readout:
            if history.size(1) > 1:
                velocity = wrap_phase(history[:, 1:] - history[:, :-1]) / max(self.dt, 1.0e-8)
                signed_mean_velocity = velocity.mean(dim=1)
                mean_speed = velocity.abs().mean(dim=1)
                residual = velocity - omega[:, None, :, :]
                mean_abs_residual = residual.abs().mean(dim=1)
                max_abs_residual = residual.abs().max(dim=1).values
                centered_velocity = velocity - signed_mean_velocity[:, None, :, :]
                velocity_variability = torch.sqrt(centered_velocity.square().mean(dim=1) + 1.0e-8)
                final_abs_residual = residual[:, -1].abs()
            else:
                signed_mean_velocity = torch.zeros_like(mean_order)
                mean_speed = torch.zeros_like(mean_order)
                mean_abs_residual = torch.zeros_like(mean_order)
                max_abs_residual = torch.zeros_like(mean_order)
                velocity_variability = torch.zeros_like(mean_order)
                final_abs_residual = torch.zeros_like(mean_order)

            abs_norm = signed_adj.abs()
            abs_norm = abs_norm / abs_norm.sum(dim=-1, keepdim=True).clamp_min(1.0)
            neighbor_omega = abs_norm @ omega
            neighbor_frequency_gap = (omega - neighbor_omega).abs()
            lock_gate = torch.sigmoid(10.0 * (local_order - 0.55))
            frequency_normalized_lock = (lock_gate / (1.0 + neighbor_frequency_gap[:, None, :, :])).mean(dim=1)
            features.extend(
                [
                    gather_query(signed_mean_velocity, query_index),
                    gather_query(mean_speed, query_index),
                    gather_query(mean_abs_residual, query_index),
                    gather_query(max_abs_residual, query_index),
                    gather_query(velocity_variability, query_index),
                    gather_query(final_abs_residual, query_index),
                    gather_query(neighbor_frequency_gap, query_index),
                    gather_query(frequency_normalized_lock, query_index),
                ]
            )
        return torch.cat(features, dim=-1)

    def anchor_sync_logits(self, theta: Tensor, query_index: Tensor, anchor_labels: Tensor) -> Tensor:
        query_theta = gather_query(theta, query_index)
        denom = anchor_labels.sum(dim=1).clamp_min(1.0)
        proto_cos = torch.einsum("bnc,bnd->bcd", anchor_labels, torch.cos(theta)) / denom[:, :, None]
        proto_sin = torch.einsum("bnc,bnd->bcd", anchor_labels, torch.sin(theta)) / denom[:, :, None]
        query_cos = torch.cos(query_theta)[:, None, :]
        query_sin = torch.sin(query_theta)[:, None, :]
        return (query_cos * proto_cos + query_sin * proto_sin).mean(dim=-1)

    def diffuse_anchor_beliefs(self, signed_adj: Tensor, anchor_labels: Tensor) -> Tensor:
        if not self.use_kuramoto or self.steps <= 0:
            return anchor_labels
        eye = torch.eye(signed_adj.size(-1), device=signed_adj.device, dtype=signed_adj.dtype)
        signed_hat = signed_adj + eye[None, :, :]
        norm = signed_hat / signed_hat.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
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
        observed_omega = batch["omega"][:, :, None] * self.omega_gain[None, None, :]
        omega = observed_omega + 0.25 * torch.tanh(self.frequency(h))
        coupling = F.softplus(self.coupling(h))
        if self.use_trajectory_readout:
            history = self.evolve_history(theta, omega, batch["signed_adj"], coupling, batch["anchor_labels"])
            theta = history[:, -1]
        else:
            history = None
            theta = self.evolve(theta, omega, batch["signed_adj"], coupling, batch["anchor_labels"])

        cos_sin = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        abs_norm = batch["signed_adj"].abs()
        abs_norm = abs_norm / abs_norm.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_cos = abs_norm @ torch.cos(theta)
        local_sin = abs_norm @ torch.sin(theta)
        local_order = torch.sqrt(local_cos.square() + local_sin.square() + 1.0e-8).clamp_max(1.0)

        query_h = gather_query(h, batch["query_index"])
        query_phase = gather_query(cos_sin, batch["query_index"])
        query_order = gather_query(local_order, batch["query_index"])
        trajectory_features = (
            self.trajectory_readout(history, batch["signed_adj"], batch["query_index"], omega)
            if history is not None
            else torch.empty(query_h.size(0), 0, device=query_h.device, dtype=query_h.dtype)
        )
        sync_logits = self.anchor_sync_logits(theta, batch["query_index"], batch["anchor_labels"])
        belief_logits = gather_query(self.diffuse_anchor_beliefs(batch["signed_adj"], batch["anchor_labels"]), batch["query_index"])
        neural_logits = self.out(torch.cat([query_h, query_phase, query_order, trajectory_features, sync_logits, belief_logits], dim=-1))
        return neural_logits + self.belief_logit_scale * belief_logits


def make_model(args: argparse.Namespace, model_name: str, feature_dim: int) -> nn.Module:
    if model_name == "label_propagation":
        return LabelPropagationClassifier(args.kuramoto_steps, args.num_communities, use_signed=False)
    if model_name == "signed_label_propagation":
        return LabelPropagationClassifier(args.kuramoto_steps, args.num_communities, use_signed=True)
    if model_name == "deep_sets":
        return DeepSetsGraphClassifier(feature_dim, args.hidden_dim, args.num_communities, args.dropout)
    if model_name == "gcn":
        return SignedGCNGraphClassifier(feature_dim, args.hidden_dim, args.layers, args.num_communities, args.dropout)
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
        return SignedLOPGraphClassifier(
            feature_dim,
            args.hidden_dim,
            args.oscillator_dim,
            args.kuramoto_steps,
            args.dt,
            args.num_communities,
            args.dropout,
            use_kuramoto=True,
        )
    if model_name == "lop_graph_traj_readout":
        return SignedLOPGraphClassifier(
            feature_dim,
            args.hidden_dim,
            args.oscillator_dim,
            args.kuramoto_steps,
            args.dt,
            args.num_communities,
            args.dropout,
            use_kuramoto=True,
            use_trajectory_readout=True,
        )
    if model_name == "lop_graph_freq_traj_readout":
        return SignedLOPGraphClassifier(
            feature_dim,
            args.hidden_dim,
            args.oscillator_dim,
            args.kuramoto_steps,
            args.dt,
            args.num_communities,
            args.dropout,
            use_kuramoto=True,
            use_trajectory_readout=True,
            use_frequency_aware_readout=True,
        )
    if model_name == "lop_graph_no_kuramoto":
        return SignedLOPGraphClassifier(
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


def build_loader(
    args: argparse.Namespace,
    size: int,
    num_nodes: int,
    p_out: float,
    negative_fraction: float,
    frequency_gap: float,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    cfg = FrustratedSBMConfig(
        num_nodes=num_nodes,
        num_communities=args.num_communities,
        p_in=args.p_in,
        p_out=p_out,
        positive_strength=args.positive_strength,
        negative_strength=args.negative_strength,
        negative_fraction=negative_fraction,
        frequency_gap=frequency_gap,
        frequency_noise=args.frequency_noise,
        initial_phase_noise=args.initial_phase_noise,
        teacher_steps=args.teacher_steps,
        teacher_dt=args.teacher_dt,
        label_mode=args.label_mode,
        lock_threshold=args.lock_threshold,
        lock_window=args.lock_window,
        slip_threshold=args.slip_threshold,
        anchors_per_class=args.anchors_per_class,
        seed=seed,
    )
    dataset = FrustratedSBMPhaseDataset(size=size, config=cfg)
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


def parse_int_values(raw: str) -> list[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def parse_float_values(raw: str) -> list[float]:
    return [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def train_one(args: argparse.Namespace, model_name: str, seed: int, device: torch.device, out_dir: Path) -> dict[str, Any]:
    set_seed(seed)
    feature_dim = 9 + args.num_communities
    model = make_model(args, model_name, feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(args, args.train_graphs, args.train_nodes, args.p_out, args.negative_fraction, args.frequency_gap, seed + 10_000, True)
    val_loader = build_loader(args, args.val_graphs, args.train_nodes, args.p_out, args.negative_fraction, args.frequency_gap, seed + 20_000, False)
    test_loader = build_loader(args, args.test_graphs, args.train_nodes, args.p_out, args.negative_fraction, args.frequency_gap, seed + 30_000, False)
    ood_size_loader = build_loader(args, args.test_graphs, args.test_nodes, args.p_out, args.negative_fraction, args.frequency_gap, seed + 40_000, False)
    ood_frustration_loader = build_loader(
        args,
        args.test_graphs,
        args.train_nodes,
        args.ood_p_out,
        args.ood_negative_fraction,
        args.frequency_gap,
        seed + 50_000,
        False,
    )
    ood_frequency_loader = build_loader(
        args,
        args.test_graphs,
        args.train_nodes,
        args.p_out,
        args.negative_fraction,
        args.ood_frequency_gap,
        seed + 60_000,
        False,
    )

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
    ood_frustration_metrics = run_epoch(model, ood_frustration_loader, device, optimizer=None, show_progress=False, desc="ood-frustration")
    ood_frequency_metrics = run_epoch(model, ood_frequency_loader, device, optimizer=None, show_progress=False, desc="ood-frequency")
    eval_time = time.perf_counter() - eval_start

    run_name = f"sbm_frustrated_phase__{model_name}__seed{seed}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "history.csv", history)
    checkpoint_path = ""
    if args.save_checkpoints and best_state is not None:
        checkpoint_path = str(run_dir / "best_model.pt")
        torch.save(
            {
                "schema_version": 1,
                "task": "sbm_frustrated_phase",
                "model": model_name,
                "seed": seed,
                "feature_dim": feature_dim,
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val,
                "model_state": best_state,
            },
            checkpoint_path,
        )
    metrics = {
        "task": "sbm_frustrated_phase",
        "model": model_name,
        "seed": seed,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "ood_size_accuracy": ood_size_metrics["accuracy"],
        "ood_size_loss": ood_size_metrics["loss"],
        "ood_frustration_accuracy": ood_frustration_metrics["accuracy"],
        "ood_frustration_loss": ood_frustration_metrics["loss"],
        "ood_frequency_accuracy": ood_frequency_metrics["accuracy"],
        "ood_frequency_loss": ood_frequency_metrics["loss"],
        "train_wall_seconds": train_time,
        "eval_wall_seconds": eval_time,
        "seconds_per_epoch": train_time / max(len(history), 1),
        "device": str(device),
        "run_dir": str(run_dir),
        "num_nodes_train": args.train_nodes,
        "num_nodes_test": args.test_nodes,
        "num_communities": args.num_communities,
        "anchors_per_class": args.anchors_per_class,
        "p_in": args.p_in,
        "p_out": args.p_out,
        "ood_p_out": args.ood_p_out,
        "positive_strength": args.positive_strength,
        "negative_strength": args.negative_strength,
        "negative_fraction": args.negative_fraction,
        "ood_negative_fraction": args.ood_negative_fraction,
        "frequency_gap": args.frequency_gap,
        "ood_frequency_gap": args.ood_frequency_gap,
        "frequency_noise": args.frequency_noise,
        "initial_phase_noise": args.initial_phase_noise,
        "teacher_steps": args.teacher_steps,
        "teacher_dt": args.teacher_dt,
        "label_mode": args.label_mode,
        "lock_threshold": args.lock_threshold,
        "lock_window": args.lock_window,
        "slip_threshold": args.slip_threshold,
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
        "checkpoint_path": checkpoint_path,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def checkpoint_run_dir(root: Path, model_name: str, seed: int) -> Path:
    return root / f"sbm_frustrated_phase__{model_name}__seed{seed}"


def make_eval_specs(args: argparse.Namespace, base_args: argparse.Namespace | None = None) -> list[dict[str, Any]]:
    base = base_args or args
    specs: list[dict[str, Any]] = []
    if args.eval_include_standard:
        specs.extend(
            [
                {
                    "condition": "test_id",
                    "sweep": "standard",
                    "sweep_value": "id",
                    "num_nodes": base.train_nodes,
                    "p_out": base.p_out,
                    "negative_fraction": base.negative_fraction,
                    "frequency_gap": base.frequency_gap,
                    "seed_offset": 30_000,
                },
                {
                    "condition": f"ood_size_nodes_{base.test_nodes}",
                    "sweep": "ood_size_nodes",
                    "sweep_value": base.test_nodes,
                    "num_nodes": base.test_nodes,
                    "p_out": base.p_out,
                    "negative_fraction": base.negative_fraction,
                    "frequency_gap": base.frequency_gap,
                    "seed_offset": 40_000,
                },
                {
                    "condition": f"ood_negative_fraction_{base.ood_negative_fraction:g}",
                    "sweep": "ood_negative_fraction",
                    "sweep_value": base.ood_negative_fraction,
                    "num_nodes": base.train_nodes,
                    "p_out": base.ood_p_out,
                    "negative_fraction": base.ood_negative_fraction,
                    "frequency_gap": base.frequency_gap,
                    "seed_offset": 50_000,
                },
                {
                    "condition": f"ood_frequency_gap_{base.ood_frequency_gap:g}",
                    "sweep": "ood_frequency_gap",
                    "sweep_value": base.ood_frequency_gap,
                    "num_nodes": base.train_nodes,
                    "p_out": base.p_out,
                    "negative_fraction": base.negative_fraction,
                    "frequency_gap": base.ood_frequency_gap,
                    "seed_offset": 60_000,
                },
            ]
        )

    for value in parse_int_values(args.eval_ood_size_nodes):
        specs.append(
            {
                "condition": f"ood_size_nodes_{value}",
                "sweep": "ood_size_nodes",
                "sweep_value": value,
                "num_nodes": value,
                "p_out": base.p_out,
                "negative_fraction": base.negative_fraction,
                "frequency_gap": base.frequency_gap,
                "seed_offset": 40_000,
            }
        )
    for value in parse_float_values(args.eval_ood_frequency_gaps):
        specs.append(
            {
                "condition": f"ood_frequency_gap_{value:g}",
                "sweep": "ood_frequency_gap",
                "sweep_value": value,
                "num_nodes": base.train_nodes,
                "p_out": base.p_out,
                "negative_fraction": base.negative_fraction,
                "frequency_gap": value,
                "seed_offset": 60_000,
            }
        )
    for value in parse_float_values(args.eval_ood_negative_fractions):
        specs.append(
            {
                "condition": f"ood_negative_fraction_{value:g}",
                "sweep": "ood_negative_fraction",
                "sweep_value": value,
                "num_nodes": base.train_nodes,
                "p_out": base.ood_p_out,
                "negative_fraction": value,
                "frequency_gap": base.frequency_gap,
                "seed_offset": 50_000,
            }
        )
    if not specs:
        raise ValueError(
            "eval-only needs at least one eval condition; pass --eval-include-standard or one of "
            "--eval-ood-size-nodes/--eval-ood-frequency-gaps/--eval-ood-negative-fractions."
        )
    return specs


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    feature_dim = int(checkpoint.get("feature_dim", 9 + checkpoint_args.num_communities))
    model_name = str(checkpoint.get("model", checkpoint_args.models))
    model = make_model(checkpoint_args, model_name, feature_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def eval_checkpoint_one(
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    device: torch.device,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    checkpoint_path = checkpoint_run_dir(checkpoint_root, model_name, seed) / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    checkpoint_args.batch_size = args.batch_size
    checkpoint_args.num_workers = args.num_workers
    checkpoint_args.test_graphs = args.eval_graphs or checkpoint_args.test_graphs
    checkpoint_args.no_progress = True
    specs = make_eval_specs(args, checkpoint_args)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        loader = build_loader(
            checkpoint_args,
            checkpoint_args.test_graphs,
            int(spec["num_nodes"]),
            float(spec["p_out"]),
            float(spec["negative_fraction"]),
            float(spec["frequency_gap"]),
            seed + int(spec["seed_offset"]),
            False,
        )
        start = time.perf_counter()
        metrics = run_epoch(model, loader, device, optimizer=None, show_progress=False, desc=str(spec["condition"]))
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "task": "sbm_frustrated_phase",
                "model": model_name,
                "seed": seed,
                "condition": spec["condition"],
                "sweep": spec["sweep"],
                "sweep_value": spec["sweep_value"],
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "eval_wall_seconds": elapsed,
                "device": str(device),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_best_epoch": checkpoint.get("best_epoch", ""),
                "checkpoint_best_val_accuracy": checkpoint.get("best_val_accuracy", ""),
                "num_nodes_eval": int(spec["num_nodes"]),
                "p_out_eval": float(spec["p_out"]),
                "negative_fraction_eval": float(spec["negative_fraction"]),
                "frequency_gap_eval": float(spec["frequency_gap"]),
                "test_graphs": checkpoint_args.test_graphs,
                "batch_size": checkpoint_args.batch_size,
                "num_communities": checkpoint_args.num_communities,
                "kuramoto_steps": checkpoint_args.kuramoto_steps,
                "dt": checkpoint_args.dt,
                "hidden_dim": checkpoint_args.hidden_dim,
                "ff_dim": checkpoint_args.ff_dim,
                "oscillator_dim": checkpoint_args.oscillator_dim,
            }
        )
    return rows


def aggregate_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["condition"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (model, condition), group in sorted(groups.items()):
        values = [float(row["accuracy"]) for row in group]
        losses = [float(row["loss"]) for row in group]
        times = [float(row["eval_wall_seconds"]) for row in group]
        mean = sum(values) / len(values)
        loss_mean = sum(losses) / len(losses)
        time_mean = sum(times) / len(times)
        var = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        summary.append(
            {
                "model": model,
                "condition": condition,
                "sweep": group[0]["sweep"],
                "sweep_value": group[0]["sweep_value"],
                "n": len(group),
                "accuracy_mean": mean,
                "accuracy_sd": math.sqrt(var),
                "loss_mean": loss_mean,
                "eval_wall_seconds_mean": time_mean,
                "num_nodes_eval": group[0]["num_nodes_eval"],
                "p_out_eval": group[0]["p_out_eval"],
                "negative_fraction_eval": group[0]["negative_fraction_eval"],
                "frequency_gap_eval": group[0]["frequency_gap_eval"],
            }
        )
    return summary


def write_eval_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# SBM Frustrated Phase Eval-Only OOD Sweep",
        "",
        "## Material Passport",
        "",
        "- Verification Status: ANALYZED",
        f"- Checkpoint root: `{args.checkpoint_root}`",
        f"- Eval rows: {len(rows)}",
        f"- Eval graphs per condition: {args.eval_graphs or 'checkpoint default'}",
        "",
        "## Aggregate",
        "",
        "| Model | Condition | n | Accuracy | Loss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {condition} | {n} | {acc:.4f} +/- {sd:.4f} | {loss:.4f} |".format(
                model=row["model"],
                condition=row["condition"],
                n=int(row["n"]),
                acc=row["accuracy_mean"],
                sd=row["accuracy_sd"],
                loss=row["loss_mean"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def eval_only(args: argparse.Namespace, device: torch.device, out_dir: Path) -> int:
    if not args.checkpoint_root:
        raise ValueError("--eval-only requires --checkpoint-root")
    checkpoint_root = Path(args.checkpoint_root)
    models = parse_models(args.models)
    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, Any]] = []
    run_specs = [(model, seed) for model in models for seed in seeds]
    for model_name, seed in run_specs:
        print(f"\n==> eval model={model_name} seed={seed} device={device}", flush=True)
        seed_rows = eval_checkpoint_one(args, model_name, seed, device, checkpoint_root)
        rows.extend(seed_rows)
        write_csv(out_dir / "eval_summary.csv", rows)
        summary = aggregate_eval(rows)
        write_csv(out_dir / "eval_group_summary.csv", summary)
        write_eval_report(out_dir / "eval_analysis.md", rows, summary, args)
        compact = ", ".join(f"{row['condition']}={row['accuracy']:.4f}" for row in seed_rows)
        print(f"<== eval {model_name} seed={seed} {compact}", flush=True)
    return 0


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for model, group in sorted(by_model.items()):
        out: dict[str, Any] = {"model": model, "n": len(group)}
        for metric in [
            "test_accuracy",
            "ood_size_accuracy",
            "ood_frustration_accuracy",
            "ood_frequency_accuracy",
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


def write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# SBM frustrated phase-locking demo analysis",
        "",
        "## Material Passport",
        "",
        "- Verification Status: ANALYZED",
        f"- Runs: {len(rows)}",
        f"- Train nodes: {args.train_nodes}; OOD size nodes: {args.test_nodes}",
        f"- Phase classes: {args.num_communities}; anchors/class: {args.anchors_per_class}",
        f"- Teacher Kuramoto: steps={args.teacher_steps}, dt={args.teacher_dt}",
        f"- Label mode: {args.label_mode}",
        f"- Coupling: p_in={args.p_in}, p_out={args.p_out}, neg_fraction={args.negative_fraction}, neg_strength={args.negative_strength}",
        "",
        "## Aggregate",
        "",
        "| Model | n | Test acc | OOD size acc | OOD frustration acc | OOD frequency acc | sec/epoch | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {n} | {test:.4f} +/- {test_sd:.4f} | {size:.4f} +/- {size_sd:.4f} | "
            "{frust:.4f} +/- {frust_sd:.4f} | {freq:.4f} +/- {freq_sd:.4f} | {sec:.3f} | {params:.0f} |".format(
                model=row["model"],
                n=int(row["n"]),
                test=row["test_accuracy_mean"],
                test_sd=row["test_accuracy_sd"],
                size=row["ood_size_accuracy_mean"],
                size_sd=row["ood_size_accuracy_sd"],
                frust=row["ood_frustration_accuracy_mean"],
                frust_sd=row["ood_frustration_accuracy_sd"],
                freq=row["ood_frequency_accuracy_mean"],
                freq_sd=row["ood_frequency_accuracy_sd"],
                sec=row["seconds_per_epoch_mean"],
                params=row["parameters_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This task changes the target from raw SBM community id to trajectory labels generated by a signed teacher Kuramoto system.",
            "The intended diagnostic is whether the explicit signed synchronization pathway helps more than ordinary graph message passing when repulsive coupling and frequency drift define the label.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="kfwa_experiments/runs_sbm_frustrated_phase_demo")
    parser.add_argument(
        "--models",
        default="label_propagation,deep_sets,gcn,graph_transformer,lop_graph,lop_graph_no_kuramoto",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")

    parser.add_argument("--train-graphs", type=int, default=1024)
    parser.add_argument("--val-graphs", type=int, default=256)
    parser.add_argument("--test-graphs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--train-nodes", type=int, default=32)
    parser.add_argument("--test-nodes", type=int, default=64)
    parser.add_argument("--num-communities", type=int, default=4)
    parser.add_argument("--anchors-per-class", type=int, default=1)
    parser.add_argument("--p-in", type=float, default=0.35)
    parser.add_argument("--p-out", type=float, default=0.10)
    parser.add_argument("--ood-p-out", type=float, default=0.18)
    parser.add_argument("--positive-strength", type=float, default=1.0)
    parser.add_argument("--negative-strength", type=float, default=1.4)
    parser.add_argument("--negative-fraction", type=float, default=0.75)
    parser.add_argument("--ood-negative-fraction", type=float, default=0.90)
    parser.add_argument("--frequency-gap", type=float, default=0.35)
    parser.add_argument("--ood-frequency-gap", type=float, default=0.50)
    parser.add_argument("--frequency-noise", type=float, default=0.08)
    parser.add_argument("--initial-phase-noise", type=float, default=0.55)
    parser.add_argument("--teacher-steps", type=int, default=32)
    parser.add_argument("--teacher-dt", type=float, default=0.15)
    parser.add_argument(
        "--label-mode",
        choices=["final_phase_rank", "transient_lock_time", "phase_slip_count"],
        default="final_phase_rank",
    )
    parser.add_argument("--lock-threshold", type=float, default=0.78)
    parser.add_argument("--lock-window", type=int, default=4)
    parser.add_argument("--slip-threshold", type=float, default=1.1)

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
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--eval-graphs", type=int, default=None)
    parser.add_argument("--eval-include-standard", action="store_true")
    parser.add_argument("--eval-ood-size-nodes", default="")
    parser.add_argument("--eval-ood-frequency-gaps", default="")
    parser.add_argument("--eval-ood-negative-fractions", default="")
    parser.set_defaults(save_checkpoints=True)


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
    args.teacher_steps = min(args.teacher_steps, 16)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run signed SBM frustrated Kuramoto phase-locking demo.")
    add_args(parser)
    args = parser.parse_args(argv)
    apply_smoke(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = get_device(args.device)
    if args.eval_only:
        return eval_only(args, device, out_dir)

    models = parse_models(args.models)
    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, Any]] = []
    run_specs = [(model, seed) for model in models for seed in seeds]
    iterator: Any = run_specs
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(run_specs, desc="SBM frustrated demo runs", unit="run", dynamic_ncols=True, file=sys.stdout)
    for model_name, seed in iterator:
        print(f"\n==> model={model_name} seed={seed} device={device}", flush=True)
        metrics = train_one(args, model_name, seed, device, out_dir)
        rows.append(metrics)
        write_csv(out_dir / "summary.csv", rows)
        summary = aggregate(rows)
        write_csv(out_dir / "group_summary.csv", summary)
        write_report(out_dir / "analysis.md", rows, summary, args)
        print(
            f"<== {model_name} seed={seed} test={metrics['test_accuracy']:.4f} "
            f"ood_size={metrics['ood_size_accuracy']:.4f} "
            f"ood_frustration={metrics['ood_frustration_accuracy']:.4f} "
            f"ood_frequency={metrics['ood_frequency_accuracy']:.4f} "
            f"sec/epoch={metrics['seconds_per_epoch']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
