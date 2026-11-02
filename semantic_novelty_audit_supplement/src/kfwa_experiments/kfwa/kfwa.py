from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baselines import PatchEmbed, batch_context, batch_image


def normalize_state(state: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize oscillator vectors onto a sphere."""

    return state / state.norm(dim=-1, keepdim=True).clamp_min(eps)


def tangent_projection(mu: Tensor, update: Tensor) -> Tensor:
    """Project an update onto the tangent space at mu."""

    return update - (update * mu).sum(dim=-1, keepdim=True) * mu


@dataclass
class KFWAState:
    """Optional diagnostics returned by a KFWA block."""

    mu: Tensor
    synchrony: Tensor
    energy: Tensor


class KuramotoFastWeightAttention(nn.Module):
    """Attention block controlled by Kuramoto oscillator fast weights.

    The block keeps the mainstream token state z, but evolves a separate
    oscillator state mu on a sphere. Synchrony between oscillator states is
    injected into attention logits, and each token receives a low-rank adapter
    generated from its oscillator state.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        oscillator_dim: int = 16,
        adapter_rank: int = 8,
        top_k: int = 16,
        steps: int = 2,
        dt: float = 0.2,
        dropout: float = 0.05,
        use_kuramoto: bool = True,
        use_sync_bias: bool = True,
        use_adapter: bool = True,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.oscillator_dim = oscillator_dim
        self.adapter_rank = adapter_rank
        self.top_k = top_k
        self.steps = steps
        self.dt = dt
        self.use_kuramoto = use_kuramoto
        self.use_sync_bias = use_sync_bias
        self.use_adapter = use_adapter

        self.norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.coupling_q = nn.Linear(model_dim, model_dim)
        self.coupling_k = nn.Linear(model_dim, model_dim)
        self.mu_value = nn.Linear(oscillator_dim, oscillator_dim, bias=False)
        self.mu_bias = nn.Linear(model_dim, oscillator_dim)
        self.omega = nn.Parameter(torch.zeros(oscillator_dim, oscillator_dim))
        self.log_sync_scale = nn.Parameter(torch.tensor(math.log(0.2)))

        self.adapter_coeff = nn.Linear(model_dim + oscillator_dim, adapter_rank)
        self.adapter_down = nn.Linear(model_dim, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, model_dim, bias=False)
        self.adapter_scale = nn.Parameter(torch.tensor(0.1))
        self._freeze_disabled_branches()

    def _freeze_disabled_branches(self) -> None:
        """Exclude disabled ablation branches from trainable parameter counts."""

        if not self.use_kuramoto:
            for module in (self.coupling_q, self.coupling_k, self.mu_value, self.mu_bias):
                for param in module.parameters():
                    param.requires_grad = False
            self.omega.requires_grad = False
        if not self.use_sync_bias:
            self.log_sync_scale.requires_grad = False
        if not self.use_adapter:
            for module in (self.adapter_coeff, self.adapter_down, self.adapter_up):
                for param in module.parameters():
                    param.requires_grad = False
            self.adapter_scale.requires_grad = False

    def skew_omega(self) -> Tensor:
        """Return a skew-symmetric frequency matrix."""

        return self.omega - self.omega.transpose(0, 1)

    def build_coupling(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Build top-k token coupling weights from the current token state."""

        batch, length, _ = tokens.shape
        q = self.coupling_q(tokens)
        k = self.coupling_k(tokens)
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(tokens.shape[-1])
        effective_k = min(self.top_k, length)
        values, indices = torch.topk(logits, k=effective_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        return weights, indices

    @staticmethod
    def gather_neighbors(values: Tensor, indices: Tensor) -> Tensor:
        """Gather [B, T, K, D] neighbors from [B, T, D]."""

        batch, length, features = values.shape
        batch_index = torch.arange(batch, device=values.device).view(batch, 1, 1)
        return values[batch_index, indices]

    def kuramoto_update(
        self,
        tokens: Tensor,
        mu: Tensor,
        weights: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evolve oscillator states and return final energy estimate."""

        if not self.use_kuramoto or self.steps <= 0:
            synchrony = torch.matmul(mu, mu.transpose(1, 2))
            energy = -0.5 * synchrony.mean(dim=(1, 2))
            return mu, energy

        omega = self.skew_omega()
        energy = tokens.new_zeros(tokens.shape[0])
        for _ in range(self.steps):
            neighbor_mu = self.gather_neighbors(mu, indices)
            coupled = (weights.unsqueeze(-1) * self.mu_value(neighbor_mu)).sum(dim=2)
            drive = self.mu_bias(tokens) + coupled
            rotation = torch.matmul(mu, omega)
            update = rotation + tangent_projection(mu, drive)
            mu = normalize_state(mu + self.dt * update)
            local_sync = (weights.unsqueeze(-1) * (mu.unsqueeze(2) * neighbor_mu).sum(dim=-1, keepdim=True)).sum(dim=(1, 2, 3))
            energy = -local_sync / max(mu.shape[1], 1)
        return mu, energy

    def attention_with_fast_weights(self, tokens: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        """Run multi-head attention with synchrony bias and dynamic adapters."""

        batch, length, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if self.use_adapter:
            coeff = torch.tanh(self.adapter_coeff(torch.cat([tokens, mu], dim=-1)))
            adapter = self.adapter_up(coeff * self.adapter_down(tokens))
            v = v + self.adapter_scale * adapter.view(batch, length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        synchrony = torch.matmul(mu, mu.transpose(1, 2))
        if self.use_sync_bias:
            logits = logits + self.log_sync_scale.exp() * synchrony.unsqueeze(1)
        weights = torch.softmax(logits, dim=-1)
        mixed = torch.matmul(self.dropout(weights), v)
        mixed = mixed.transpose(1, 2).reshape(batch, length, self.model_dim)
        return self.out(mixed), synchrony

    def forward(self, tokens: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, KFWAState]:
        normalized = self.norm(tokens)
        coupling_weights, coupling_indices = self.build_coupling(normalized)
        mu, energy = self.kuramoto_update(normalized, mu, coupling_weights, coupling_indices)
        mixed, synchrony = self.attention_with_fast_weights(normalized, mu)
        tokens = tokens + mixed
        return tokens, mu, KFWAState(mu=mu, synchrony=synchrony, energy=energy)


class KFWABlock(nn.Module):
    """KFWA attention followed by a Transformer-style MLP."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        oscillator_dim: int,
        adapter_rank: int,
        top_k: int,
        steps: int,
        dt: float,
        dropout: float,
        use_kuramoto: bool,
        use_sync_bias: bool,
        use_adapter: bool,
    ) -> None:
        super().__init__()
        self.attn = KuramotoFastWeightAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            oscillator_dim=oscillator_dim,
            adapter_rank=adapter_rank,
            top_k=top_k,
            steps=steps,
            dt=dt,
            dropout=dropout,
            use_kuramoto=use_kuramoto,
            use_sync_bias=use_sync_bias,
            use_adapter=use_adapter,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, KFWAState]:
        tokens, mu, state = self.attn(tokens, mu)
        tokens = tokens + self.mlp(self.norm(tokens))
        return tokens, mu, state


class KFWAClassifier(nn.Module):
    """KuraFlow Fast-Weight Adapter classifier for benchmark 1."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        patch_size: int = 4,
        oscillator_dim: int = 16,
        adapter_rank: int = 8,
        top_k: int = 16,
        steps: int = 2,
        dt: float = 0.2,
        dropout: float = 0.05,
        use_kuramoto: bool = True,
        use_sync_bias: bool = True,
        use_adapter: bool = True,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(patch_size=patch_size, model_dim=model_dim)
        tokens = self.patch.num_patches + 2
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.context_proj = nn.Linear(context_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, tokens, model_dim))
        self.mu_from_token = nn.Linear(model_dim, oscillator_dim)
        self.mu_position = nn.Parameter(
            self._build_2d_oscillator_init(tokens, self.patch.grid_size, oscillator_dim)
        )
        self.blocks = nn.ModuleList(
            [
                KFWABlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    oscillator_dim=oscillator_dim,
                    adapter_rank=adapter_rank,
                    top_k=top_k,
                    steps=steps,
                    dt=dt,
                    dropout=dropout,
                    use_kuramoto=use_kuramoto,
                    use_sync_bias=use_sync_bias,
                    use_adapter=use_adapter,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, target_dim)
        self.last_states: list[KFWAState] = []
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    @staticmethod
    def _build_2d_oscillator_init(tokens: int, grid_size: int, dim: int) -> Tensor:
        """Create deterministic 2D multi-frequency phase initialization."""

        position = torch.zeros(1, tokens, dim)
        if dim < 4:
            return position
        patch_start = 2
        coords = []
        for y in range(grid_size):
            for x in range(grid_size):
                coords.append((x / max(grid_size - 1, 1), y / max(grid_size - 1, 1)))
        for patch_idx, (x_coord, y_coord) in enumerate(coords):
            token_idx = patch_start + patch_idx
            for offset in range(0, dim - 3, 4):
                frequency = 2.0 ** (offset // 4)
                x_phase = 2.0 * math.pi * frequency * x_coord
                y_phase = 2.0 * math.pi * frequency * y_coord
                position[0, token_idx, offset + 0] = math.sin(x_phase)
                position[0, token_idx, offset + 1] = math.cos(x_phase)
                position[0, token_idx, offset + 2] = math.sin(y_phase)
                position[0, token_idx, offset + 3] = math.cos(y_phase)
        return position

    def initial_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        patches = self.patch(batch_image(batch))
        batch_size = patches.shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        context = self.context_proj(batch_context(batch)).unsqueeze(1)
        tokens = torch.cat([cls, context, patches], dim=1)
        return tokens + self.position[:, : tokens.shape[1]]

    def initial_mu(self, tokens: Tensor) -> Tensor:
        mu = self.mu_from_token(tokens) + self.mu_position[:, : tokens.shape[1]]
        return normalize_state(mu)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self.initial_tokens(batch)
        mu = self.initial_mu(tokens)
        states: list[KFWAState] = []
        for block in self.blocks:
            tokens, mu, state = block(tokens, mu)
            states.append(state)
        self.last_states = states
        return self.head(self.norm(tokens[:, 0]))
