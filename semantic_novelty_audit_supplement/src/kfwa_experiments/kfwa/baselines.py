from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def batch_image(batch: dict[str, Tensor]) -> Tensor:
    """Extract image tensor from a collated benchmark batch."""

    return batch["image"]


def batch_context(batch: dict[str, Tensor]) -> Tensor:
    """Extract context tensor from a collated benchmark batch."""

    return batch["context"]


class ConvFeatureExtractor(nn.Module):
    """Small 28x28 CNN used by multiple baselines."""

    def __init__(self, channels: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_dim = channels

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image).flatten(1)


class MLPContextClassifier(nn.Module):
    """Flattened image plus context MLP baseline."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 28 * 28 + context_dim
        for layer_idx in range(depth):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        image = batch_image(batch).flatten(1)
        context = batch_context(batch)
        return self.net(torch.cat([image, context], dim=-1))


class CNNContextClassifier(nn.Module):
    """CNN baseline with late context concatenation."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        channels: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.features = ConvFeatureExtractor(channels=channels, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(channels + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = self.features(batch_image(batch))
        return self.head(torch.cat([features, batch_context(batch)], dim=-1))


class OracleRemapCNNClassifier(nn.Module):
    """CNN digit classifier with explicit permutation-matrix remapping.

    This baseline tests whether the benchmark is learnable when the model is
    forced to factor the task into:

    1. predict the original digit from the image;
    2. apply the context-provided permutation matrix to the digit distribution.

    It is "oracle" only in the sense that it uses the known structure of the
    context encoding. It does not access the target label at test time.
    """

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        channels: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if target_dim != 10:
            raise ValueError("OracleRemapCNNClassifier currently assumes 10 classes.")
        if context_dim != 100:
            raise ValueError("oracle_remap requires --context-encoding perm_matrix.")
        self.features = ConvFeatureExtractor(channels=channels, dropout=dropout)
        self.digit_head = nn.Linear(channels, target_dim)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = self.features(batch_image(batch))
        digit_logits = self.digit_head(features)
        digit_probs = digit_logits.softmax(dim=-1)
        permutation = batch_context(batch).view(-1, 10, 10)
        remapped_probs = torch.bmm(digit_probs.unsqueeze(1), permutation).squeeze(1)
        return torch.log(remapped_probs.clamp_min(1e-8))


class FiLMCNNClassifier(nn.Module):
    """CNN baseline where context produces feature-wise affine modulation."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        channels: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1)
        self.context_to_film = nn.Sequential(
            nn.Linear(context_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Linear(channels, target_dim)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        image = batch_image(batch)
        context = batch_context(batch)
        hidden = F.gelu(self.conv1(image))
        hidden = F.max_pool2d(hidden, 2)
        hidden = F.gelu(self.conv2(hidden))
        gamma, beta = self.context_to_film(context).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        hidden = hidden * (1.0 + gamma) + beta
        hidden = self.dropout(hidden)
        pooled = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        return self.head(pooled)


class HyperCNNClassifier(nn.Module):
    """Context hypernetwork baseline that generates classifier corrections."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        channels: int = 64,
        rank: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.features = ConvFeatureExtractor(channels=channels, dropout=dropout)
        self.base_head = nn.Linear(channels, target_dim)
        self.rank = rank
        self.a = nn.Linear(channels, rank, bias=False)
        self.b = nn.Linear(rank, target_dim, bias=False)
        self.context_to_rank = nn.Linear(context_dim, rank)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = self.features(batch_image(batch))
        base = self.base_head(features)
        coeff = torch.tanh(self.context_to_rank(batch_context(batch)))
        low_rank = self.b(coeff * self.a(features))
        return base + low_rank


class LoRAContextCNNClassifier(nn.Module):
    """Static context-gated low-rank adapter baseline without oscillators."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        channels: int = 64,
        rank: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.features = ConvFeatureExtractor(channels=channels, dropout=dropout)
        self.context_proj = nn.Sequential(nn.Linear(context_dim, channels), nn.GELU())
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, channels, bias=False)
        self.head = nn.Linear(channels, target_dim)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = self.features(batch_image(batch))
        context_features = self.context_proj(batch_context(batch))
        adapter = self.up(self.down(features) * torch.sigmoid(self.down(context_features)))
        return self.head(features + adapter)


class PatchEmbed(nn.Module):
    """Image-to-token patch embedding for 28x28 images."""

    def __init__(self, patch_size: int, model_dim: int) -> None:
        super().__init__()
        if 28 % patch_size != 0:
            raise ValueError("patch_size must divide 28.")
        self.patch_size = patch_size
        self.grid_size = 28 // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(1, model_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: Tensor) -> Tensor:
        patches = self.proj(image)
        return patches.flatten(2).transpose(1, 2)


class TinyTransformerClassifier(nn.Module):
    """Small ViT-like baseline with a context token."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        patch_size: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(patch_size=patch_size, model_dim=model_dim)
        tokens = self.patch.num_patches + 2
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.context_proj = nn.Linear(context_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, tokens, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, target_dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        patch_tokens = self.patch(batch_image(batch))
        batch_size = patch_tokens.shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        context = self.context_proj(batch_context(batch)).unsqueeze(1)
        tokens = torch.cat([cls, context, patch_tokens], dim=1)
        return tokens + self.position[:, : tokens.shape[1]]

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self.encoder(self.forward_tokens(batch))
        return self.head(self.norm(tokens[:, 0]))


class PhaseSelfAttention(nn.Module):
    """Tiny attention block with KoPE-style phase bias."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.05) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.phase_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens: Tensor, phases: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        phase_bias = torch.cos(phases[:, :, None] - phases[:, None, :]).unsqueeze(1)
        logits = logits + self.phase_scale * phase_bias
        weights = torch.softmax(logits, dim=-1)
        mixed = torch.matmul(self.dropout(weights), v)
        mixed = mixed.transpose(1, 2).reshape(batch, length, self.model_dim)
        return self.out(mixed)


class KoPELiteClassifier(nn.Module):
    """Lightweight KoPE-style baseline with phase-biased attention."""

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        patch_size: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(patch_size=patch_size, model_dim=model_dim)
        tokens = self.patch.num_patches + 2
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.context_proj = nn.Linear(context_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, tokens, model_dim))
        base_phase = torch.linspace(0.0, 2.0 * math.pi, tokens)
        self.register_buffer("base_phase", base_phase, persistent=False)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(model_dim),
                        "attn": PhaseSelfAttention(model_dim, num_heads, dropout),
                        "norm2": nn.LayerNorm(model_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(model_dim, ff_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(ff_dim, model_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.phase_from_context = nn.Linear(context_dim, tokens)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, target_dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        patches = self.patch(batch_image(batch))
        batch_size = patches.shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        context = self.context_proj(batch_context(batch)).unsqueeze(1)
        tokens = torch.cat([cls, context, patches], dim=1)
        return tokens + self.position[:, : tokens.shape[1]]

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self.forward_tokens(batch)
        phases = self.base_phase[None, : tokens.shape[1]] + 0.1 * self.phase_from_context(batch_context(batch))[:, : tokens.shape[1]]
        for layer in self.layers:
            tokens = tokens + layer["attn"](layer["norm1"](tokens), phases)
            tokens = tokens + layer["mlp"](layer["norm2"](tokens))
        return self.head(self.norm(tokens[:, 0]))
