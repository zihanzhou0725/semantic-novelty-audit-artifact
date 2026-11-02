from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .datasets import build_symbolic_rules, symbolic_function_representatives
from .kfwa import KFWABlock, KFWAState, normalize_state


def one_hot_digits(values: Tensor, num_symbols: int) -> Tensor:
    """Return float one-hot encodings for digit tensors."""

    return F.one_hot(values.long(), num_classes=num_symbols).float()


def symbolic_pair_features(batch: dict[str, Tensor], num_symbols: int) -> Tensor:
    """Flatten query digit and padded support pairs into one MLP input."""

    query = one_hot_digits(batch["query_digit"], num_symbols)
    support_digits = one_hot_digits(batch["support_digits"], num_symbols)
    support_labels = one_hot_digits(batch["support_labels"], num_symbols)
    support_mask = batch["support_mask"].float().unsqueeze(-1)
    support = torch.cat([support_digits, support_labels, support_mask], dim=-1).flatten(1)
    return torch.cat([query, support], dim=-1)


class SymbolicPairEmbedding(nn.Module):
    """Shared embedding helper for query digits and support mapping pairs."""

    def __init__(
        self,
        num_symbols: int = 10,
        model_dim: int = 64,
        max_support_k: int = 6,
        use_support_position: bool = True,
    ) -> None:
        super().__init__()
        self.num_symbols = num_symbols
        self.max_support_k = max_support_k
        self.query_digit = nn.Embedding(num_symbols, model_dim)
        self.source_digit = nn.Embedding(num_symbols, model_dim)
        self.target_label = nn.Embedding(num_symbols, model_dim)
        if use_support_position:
            self.support_position = nn.Parameter(
                torch.zeros(1, max_support_k, model_dim)
            )
            nn.init.trunc_normal_(self.support_position, std=0.02)
        else:
            self.register_parameter("support_position", None)

    def query(self, batch: dict[str, Tensor]) -> Tensor:
        """Embed the query digit."""

        return self.query_digit(batch["query_digit"].long())

    def pairs(self, batch: dict[str, Tensor]) -> Tensor:
        """Embed padded support pairs and zero padded positions."""

        digits = batch["support_digits"].long()
        labels = batch["support_labels"].long()
        mask = batch["support_mask"].float().unsqueeze(-1)
        tokens = self.source_digit(digits) + self.target_label(labels)
        if self.support_position is not None:
            tokens = tokens + self.support_position[:, : tokens.shape[1]]
        return tokens * mask


class SymbolicMLPClassifier(nn.Module):
    """Flat baseline over query digit and padded support pairs."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        input_dim = num_symbols + max_support_k * (2 * num_symbols + 1)
        layers: list[nn.Module] = []
        for layer_idx in range(depth):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, target_dim))
        self.num_symbols = num_symbols
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.net(symbolic_pair_features(batch, self.num_symbols))


class SymbolicDeepSetsClassifier(nn.Module):
    """Permutation-invariant support-set baseline."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(
            num_symbols=num_symbols,
            model_dim=model_dim,
            max_support_k=max_support_k,
            use_support_position=False,
        )
        self.phi = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        self.public_architecture = "Deep Sets"
        self.support_order_invariant = True

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        pair_tokens = self.phi(self.embed.pairs(batch))
        mask = batch["support_mask"].float().unsqueeze(-1)
        pooled = (pair_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([self.embed.query(batch), pooled], dim=-1))


class SymbolicGRUClassifier(nn.Module):
    """Recurrent baseline that reads support pairs as a short sequence."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(num_symbols=num_symbols, model_dim=model_dim, max_support_k=max_support_k)
        self.gru = nn.GRU(model_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        pair_tokens = self.embed.pairs(batch)
        lengths = batch["support_mask"].sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(pair_tokens, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(torch.cat([self.embed.query(batch), hidden[-1]], dim=-1))


class SymbolicTransformerClassifier(nn.Module):
    """Transformer baseline over [CLS], query, and support-pair tokens."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(num_symbols=num_symbols, model_dim=model_dim, max_support_k=max_support_k)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.query_role = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, max_support_k + 2, model_dim))
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
        self.attribution_id = "A0"
        self.readout_type = "transformer_cls"
        self.uses_explicit_rule_features = False
        self.uses_kuramoto = False
        self.effective_model_dim = model_dim
        self.effective_ff_dim = ff_dim
        self.population_size = 0
        self.effective_lop_steps = 0
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.query_role, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def token_padding_mask(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the Transformer padding mask for prefix and support tokens."""

        batch_size = batch["query_digit"].shape[0]
        return torch.cat(
            [
                torch.zeros(batch_size, 2, dtype=torch.bool, device=batch["query_digit"].device),
                ~batch["support_mask"].bool(),
            ],
            dim=1,
        )

    def initial_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        """Build [CLS, query, support-pair] input tokens."""

        batch_size = batch["query_digit"].shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        query = self.embed.query(batch).unsqueeze(1) + self.query_role
        pairs = self.embed.pairs(batch)
        tokens = torch.cat([cls, query, pairs], dim=1)
        return tokens + self.position[:, : tokens.shape[1]]

    def encode_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        """Return contextualized prefix and support tokens before classification."""

        tokens = self.initial_tokens(batch)
        return self.encoder(tokens, src_key_padding_mask=self.token_padding_mask(batch))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self.encode_tokens(batch)
        return self.head(self.norm(tokens[:, 0]))


class SymbolicRelationNetworkClassifier(nn.Module):
    """Relation Network adapted to query-conditioned support comparisons."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(
            num_symbols=num_symbols,
            model_dim=model_dim,
            max_support_k=max_support_k,
            use_support_position=False,
        )
        self.relation = nn.Sequential(
            nn.Linear(2 * model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(model_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        self.public_architecture = "Relation Network"
        self.support_order_invariant = True

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        query = self.embed.query(batch)
        pairs = self.embed.pairs(batch)
        expanded_query = query.unsqueeze(1).expand(-1, pairs.shape[1], -1)
        relations = self.relation(torch.cat([expanded_query, pairs], dim=-1))
        mask = batch["support_mask"].float().unsqueeze(-1)
        pooled = (relations * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([query, pooled], dim=-1))


class SymbolicAttentionBlock(nn.Module):
    """Residual attention and feed-forward block for public baselines."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        update, _ = self.attention(
            normalized_query,
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        state = query + self.attention_dropout(update)
        return state + self.ff(self.ff_norm(state))


class SymbolicSetTransformerClassifier(nn.Module):
    """Set Transformer with support SABs and query-conditioned pooling."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(
            num_symbols=num_symbols,
            model_dim=model_dim,
            max_support_k=max_support_k,
            use_support_position=False,
        )
        self.support_blocks = nn.ModuleList(
            [
                SymbolicAttentionBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = SymbolicAttentionBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(model_dim, target_dim)
        self.public_architecture = "Set Transformer"
        self.support_order_invariant = True

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        support_mask = batch["support_mask"].bool()
        support = self.embed.pairs(batch)
        for block in self.support_blocks:
            support = block(
                support,
                support,
                memory_padding_mask=~support_mask,
            )
            support = support * support_mask.unsqueeze(-1)
        query = self.embed.query(batch).unsqueeze(1)
        pooled = self.pool(
            query,
            support,
            memory_padding_mask=~support_mask,
        )
        return self.head(pooled[:, 0])


class SymbolicHopfieldClassifier(nn.Module):
    """Modern Hopfield-style associative retrieval over support pairs."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(
            num_symbols=num_symbols,
            model_dim=model_dim,
            max_support_k=max_support_k,
            use_support_position=False,
        )
        self.retrieval_blocks = nn.ModuleList(
            [
                SymbolicAttentionBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(model_dim, target_dim)
        self.public_architecture = "Modern Hopfield-style memory"
        self.support_order_invariant = True

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        state = self.embed.query(batch).unsqueeze(1)
        memory = self.embed.pairs(batch)
        memory_padding_mask = ~batch["support_mask"].bool()
        for block in self.retrieval_blocks:
            state = block(
                state,
                memory,
                memory_padding_mask=memory_padding_mask,
            )
        return self.head(state[:, 0])


class SymbolicPerceiverClassifier(nn.Module):
    """Perceiver-style latent bottleneck over query and support tokens."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        latent_count: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if latent_count <= 0:
            raise ValueError("latent_count must be positive.")
        self.embed = SymbolicPairEmbedding(
            num_symbols=num_symbols,
            model_dim=model_dim,
            max_support_k=max_support_k,
            use_support_position=False,
        )
        self.query_role = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.latents = nn.Parameter(torch.zeros(1, latent_count, model_dim))
        self.cross_blocks = nn.ModuleList(
            [
                SymbolicAttentionBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                SymbolicAttentionBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, target_dim)
        self.public_architecture = "Perceiver-style"
        self.support_order_invariant = True
        self.population_size = latent_count
        nn.init.trunc_normal_(self.query_role, std=0.02)
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch["query_digit"].shape[0]
        query = self.embed.query(batch).unsqueeze(1) + self.query_role
        support = self.embed.pairs(batch)
        inputs = torch.cat([query, support], dim=1)
        input_padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=query.device,
                ),
                ~batch["support_mask"].bool(),
            ],
            dim=1,
        )
        latents = self.latents.expand(batch_size, -1, -1)
        for cross_block, self_block in zip(
            self.cross_blocks,
            self.self_blocks,
        ):
            latents = cross_block(
                latents,
                inputs,
                memory_padding_mask=input_padding_mask,
            )
            latents = self_block(latents, latents)
        return self.head(self.norm(latents.mean(dim=1)))


class DigitFourierRuleFeatures(nn.Module):
    """Fixed symbolic rule features for modular digit transformations."""

    VALID_ABLATIONS = {"none", "fourier", "parity", "block", "scaled", "delta"}

    def __init__(
        self,
        num_symbols: int = 10,
        fourier_bands: int = 4,
        feature_ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_symbols < 2:
            raise ValueError("num_symbols must be at least 2.")
        if fourier_bands <= 0:
            raise ValueError("fourier_bands must be positive.")
        if feature_ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown rule feature ablation: {feature_ablation}.")
        self.num_symbols = num_symbols
        self.fourier_bands = fourier_bands
        self.feature_ablation = feature_ablation
        self.digit_dim = 2 * fourier_bands + 3
        self.pair_dim = 3 * self.digit_dim + 2

    def digit_features(self, values: Tensor) -> Tensor:
        """Encode digits with modular Fourier, parity, block, and scaled-value features."""

        values = values.float()
        features: list[Tensor] = []
        for band in range(1, self.fourier_bands + 1):
            angle = 2.0 * math.pi * float(band) * values / float(self.num_symbols)
            if self.feature_ablation == "fourier":
                features.extend([torch.zeros_like(values), torch.zeros_like(values)])
            else:
                features.extend([torch.sin(angle), torch.cos(angle)])
        parity = ((values.long() % 2).float() * 2.0) - 1.0
        block = ((values.long() >= (self.num_symbols // 2)).float() * 2.0) - 1.0
        scaled = (2.0 * values / float(self.num_symbols - 1)) - 1.0
        if self.feature_ablation == "parity":
            parity = torch.zeros_like(parity)
        if self.feature_ablation == "block":
            block = torch.zeros_like(block)
        if self.feature_ablation == "scaled":
            scaled = torch.zeros_like(scaled)
        features.extend([parity, block, scaled])
        return torch.stack(features, dim=-1)

    def pair_features(self, digits: Tensor, labels: Tensor) -> Tensor:
        """Encode support pairs with source, target, and modular-delta structure."""

        delta = (labels.long() - digits.long()).remainder(self.num_symbols)
        same_parity = (digits.long() % 2).eq(labels.long() % 2).float() * 2.0 - 1.0
        same_block = (digits.long() >= (self.num_symbols // 2)).eq(
            labels.long() >= (self.num_symbols // 2)
        ).float() * 2.0 - 1.0
        digit_features = self.digit_features(digits)
        label_features = self.digit_features(labels)
        delta_features = self.digit_features(delta)
        if self.feature_ablation == "delta":
            delta_features = torch.zeros_like(delta_features)
        if self.feature_ablation == "parity":
            same_parity = torch.zeros_like(same_parity)
        if self.feature_ablation == "block":
            same_block = torch.zeros_like(same_block)
        return torch.cat(
            [
                digit_features,
                label_features,
                delta_features,
                same_parity.unsqueeze(-1),
                same_block.unsqueeze(-1),
            ],
            dim=-1,
        )

    def token_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Return [CLS, query, support-pair] rule features padded to pair_dim."""

        batch_size = batch["query_digit"].shape[0]
        support_digits = batch["support_digits"].long()
        query_features = support_digits.new_zeros(batch_size, self.pair_dim).float()
        query_digit_features = self.digit_features(batch["query_digit"].long())
        query_features[:, : self.digit_dim] = query_digit_features
        pair_features = self.pair_features(support_digits, batch["support_labels"].long())
        cls_features = query_features.new_zeros(batch_size, 1, self.pair_dim)
        return torch.cat([cls_features, query_features.unsqueeze(1), pair_features], dim=1)


class SymbolicRuleFeatureTransformerClassifier(SymbolicTransformerClassifier):
    """A7 Transformer with fixed symbolic rule features injected into tokens."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        feature_ablation: str = "none",
    ) -> None:
        super().__init__(
            num_symbols=num_symbols,
            max_support_k=max_support_k,
            target_dim=target_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.rule_features = DigitFourierRuleFeatures(
            num_symbols=num_symbols,
            fourier_bands=rule_fourier_bands,
            feature_ablation=feature_ablation,
        )
        self.rule_token_proj = nn.Linear(self.rule_features.pair_dim, model_dim)
        self.rule_feature_ablation = feature_ablation
        self.attribution_id = (
            "A7" if feature_ablation == "none" else f"A7-no-{feature_ablation}"
        )
        self.readout_type = "transformer_cls"
        self.uses_explicit_rule_features = True

    def initial_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = super().initial_tokens(batch)
        rule_features = self.rule_features.token_features(batch).to(tokens.device)
        return tokens + self.rule_token_proj(rule_features)


class SymbolicParameterMatchedTransformerClassifier(SymbolicTransformerClassifier):
    """A8 widened Transformer parameter control without extra feature branches."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 92,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 192,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            num_symbols=num_symbols,
            max_support_k=max_support_k,
            target_dim=target_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.attribution_id = "A8"
        self.readout_type = "parameter_matched_transformer_cls"


class LatentPopulationEncoder(nn.Module):
    """Build a query-conditioned latent population from Transformer support states."""

    def __init__(
        self,
        model_dim: int,
        population_size: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if population_size <= 0:
            raise ValueError("population_size must be positive.")
        self.population_size = population_size
        self.particle_base = nn.Parameter(torch.empty(1, population_size, model_dim))
        self.context_condition = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.particle_query = nn.Linear(model_dim, model_dim)
        self.support_key = nn.Linear(model_dim, model_dim)
        self.support_value = nn.Linear(model_dim, model_dim)
        self.particle_norm = nn.LayerNorm(model_dim)
        nn.init.trunc_normal_(self.particle_base, std=0.02)

    def forward(
        self,
        cls_state: Tensor,
        query_state: Tensor,
        support_states: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        batch_size = cls_state.shape[0]
        context = self.context_condition(torch.cat([cls_state, query_state], dim=-1))
        particles = self.particle_base.expand(batch_size, -1, -1) + context.unsqueeze(1)
        scores = torch.matmul(
            self.particle_query(particles),
            self.support_key(support_states).transpose(1, 2),
        )
        scores = scores / math.sqrt(float(support_states.shape[-1]))
        scores = scores.masked_fill(~support_mask.bool().unsqueeze(1), -1.0e9)
        support_summary = torch.matmul(torch.softmax(scores, dim=-1), self.support_value(support_states))
        return self.particle_norm(particles + support_summary)


class GenericLatentAttentionReadout(nn.Module):
    """A2 ordinary learned-query attention pooling over a latent population."""

    def __init__(
        self,
        model_dim: int,
        query_count: int = 4,
        num_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if query_count <= 0:
            raise ValueError("query_count must be positive.")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.model_dim = model_dim
        self.query_count = query_count
        self.pool_queries = nn.Parameter(torch.empty(1, query_count, model_dim))
        self.query_condition = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: Tensor | None = None
        nn.init.trunc_normal_(self.pool_queries, std=0.02)

    @property
    def output_dim_multiplier(self) -> int:
        return 2

    def forward(self, particles: Tensor, cls_state: Tensor, query_state: Tensor) -> Tensor:
        batch_size = particles.shape[0]
        conditioned = self.query_condition(torch.cat([cls_state, query_state], dim=-1))
        queries = self.pool_queries.expand(batch_size, -1, -1)
        queries = queries + conditioned.unsqueeze(1)
        pooled, attention = self.attention(
            query=queries,
            key=particles,
            value=particles,
            need_weights=True,
        )
        self.last_attention = attention
        pooled = self.dropout(self.norm(pooled))
        return torch.cat([pooled.mean(dim=1), pooled.max(dim=1).values], dim=-1)


class HarmonicPhaseClusterReadout(nn.Module):
    """A4 harmonic order and soft phase-cluster readout without phase dynamics."""

    def __init__(
        self,
        model_dim: int,
        population_size: int,
        harmonics: int = 4,
        cluster_count: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if harmonics <= 0:
            raise ValueError("harmonics must be positive.")
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive.")
        self.harmonics = harmonics
        self.cluster_count = cluster_count
        phase_base = torch.arange(population_size).float() * (2.0 * math.pi / float(population_size))
        cluster_anchors = torch.arange(cluster_count).float() * (2.0 * math.pi / float(cluster_count))
        self.phase_base = nn.Parameter(phase_base.unsqueeze(0))
        self.phase_init = nn.Linear(model_dim, 1)
        self.register_buffer("cluster_anchors", cluster_anchors, persistent=False)
        self.cluster_log_temp = nn.Parameter(torch.tensor(math.log(4.0)))
        self.order_encoder = nn.Sequential(
            nn.Linear(3 * harmonics, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.cluster_encoder = nn.Sequential(
            nn.Linear(cluster_count * model_dim + cluster_count, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.last_theta: Tensor | None = None
        self.last_order_parameters: Tensor | None = None
        self.last_cluster_mass: Tensor | None = None
        nn.init.zeros_(self.phase_init.weight)
        nn.init.zeros_(self.phase_init.bias)

    @property
    def output_dim_multiplier(self) -> int:
        return 2

    @staticmethod
    def wrap_phase(theta: Tensor) -> Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def forward(self, particles: Tensor, cls_state: Tensor, query_state: Tensor) -> Tensor:
        del cls_state, query_state
        batch_size = particles.shape[0]
        theta = self.phase_base.expand(batch_size, -1)
        theta = self.wrap_phase(theta + math.pi * torch.tanh(self.phase_init(particles).squeeze(-1)))
        self.last_theta = theta

        harmonics = torch.arange(1, self.harmonics + 1, dtype=theta.dtype, device=theta.device)
        angles = theta.unsqueeze(-1) * harmonics
        real = torch.cos(angles).mean(dim=1)
        imag = torch.sin(angles).mean(dim=1)
        radius = torch.sqrt(real.square() + imag.square() + 1.0e-8)
        order_parameters = torch.cat([real, imag, radius], dim=-1)
        self.last_order_parameters = order_parameters
        order_state = self.order_encoder(order_parameters)

        cluster_logits = self.cluster_log_temp.exp() * torch.cos(
            theta.unsqueeze(-1) - self.cluster_anchors.view(1, 1, -1)
        )
        cluster_mass = torch.softmax(cluster_logits, dim=-1).mean(dim=1)
        cluster_weights = torch.softmax(cluster_logits, dim=1)
        cluster_values = torch.einsum("bmc,bmd->bcd", cluster_weights, particles)
        cluster_input = torch.cat([cluster_values.flatten(1), cluster_mass], dim=-1)
        self.last_cluster_mass = cluster_mass
        cluster_state = self.cluster_encoder(cluster_input)
        return torch.cat([order_state, cluster_state], dim=-1)


class SymbolicTransformerPopulationClassifier(nn.Module):
    """Shared A2/A4 Transformer plus latent-population attribution model."""

    def __init__(
        self,
        readout_type: str,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.05,
        population_size: int = 32,
        harmonics: int = 4,
        cluster_count: int = 4,
        use_rule_features: bool = False,
        rule_fourier_bands: int = 4,
    ) -> None:
        super().__init__()
        if readout_type not in {"attention", "harmonic_phase_cluster"}:
            raise ValueError(f"Unsupported population readout: {readout_type}")
        if use_rule_features and readout_type != "harmonic_phase_cluster":
            raise ValueError("Explicit rule features are only defined for the A6 harmonic readout.")
        transformer_kwargs = {
            "num_symbols": num_symbols,
            "max_support_k": max_support_k,
            "target_dim": target_dim,
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "dropout": dropout,
        }
        if use_rule_features:
            self.transformer = SymbolicRuleFeatureTransformerClassifier(
                **transformer_kwargs,
                rule_fourier_bands=rule_fourier_bands,
            )
        else:
            self.transformer = SymbolicTransformerClassifier(**transformer_kwargs)
        for param in self.transformer.head.parameters():
            param.requires_grad = False
        self.population = LatentPopulationEncoder(
            model_dim=model_dim,
            population_size=population_size,
            dropout=dropout,
        )
        if readout_type == "attention":
            self.readout: nn.Module = GenericLatentAttentionReadout(
                model_dim=model_dim,
                query_count=cluster_count,
                num_heads=num_heads,
                dropout=dropout,
            )
            attribution_id = "A2"
        else:
            self.readout = HarmonicPhaseClusterReadout(
                model_dim=model_dim,
                population_size=population_size,
                harmonics=harmonics,
                cluster_count=cluster_count,
                dropout=dropout,
            )
            attribution_id = "A6" if use_rule_features else "A4"
        side_dim = self.readout.output_dim_multiplier * model_dim
        self.side_norm = nn.LayerNorm(side_dim)
        self.fusion_norm = nn.LayerNorm(model_dim + side_dim)
        self.branch_gate = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Sequential(
            nn.Linear(model_dim + side_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, target_dim),
        )
        self.attribution_id = attribution_id
        self.readout_type = readout_type
        self.uses_explicit_rule_features = use_rule_features
        self.uses_kuramoto = False
        self.effective_model_dim = model_dim
        self.effective_ff_dim = ff_dim
        self.population_size = population_size
        self.effective_lop_steps = 0
        self.last_population: Tensor | None = None
        self.last_population_features: Tensor | None = None

    def transformer_states(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.transformer.encode_tokens(batch)
        normalized = self.transformer.norm(tokens)
        return normalized[:, 0], normalized[:, 1], normalized[:, 2:]

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        cls_state, query_state, support_states = self.transformer_states(batch)
        particles = self.population(
            cls_state=cls_state,
            query_state=query_state,
            support_states=support_states,
            support_mask=batch["support_mask"],
        )
        self.last_population = particles
        side = self.readout(particles, cls_state, query_state)
        side = self.side_norm(side)
        self.last_population_features = side
        gated_side = torch.sigmoid(self.branch_gate) * side
        fused = self.fusion_norm(torch.cat([cls_state, gated_side], dim=-1))
        return self.head(fused)


class RuleCodePool(nn.Module):
    """Pool support evidence into a global rule code using oscillator synchrony."""

    def __init__(self, model_dim: int, oscillator_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        score_dim = 3 * model_dim + oscillator_dim + 1
        value_dim = 2 * model_dim + oscillator_dim
        self.score = nn.Sequential(
            nn.Linear(score_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(value_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(
        self,
        tokens: Tensor,
        mu: Tensor,
        rule_embeddings: Tensor,
        synchrony: Tensor,
        support_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return rule code, support scores, and support values."""

        support_tokens = tokens[:, 2:]
        support_mu = mu[:, 2:]
        support_rule = rule_embeddings[:, 2:]
        query_tokens = tokens[:, 1:2].expand(-1, support_tokens.shape[1], -1)
        query_support_sync = synchrony[:, 1, 2:].unsqueeze(-1)
        score_input = torch.cat(
            [query_tokens, support_tokens, support_rule, support_mu, query_support_sync],
            dim=-1,
        )
        scores = self.score(score_input).squeeze(-1)
        scores = scores.masked_fill(~support_mask.bool(), -1.0e9)
        weights = torch.softmax(scores, dim=-1)
        value_input = torch.cat([support_tokens, support_rule, support_mu], dim=-1)
        values = self.value(value_input)
        rule_code = (weights.unsqueeze(-1) * values).sum(dim=1)
        return rule_code, scores, values


class SymbolicLatentOscillatorPopulationClassifier(nn.Module):
    """Latent oscillator population model for symbolic rule induction."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        ff_dim: int = 128,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        oscillator_count: int = 64,
        oscillator_steps: int = 8,
        oscillator_dt: float = 0.1,
        harmonics: int = 4,
        cluster_count: int = 4,
        coupling_dim: int = 32,
        use_kuramoto: bool = True,
        use_cluster_readout: bool = True,
        use_higher_harmonics: bool = True,
        use_rule_fourier: bool = True,
    ) -> None:
        super().__init__()
        if oscillator_count <= 0:
            raise ValueError("oscillator_count must be positive.")
        if oscillator_steps < 0:
            raise ValueError("oscillator_steps must be non-negative.")
        if harmonics <= 0:
            raise ValueError("harmonics must be positive.")
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive.")
        self.embed = SymbolicPairEmbedding(num_symbols=num_symbols, model_dim=model_dim, max_support_k=max_support_k)
        self.rule_features = DigitFourierRuleFeatures(num_symbols=num_symbols, fourier_bands=rule_fourier_bands)
        self.oscillator_count = oscillator_count
        self.oscillator_steps = oscillator_steps
        self.oscillator_dt = oscillator_dt
        self.harmonics = 1 if not use_higher_harmonics else harmonics
        self.cluster_count = cluster_count
        self.use_kuramoto = use_kuramoto
        self.use_cluster_readout = use_cluster_readout
        self.use_rule_fourier = use_rule_fourier

        self.query_rule_proj = nn.Linear(self.rule_features.pair_dim, model_dim)
        self.pair_rule_proj = nn.Linear(self.rule_features.pair_dim, model_dim)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.support_encoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.particle_base = nn.Parameter(torch.empty(1, oscillator_count, model_dim))
        self.particle_condition = nn.Linear(model_dim, model_dim)
        self.particle_query = nn.Linear(model_dim, model_dim)
        self.support_key = nn.Linear(model_dim, model_dim)
        self.support_value = nn.Linear(model_dim, model_dim)
        self.particle_norm = nn.LayerNorm(model_dim)

        phase_base = torch.arange(oscillator_count).float() * (2.0 * math.pi / float(oscillator_count))
        self.phase_base = nn.Parameter(phase_base.unsqueeze(0))
        self.phase_init = nn.Linear(model_dim, 1)
        self.omega_head = nn.Linear(model_dim, 1)
        self.drive_amp = nn.Linear(model_dim, 1)
        self.drive_phase = nn.Linear(model_dim, 1)
        self.coupling_q = nn.Linear(model_dim, coupling_dim)
        self.coupling_k = nn.Linear(model_dim, coupling_dim)
        self.coupling_scale = nn.Parameter(torch.tensor(-1.0))

        order_dim = 3 * self.harmonics
        self.order_encoder = nn.Sequential(
            nn.Linear(order_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        if use_cluster_readout:
            anchors = torch.arange(cluster_count).float() * (2.0 * math.pi / float(cluster_count))
            self.register_buffer("cluster_anchors", anchors, persistent=False)
            self.cluster_log_temp = nn.Parameter(torch.tensor(math.log(4.0)))
            self.cluster_encoder = nn.Sequential(
                nn.Linear(cluster_count * model_dim + cluster_count, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
                nn.LayerNorm(model_dim),
            )
            head_input_dim = 4 * model_dim
        else:
            self.register_buffer("cluster_anchors", torch.empty(0), persistent=False)
            self.cluster_log_temp = nn.Parameter(torch.tensor(math.log(4.0)), requires_grad=False)
            self.cluster_encoder = None
            head_input_dim = 3 * model_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, target_dim),
        )
        self.last_theta: Tensor | None = None
        self.last_order_parameters: Tensor | None = None
        nn.init.trunc_normal_(self.particle_base, std=0.02)
        nn.init.zeros_(self.phase_init.weight)
        nn.init.zeros_(self.phase_init.bias)
        nn.init.zeros_(self.omega_head.weight)
        nn.init.zeros_(self.omega_head.bias)
        nn.init.constant_(self.drive_amp.bias, -1.0)
        self._freeze_disabled_branches()

    def _freeze_disabled_branches(self) -> None:
        """Exclude disabled ablation branches from trainable parameter counts."""

        if not self.use_rule_fourier:
            for module in (self.query_rule_proj, self.pair_rule_proj):
                for param in module.parameters():
                    param.requires_grad = False
        if not self.use_kuramoto:
            for module in (self.omega_head, self.drive_amp, self.drive_phase, self.coupling_q, self.coupling_k):
                for param in module.parameters():
                    param.requires_grad = False
            self.coupling_scale.requires_grad = False

    @staticmethod
    def wrap_phase(theta: Tensor) -> Tensor:
        """Wrap phases to [-pi, pi] without breaking gradients."""

        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def rule_token_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Return fixed symbolic rule features, or zeros for the ablation."""

        if not self.use_rule_fourier:
            batch_size = batch["query_digit"].shape[0]
            support_count = batch["support_digits"].shape[1]
            return self.phase_base.new_zeros(batch_size, support_count + 2, self.rule_features.pair_dim)
        return self.rule_features.token_features(batch).to(batch["query_digit"].device)

    def encode_support_query(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Encode query, support pairs, and a global support summary."""

        rule_features = self.rule_token_features(batch)
        query = self.embed.query(batch)
        pairs = self.embed.pairs(batch)
        if self.use_rule_fourier:
            query = query + self.query_rule_proj(rule_features[:, 1])
            pairs = pairs + self.pair_rule_proj(rule_features[:, 2:])
        query = self.query_encoder(query)
        support = self.support_encoder(pairs)
        support_mask = batch["support_mask"].float().unsqueeze(-1)
        support = support * support_mask
        pooled = support.sum(dim=1) / support_mask.sum(dim=1).clamp_min(1.0)
        global_state = self.global_encoder(torch.cat([query, pooled], dim=-1))
        return query, support, global_state

    def initialize_particles(
        self,
        global_state: Tensor,
        support: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        """Condition the latent oscillator bank on support evidence."""

        batch_size = global_state.shape[0]
        particles = self.particle_base.expand(batch_size, -1, -1)
        particles = particles + self.particle_condition(global_state).unsqueeze(1)
        scores = torch.matmul(self.particle_query(particles), self.support_key(support).transpose(1, 2))
        scores = scores / math.sqrt(float(support.shape[-1]))
        scores = scores.masked_fill(~support_mask.bool().unsqueeze(1), -1.0e9)
        weights = torch.softmax(scores, dim=-1)
        support_summary = torch.matmul(weights, self.support_value(support))
        return self.particle_norm(particles + support_summary)

    def evolve_phases(self, particles: Tensor) -> Tensor:
        """Run a support-conditioned Kuramoto population update."""

        batch_size = particles.shape[0]
        theta = self.phase_base.expand(batch_size, -1) + math.pi * torch.tanh(self.phase_init(particles).squeeze(-1))
        theta = self.wrap_phase(theta)
        if not self.use_kuramoto or self.oscillator_steps <= 0:
            return theta

        omega = 0.5 * torch.tanh(self.omega_head(particles).squeeze(-1))
        drive_amp = F.softplus(self.drive_amp(particles).squeeze(-1))
        drive_phase = math.pi * torch.tanh(self.drive_phase(particles).squeeze(-1))
        coupling_q = self.coupling_q(particles)
        coupling_k = self.coupling_k(particles)
        coupling_logits = torch.matmul(coupling_q, coupling_k.transpose(1, 2)) / math.sqrt(float(coupling_q.shape[-1]))
        coupling = torch.tanh(coupling_logits) * F.softplus(self.coupling_scale)
        eye = torch.eye(self.oscillator_count, dtype=torch.bool, device=particles.device).unsqueeze(0)
        coupling = coupling.masked_fill(eye, 0.0)
        normalizer = float(max(self.oscillator_count - 1, 1))
        for _ in range(self.oscillator_steps):
            phase_diff = theta.unsqueeze(1) - theta.unsqueeze(2)
            coupled = (coupling * torch.sin(phase_diff)).sum(dim=-1) / normalizer
            drive = drive_amp * torch.sin(drive_phase - theta)
            theta = self.wrap_phase(theta + self.oscillator_dt * (omega + coupled + drive))
        return theta

    def order_features(self, theta: Tensor) -> Tensor:
        """Read global Kuramoto order parameters for several harmonics."""

        harmonics = torch.arange(1, self.harmonics + 1, dtype=theta.dtype, device=theta.device)
        angles = theta.unsqueeze(-1) * harmonics
        real = torch.cos(angles).mean(dim=1)
        imag = torch.sin(angles).mean(dim=1)
        radius = torch.sqrt(real.square() + imag.square() + 1.0e-8)
        features = torch.cat([real, imag, radius], dim=-1)
        self.last_order_parameters = features
        return self.order_encoder(features)

    def cluster_features(self, theta: Tensor, particles: Tensor) -> Tensor:
        """Pool oscillator values by soft phase clusters."""

        if self.cluster_encoder is None:
            raise RuntimeError("cluster_features called when cluster readout is disabled.")
        cluster_temp = self.cluster_log_temp.exp()
        cluster_logits = cluster_temp * torch.cos(theta.unsqueeze(-1) - self.cluster_anchors.view(1, 1, -1))
        cluster_mass = torch.softmax(cluster_logits, dim=-1).mean(dim=1)
        cluster_weights = torch.softmax(cluster_logits, dim=1)
        cluster_values = torch.einsum("bmc,bmd->bcd", cluster_weights, particles)
        cluster_input = torch.cat([cluster_values.flatten(1), cluster_mass], dim=-1)
        return self.cluster_encoder(cluster_input)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        query, support, global_state = self.encode_support_query(batch)
        particles = self.initialize_particles(global_state, support, batch["support_mask"].bool())
        theta = self.evolve_phases(particles)
        self.last_theta = theta
        order_state = self.order_features(theta)
        readout_parts = [query, global_state, order_state]
        if self.use_cluster_readout:
            readout_parts.append(self.cluster_features(theta, particles))
        return self.head(torch.cat(readout_parts, dim=-1))


class SynchronyStructureReadout(nn.Module):
    """Read candidate predictions from query-conditioned synchrony communities."""

    def __init__(
        self,
        model_dim: int,
        target_dim: int,
        harmonics: int = 4,
        community_count: int = 4,
        dropout: float = 0.05,
        use_query_conditioned_assignment: bool = True,
        use_support_consistency: bool = True,
        support_aux_weight: float = 0.1,
        partial_sync_weight: float = 0.01,
        usage_entropy_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if harmonics <= 0:
            raise ValueError("harmonics must be positive.")
        if community_count <= 0:
            raise ValueError("community_count must be positive.")
        self.model_dim = model_dim
        self.target_dim = target_dim
        self.harmonics = harmonics
        self.community_count = community_count
        self.use_query_conditioned_assignment = use_query_conditioned_assignment
        self.use_support_consistency = use_support_consistency
        self.support_aux_weight = support_aux_weight
        self.partial_sync_weight = partial_sync_weight
        self.usage_entropy_weight = usage_entropy_weight

        anchors = torch.arange(community_count).float() * (2.0 * math.pi / float(community_count))
        self.register_buffer("community_anchors", anchors, persistent=False)
        self.phase_log_temp = nn.Parameter(torch.tensor(math.log(4.0)))
        self.assignment_value = nn.Linear(model_dim, community_count)
        self.assignment_query = nn.Linear(model_dim, community_count * model_dim, bias=False)

        stats_dim = 3 * harmonics + 2
        candidate_dim = (2 * model_dim) + stats_dim
        self.candidate_head = nn.Sequential(
            nn.Linear(candidate_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, target_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(candidate_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.support_score_scale = nn.Parameter(torch.tensor(1.0))
        self.last_community_mass: Tensor | None = None
        self.last_community_radius: Tensor | None = None
        self.last_gate_probs: Tensor | None = None

    @staticmethod
    def normalized_entropy(probs: Tensor, dim: int) -> Tensor:
        """Return entropy normalized to [0, 1] up to numerical precision."""

        count = probs.shape[dim]
        denom = math.log(float(max(count, 2)))
        return -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=dim) / denom

    def community_assignment(self, theta: Tensor, particles: Tensor, query: Tensor) -> Tensor:
        """Assign oscillators to soft synchrony communities."""

        phase_temp = self.phase_log_temp.exp()
        phase_logits = phase_temp * torch.cos(theta.unsqueeze(-1) - self.community_anchors.view(1, 1, -1))
        if not self.use_query_conditioned_assignment:
            return torch.softmax(phase_logits, dim=-1)

        value_logits = self.assignment_value(particles)
        query_weights = self.assignment_query(query).view(
            query.shape[0],
            self.community_count,
            self.model_dim,
        )
        query_logits = torch.einsum("bcd,bmd->bmc", query_weights, particles)
        query_logits = query_logits / math.sqrt(float(self.model_dim))
        return torch.softmax(phase_logits + value_logits + query_logits, dim=-1)

    def community_features(
        self,
        theta: Tensor,
        particles: Tensor,
        query: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return per-community feature vectors and synchrony diagnostics."""

        gamma = self.community_assignment(theta, particles, query)
        mass = gamma.mean(dim=1)
        weights = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        values = torch.einsum("bmc,bmd->bcd", weights, particles)

        harmonics = torch.arange(1, self.harmonics + 1, dtype=theta.dtype, device=theta.device)
        angles = theta.unsqueeze(-1) * harmonics
        cos_terms = torch.cos(angles)
        sin_terms = torch.sin(angles)
        real = torch.einsum("bmc,bml->bcl", weights, cos_terms)
        imag = torch.einsum("bmc,bml->bcl", weights, sin_terms)
        radius = torch.sqrt(real.square() + imag.square() + 1.0e-8)

        cluster_entropy = self.normalized_entropy(weights.transpose(1, 2), dim=-1)
        query_expanded = query.unsqueeze(1).expand(-1, self.community_count, -1)
        features = torch.cat(
            [
                query_expanded,
                values,
                real,
                imag,
                radius,
                mass.unsqueeze(-1),
                cluster_entropy.unsqueeze(-1),
            ],
            dim=-1,
        )
        self.last_community_mass = mass.detach()
        self.last_community_radius = radius.detach()
        return features, values, radius, mass, gamma

    def candidate_logits_for_queries(self, queries: Tensor, features: Tensor) -> Tensor:
        """Apply the shared candidate head to query/community feature pairs."""

        batch_size, community_count, _ = features.shape
        query_expanded = queries.unsqueeze(1).expand(-1, community_count, -1)
        candidate_features = torch.cat([query_expanded, features[:, :, self.model_dim :]], dim=-1)
        return self.candidate_head(candidate_features.reshape(batch_size * community_count, -1)).view(
            batch_size,
            community_count,
            self.target_dim,
        )

    def support_candidate_logits(self, support_queries: Tensor, features: Tensor) -> Tensor:
        """Predict support labels from each synchrony community."""

        batch_size, support_count, _ = support_queries.shape
        community_count = features.shape[1]
        support_expanded = support_queries.unsqueeze(2).expand(-1, -1, community_count, -1)
        community_tail = features[:, :, self.model_dim :].unsqueeze(1).expand(-1, support_count, -1, -1)
        candidate_features = torch.cat([support_expanded, community_tail], dim=-1)
        return self.candidate_head(candidate_features.reshape(batch_size * support_count * community_count, -1)).view(
            batch_size,
            support_count,
            community_count,
            self.target_dim,
        )

    def support_scores(
        self,
        support_logits: Tensor,
        support_labels: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        """Score each community by how well it explains support labels."""

        log_probs = F.log_softmax(support_logits, dim=-1)
        labels = support_labels.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, support_logits.shape[2], 1)
        scores = log_probs.gather(dim=-1, index=labels).squeeze(-1)
        mask = support_mask.float().unsqueeze(-1)
        return (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def regularization_loss(self, theta: Tensor, radius: Tensor, mass: Tensor) -> Tensor | None:
        """Encourage partial synchronization without global phase collapse."""

        losses: list[Tensor] = []
        if self.partial_sync_weight > 0.0:
            global_real = torch.cos(theta).mean(dim=1)
            global_imag = torch.sin(theta).mean(dim=1)
            global_r1 = torch.sqrt(global_real.square() + global_imag.square() + 1.0e-8)
            max_cluster_r1 = radius[:, :, 0].max(dim=1).values
            partial = F.relu(global_r1 - 0.95).square() + F.relu(0.25 - max_cluster_r1).square()
            losses.append(self.partial_sync_weight * partial.mean())
        if self.usage_entropy_weight > 0.0:
            usage_entropy = self.normalized_entropy(mass / mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-6), dim=-1)
            losses.append(self.usage_entropy_weight * F.relu(0.5 - usage_entropy).square().mean())
        if not losses:
            return None
        return torch.stack(losses).sum()

    def forward(
        self,
        theta: Tensor,
        particles: Tensor,
        query: Tensor,
        support_queries: Tensor | None = None,
        support_labels: Tensor | None = None,
        support_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        features, _, radius, mass, _ = self.community_features(theta, particles, query)
        candidate_logits = self.candidate_logits_for_queries(query, features)
        gate_logits = self.gate_head(features.reshape(features.shape[0] * features.shape[1], -1)).view(
            features.shape[0],
            features.shape[1],
        )

        support_logits = None
        if (
            self.use_support_consistency
            and support_queries is not None
            and support_labels is not None
            and support_mask is not None
        ):
            support_logits = self.support_candidate_logits(support_queries, features)
            gate_logits = gate_logits + self.support_score_scale * self.support_scores(
                support_logits,
                support_labels,
                support_mask,
            )

        log_pi = F.log_softmax(gate_logits, dim=-1)
        self.last_gate_probs = log_pi.detach().exp()
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
        log_probs = torch.logsumexp(log_pi.unsqueeze(-1) + candidate_log_probs, dim=1)

        aux_losses: list[Tensor] = []
        if support_logits is not None and self.support_aux_weight > 0.0:
            support_log_probs = F.log_softmax(support_logits, dim=-1)
            mixed_support = torch.logsumexp(log_pi[:, None, :, None] + support_log_probs, dim=2)
            support_loss = F.nll_loss(
                mixed_support.reshape(-1, self.target_dim),
                support_labels.long().reshape(-1),
                reduction="none",
            ).view_as(support_mask.float())
            support_loss = (support_loss * support_mask.float()).sum() / support_mask.float().sum().clamp_min(1.0)
            aux_losses.append(self.support_aux_weight * support_loss)

        regularization = self.regularization_loss(theta, radius, mass)
        if regularization is not None:
            aux_losses.append(regularization)
        aux_loss = torch.stack(aux_losses).sum() if aux_losses else None
        return log_probs, aux_loss


class SymbolicLOPSSRClassifier(SymbolicLatentOscillatorPopulationClassifier):
    """LOP-RI with synchrony-structure community readout."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        ff_dim: int = 128,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        oscillator_count: int = 64,
        oscillator_steps: int = 16,
        oscillator_dt: float = 0.1,
        harmonics: int = 4,
        community_count: int = 4,
        coupling_dim: int = 32,
        use_kuramoto: bool = True,
        use_query_conditioned_assignment: bool = True,
        use_support_consistency: bool = True,
        support_aux_weight: float = 0.1,
        partial_sync_weight: float = 0.01,
        usage_entropy_weight: float = 0.01,
    ) -> None:
        super().__init__(
            num_symbols=num_symbols,
            max_support_k=max_support_k,
            target_dim=target_dim,
            model_dim=model_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            rule_fourier_bands=rule_fourier_bands,
            oscillator_count=oscillator_count,
            oscillator_steps=oscillator_steps,
            oscillator_dt=oscillator_dt,
            harmonics=harmonics,
            cluster_count=community_count,
            coupling_dim=coupling_dim,
            use_kuramoto=use_kuramoto,
            use_cluster_readout=False,
            use_higher_harmonics=True,
            use_rule_fourier=True,
        )
        for module in (self.order_encoder, self.head):
            for param in module.parameters():
                param.requires_grad = False

        self.readout_context = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.support_digit_rule_proj = nn.Linear(self.rule_features.digit_dim, model_dim)
        self.support_query_encoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.ssr_readout = SynchronyStructureReadout(
            model_dim=model_dim,
            target_dim=target_dim,
            harmonics=harmonics,
            community_count=community_count,
            dropout=dropout,
            use_query_conditioned_assignment=use_query_conditioned_assignment,
            use_support_consistency=use_support_consistency,
            support_aux_weight=support_aux_weight,
            partial_sync_weight=partial_sync_weight,
            usage_entropy_weight=usage_entropy_weight,
        )

    def support_query_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode support source digits without leaking support labels."""

        digits = batch["support_digits"].long()
        support_queries = self.embed.source_digit(digits)
        digit_features = self.rule_features.digit_features(digits).to(support_queries.device)
        support_queries = support_queries + self.support_digit_rule_proj(digit_features)
        support_queries = support_queries * batch["support_mask"].float().unsqueeze(-1)
        return self.support_query_encoder(support_queries)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        query, support, global_state = self.encode_support_query(batch)
        particles = self.initialize_particles(global_state, support, batch["support_mask"].bool())
        theta = self.evolve_phases(particles)
        self.last_theta = theta
        query_context = self.readout_context(torch.cat([query, global_state], dim=-1))
        logits, aux_loss = self.ssr_readout(
            theta=theta,
            particles=particles,
            query=query_context,
            support_queries=self.support_query_features(batch),
            support_labels=batch["support_labels"],
            support_mask=batch["support_mask"],
        )
        return {"logits": logits, "aux_loss": aux_loss}


class SymbolicSynchronyRuleInductionClassifier(nn.Module):
    """Adapter-free synchrony rule induction model for symbolic mapping."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        oscillator_dim: int = 16,
        adapter_rank: int = 8,
        top_k: int = 8,
        steps: int = 2,
        dt: float = 0.2,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        use_rule_fourier: bool = True,
        use_sync_bias: bool = True,
        use_kuramoto: bool = True,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(num_symbols=num_symbols, model_dim=model_dim, max_support_k=max_support_k)
        self.rule_features = DigitFourierRuleFeatures(num_symbols=num_symbols, fourier_bands=rule_fourier_bands)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.query_role = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, max_support_k + 2, model_dim))
        self.rule_token_proj = nn.Linear(self.rule_features.pair_dim, model_dim)
        self.rule_mu_proj = nn.Linear(self.rule_features.pair_dim, oscillator_dim)
        self.mu_from_token = nn.Linear(model_dim, oscillator_dim)
        self.mu_position = nn.Parameter(torch.zeros(1, max_support_k + 2, oscillator_dim))
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
                    use_adapter=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, target_dim)
        self.use_rule_fourier = use_rule_fourier
        self.last_states: list[KFWAState] = []
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.query_role, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.mu_position, std=0.02)
        self._freeze_disabled_branches()

    def _freeze_disabled_branches(self) -> None:
        """Exclude disabled feature-injection branches from parameter counts."""

        if not self.use_rule_fourier:
            for module in (self.rule_token_proj, self.rule_mu_proj):
                for param in module.parameters():
                    param.requires_grad = False

    def token_mask(self, batch: dict[str, Tensor]) -> Tensor:
        """Return [B, T, 1] mask that keeps CLS/query and real support pairs."""

        batch_size = batch["query_digit"].shape[0]
        prefix = torch.ones(batch_size, 2, dtype=torch.bool, device=batch["query_digit"].device)
        mask = torch.cat([prefix, batch["support_mask"].bool()], dim=1)
        return mask.unsqueeze(-1).float()

    def rule_token_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Return fixed symbolic rule features for [CLS, query, support] tokens."""

        if not self.use_rule_fourier:
            batch_size = batch["query_digit"].shape[0]
            support_count = batch["support_digits"].shape[1]
            return self.cls.new_zeros(batch_size, support_count + 2, self.rule_features.pair_dim)
        return self.rule_features.token_features(batch).to(batch["query_digit"].device)

    def initial_tokens(self, batch: dict[str, Tensor], rule_features: Tensor, mask: Tensor) -> Tensor:
        """Build [CLS, query, support] tokens with optional rule-structure injection."""

        batch_size = batch["query_digit"].shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        query = self.embed.query(batch).unsqueeze(1) + self.query_role
        pairs = self.embed.pairs(batch)
        tokens = torch.cat([cls, query, pairs], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        if self.use_rule_fourier:
            tokens = tokens + self.rule_token_proj(rule_features)
        return tokens * mask

    def initial_mu(self, tokens: Tensor, rule_features: Tensor, mask: Tensor) -> Tensor:
        """Initialize oscillator states from token content and optional rule features."""

        mu = self.mu_from_token(tokens) + self.mu_position[:, : tokens.shape[1]]
        if self.use_rule_fourier:
            mu = mu + self.rule_mu_proj(rule_features)
        return normalize_state(mu) * mask

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        mask = self.token_mask(batch)
        rule_features = self.rule_token_features(batch)
        tokens = self.initial_tokens(batch, rule_features, mask)
        mu = self.initial_mu(tokens, rule_features, mask)
        states: list[KFWAState] = []
        for block in self.blocks:
            tokens, mu, state = block(tokens, mu)
            tokens = tokens * mask
            mu = normalize_state(mu) * mask
            states.append(state)
        self.last_states = states
        return self.head(self.norm(tokens[:, 0]))


class SymbolicSynchronyRuleAdapterClassifier(nn.Module):
    """Synchrony-gated rule-code adapter for symbolic rule inference."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        oscillator_dim: int = 16,
        adapter_rank: int = 8,
        top_k: int = 8,
        steps: int = 2,
        dt: float = 0.2,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        support_aux_weight: float = 0.1,
        adapter_gate_bias: float = -4.0,
        use_rule_fourier: bool = True,
        use_rule_adapter: bool = True,
        use_support_aux: bool = True,
        use_sync_bias: bool = True,
        use_kuramoto: bool = True,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(num_symbols=num_symbols, model_dim=model_dim, max_support_k=max_support_k)
        self.rule_features = DigitFourierRuleFeatures(num_symbols=num_symbols, fourier_bands=rule_fourier_bands)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.query_role = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, max_support_k + 2, model_dim))
        self.rule_token_proj = nn.Linear(self.rule_features.pair_dim, model_dim)
        self.rule_mu_proj = nn.Linear(self.rule_features.pair_dim, oscillator_dim)
        self.support_query_rule_proj = nn.Linear(self.rule_features.pair_dim, model_dim)
        self.mu_from_token = nn.Linear(model_dim, oscillator_dim)
        self.mu_position = nn.Parameter(torch.zeros(1, max_support_k + 2, oscillator_dim))
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
                    use_adapter=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.rule_pool = RuleCodePool(model_dim=model_dim, oscillator_dim=oscillator_dim, dropout=dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.adapter_coeff = nn.Linear(model_dim, adapter_rank)
        self.adapter_down = nn.Linear(model_dim, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, model_dim, bias=False)
        self.adapter_gate = nn.Linear(model_dim, 1)
        self.support_head = nn.Sequential(
            nn.Linear(2 * model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, target_dim),
        )
        self.head = nn.Linear(model_dim, target_dim)
        self.use_rule_fourier = use_rule_fourier
        self.use_rule_adapter = use_rule_adapter
        self.support_aux_weight = float(support_aux_weight if use_support_aux else 0.0)
        self.last_states: list[KFWAState] = []
        self.last_rule_code: Tensor | None = None
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.query_role, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.mu_position, std=0.02)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_gate.weight)
        nn.init.constant_(self.adapter_gate.bias, adapter_gate_bias)
        self._freeze_disabled_branches()

    def _freeze_disabled_branches(self) -> None:
        """Exclude disabled branches from trainable parameter counts."""

        if not self.use_rule_fourier:
            for module in (self.rule_token_proj, self.rule_mu_proj, self.support_query_rule_proj):
                for param in module.parameters():
                    param.requires_grad = False
        if not self.use_rule_adapter:
            for module in (self.adapter_coeff, self.adapter_down, self.adapter_up, self.adapter_gate):
                for param in module.parameters():
                    param.requires_grad = False
        if self.support_aux_weight <= 0.0:
            for param in self.support_head.parameters():
                param.requires_grad = False

    def token_mask(self, batch: dict[str, Tensor]) -> Tensor:
        """Return [B, T, 1] mask that keeps CLS/query and real support pairs."""

        batch_size = batch["query_digit"].shape[0]
        prefix = torch.ones(batch_size, 2, dtype=torch.bool, device=batch["query_digit"].device)
        mask = torch.cat([prefix, batch["support_mask"].bool()], dim=1)
        return mask.unsqueeze(-1).float()

    def rule_token_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Return fixed symbolic rule features, or zeros for the ablation."""

        features = self.rule_features.token_features(batch).to(batch["query_digit"].device)
        if not self.use_rule_fourier:
            return torch.zeros_like(features)
        return features

    def initial_tokens(self, batch: dict[str, Tensor], rule_features: Tensor) -> Tensor:
        """Build [CLS, query, support] tokens with optional rule-structure features."""

        batch_size = batch["query_digit"].shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        query = self.embed.query(batch).unsqueeze(1) + self.query_role
        pairs = self.embed.pairs(batch)
        tokens = torch.cat([cls, query, pairs], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        if self.use_rule_fourier:
            tokens = tokens + self.rule_token_proj(rule_features)
        return tokens * self.token_mask(batch)

    def initial_mu(self, tokens: Tensor, rule_features: Tensor, mask: Tensor) -> Tensor:
        """Initialize oscillator states from tokens and optional rule features."""

        mu = self.mu_from_token(tokens) + self.mu_position[:, : tokens.shape[1]]
        if self.use_rule_fourier:
            mu = mu + self.rule_mu_proj(rule_features)
        return normalize_state(mu) * mask

    def apply_rule_adapter(self, cls_state: Tensor, rule_code: Tensor) -> Tensor:
        """Apply a conservative gated low-rank readout adapter."""

        if not self.use_rule_adapter:
            return cls_state
        coeff = torch.tanh(self.adapter_coeff(rule_code))
        delta = self.adapter_up(coeff * self.adapter_down(cls_state))
        gate = torch.sigmoid(self.adapter_gate(rule_code))
        return cls_state + gate * delta

    def support_auxiliary_loss(
        self,
        batch: dict[str, Tensor],
        scores: Tensor,
        values: Tensor,
        rule_features: Tensor,
    ) -> Tensor | None:
        """Leave-one-support-out consistency loss for rule-code learning."""

        if self.support_aux_weight <= 0.0 or not self.training:
            return None
        support_mask = batch["support_mask"].bool()
        if not support_mask.any():
            return None
        support_count = support_mask.shape[1]
        device = scores.device
        exclude_self = ~torch.eye(support_count, dtype=torch.bool, device=device).unsqueeze(0)
        valid_sources = support_mask.unsqueeze(1).expand(-1, support_count, -1) & exclude_self
        masked_scores = scores.unsqueeze(1).expand(-1, support_count, -1).masked_fill(~valid_sources, -1.0e9)
        weights = torch.softmax(masked_scores, dim=-1)
        leave_one_rule = torch.matmul(weights, values)

        digit_tokens = self.embed.source_digit(batch["support_digits"].long())
        if self.use_rule_fourier:
            digit_tokens = digit_tokens + self.support_query_rule_proj(rule_features[:, 2:])
        aux_logits = self.support_head(torch.cat([digit_tokens, leave_one_rule], dim=-1))
        valid_targets = support_mask.reshape(-1)
        if not valid_targets.any():
            return None
        aux_loss = F.cross_entropy(
            aux_logits.reshape(-1, aux_logits.shape[-1])[valid_targets],
            batch["support_labels"].long().reshape(-1)[valid_targets],
        )
        return self.support_aux_weight * aux_loss

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        mask = self.token_mask(batch)
        rule_features = self.rule_token_features(batch)
        tokens = self.initial_tokens(batch, rule_features)
        mu = self.initial_mu(tokens, rule_features, mask)
        states: list[KFWAState] = []
        for block in self.blocks:
            tokens, mu, state = block(tokens, mu)
            tokens = tokens * mask
            mu = normalize_state(mu) * mask
            states.append(state)
        self.last_states = states
        synchrony = states[-1].synchrony if states else torch.matmul(mu, mu.transpose(1, 2))
        rule_embeddings = self.rule_token_proj(rule_features) if self.use_rule_fourier else tokens.new_zeros(tokens.shape)
        rule_code, scores, values = self.rule_pool(tokens, mu, rule_embeddings, synchrony, batch["support_mask"].bool())
        self.last_rule_code = rule_code
        cls_state = self.apply_rule_adapter(self.norm(tokens[:, 0]), rule_code)
        logits = self.head(cls_state)
        aux_loss = self.support_auxiliary_loss(batch, scores, values, rule_features)
        output: dict[str, Tensor] = {"logits": logits}
        if aux_loss is not None:
            output["aux_loss"] = aux_loss
        return output


class SymbolicKFWAClassifier(nn.Module):
    """KFWA token mixer for symbolic support-pair rule inference."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        oscillator_dim: int = 16,
        adapter_rank: int = 8,
        top_k: int = 8,
        steps: int = 2,
        dt: float = 0.2,
        dropout: float = 0.05,
        use_kuramoto: bool = True,
        use_sync_bias: bool = True,
        use_adapter: bool = True,
    ) -> None:
        super().__init__()
        self.embed = SymbolicPairEmbedding(num_symbols=num_symbols, model_dim=model_dim, max_support_k=max_support_k)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.query_role = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, max_support_k + 2, model_dim))
        self.mu_from_token = nn.Linear(model_dim, oscillator_dim)
        self.mu_position = nn.Parameter(torch.zeros(1, max_support_k + 2, oscillator_dim))
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
        nn.init.trunc_normal_(self.query_role, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.mu_position, std=0.02)

    def token_mask(self, batch: dict[str, Tensor]) -> Tensor:
        """Return [B, T, 1] mask that keeps CLS/query and real support pairs."""

        batch_size = batch["query_digit"].shape[0]
        prefix = torch.ones(batch_size, 2, dtype=torch.bool, device=batch["query_digit"].device)
        mask = torch.cat([prefix, batch["support_mask"].bool()], dim=1)
        return mask.unsqueeze(-1).float()

    def initial_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch["query_digit"].shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        query = self.embed.query(batch).unsqueeze(1) + self.query_role
        pairs = self.embed.pairs(batch)
        tokens = torch.cat([cls, query, pairs], dim=1)
        mask = self.token_mask(batch)
        return (tokens + self.position[:, : tokens.shape[1]]) * mask

    def initial_mu(self, tokens: Tensor, mask: Tensor) -> Tensor:
        mu = self.mu_from_token(tokens) + self.mu_position[:, : tokens.shape[1]]
        return normalize_state(mu) * mask

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        mask = self.token_mask(batch)
        tokens = self.initial_tokens(batch)
        mu = self.initial_mu(tokens, mask)
        states: list[KFWAState] = []
        for block in self.blocks:
            tokens, mu, state = block(tokens, mu)
            tokens = tokens * mask
            mu = normalize_state(mu) * mask
            states.append(state)
        self.last_states = states
        return self.head(self.norm(tokens[:, 0]))


class SymbolicKFWALOPFeatureClassifier(nn.Module):
    """KFWA backbone with latent oscillator population synchrony features."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        oscillator_dim: int = 16,
        adapter_rank: int = 8,
        top_k: int = 8,
        steps: int = 2,
        dt: float = 0.2,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        lop_oscillator_count: int = 64,
        lop_steps: int = 8,
        lop_dt: float = 0.1,
        lop_harmonics: int = 4,
        lop_cluster_count: int = 4,
        lop_coupling_dim: int = 32,
        use_kfwa_kuramoto: bool = True,
        use_kfwa_sync_bias: bool = True,
        use_lop_kuramoto: bool = True,
        use_lop_cluster_readout: bool = True,
        use_lop_higher_harmonics: bool = True,
    ) -> None:
        super().__init__()
        self.kfwa = SymbolicKFWAClassifier(
            num_symbols=num_symbols,
            max_support_k=max_support_k,
            target_dim=target_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            oscillator_dim=oscillator_dim,
            adapter_rank=adapter_rank,
            top_k=top_k,
            steps=steps,
            dt=dt,
            dropout=dropout,
            use_kuramoto=use_kfwa_kuramoto,
            use_sync_bias=use_kfwa_sync_bias,
            use_adapter=False,
        )
        self.lop = SymbolicLatentOscillatorPopulationClassifier(
            num_symbols=num_symbols,
            max_support_k=max_support_k,
            target_dim=target_dim,
            model_dim=model_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            rule_fourier_bands=rule_fourier_bands,
            oscillator_count=lop_oscillator_count,
            oscillator_steps=lop_steps,
            oscillator_dt=lop_dt,
            harmonics=lop_harmonics,
            cluster_count=lop_cluster_count,
            coupling_dim=lop_coupling_dim,
            use_kuramoto=use_lop_kuramoto,
            use_cluster_readout=use_lop_cluster_readout,
            use_higher_harmonics=use_lop_higher_harmonics,
            use_rule_fourier=True,
        )
        for param in self.kfwa.head.parameters():
            param.requires_grad = False
        for param in self.lop.head.parameters():
            param.requires_grad = False

        side_dim = model_dim * (2 if use_lop_cluster_readout else 1)
        self.use_lop_cluster_readout = use_lop_cluster_readout
        self.side_norm = nn.LayerNorm(side_dim)
        self.fusion_norm = nn.LayerNorm(model_dim + side_dim)
        self.lop_gate = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Sequential(
            nn.Linear(model_dim + side_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, target_dim),
        )
        self.last_states: list[KFWAState] = []
        self.last_theta: Tensor | None = None
        self.last_lop_features: Tensor | None = None

    def kfwa_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the normalized KFWA CLS state before classification."""

        mask = self.kfwa.token_mask(batch)
        tokens = self.kfwa.initial_tokens(batch)
        mu = self.kfwa.initial_mu(tokens, mask)
        states: list[KFWAState] = []
        for block in self.kfwa.blocks:
            tokens, mu, state = block(tokens, mu)
            tokens = tokens * mask
            mu = normalize_state(mu) * mask
            states.append(state)
        self.last_states = states
        self.kfwa.last_states = states
        return self.kfwa.norm(tokens[:, 0])

    def lop_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Return order and optional phase-cluster population readouts."""

        _, support, global_state = self.lop.encode_support_query(batch)
        particles = self.lop.initialize_particles(global_state, support, batch["support_mask"].bool())
        theta = self.lop.evolve_phases(particles)
        self.last_theta = theta
        self.lop.last_theta = theta
        features = [self.lop.order_features(theta)]
        if self.use_lop_cluster_readout:
            features.append(self.lop.cluster_features(theta, particles))
        side = self.side_norm(torch.cat(features, dim=-1))
        self.last_lop_features = side
        return side

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        cls_state = self.kfwa_state(batch)
        side = torch.sigmoid(self.lop_gate) * self.lop_features(batch)
        return self.head(self.fusion_norm(torch.cat([cls_state, side], dim=-1)))


class SymbolicLOPFeatureClassifier(nn.Module):
    """Latent oscillator population feature readout without a KFWA trunk."""

    def __init__(
        self,
        num_symbols: int = 10,
        max_support_k: int = 6,
        target_dim: int = 10,
        model_dim: int = 64,
        ff_dim: int = 128,
        dropout: float = 0.05,
        rule_fourier_bands: int = 4,
        oscillator_count: int = 64,
        oscillator_steps: int = 8,
        oscillator_dt: float = 0.1,
        harmonics: int = 4,
        cluster_count: int = 4,
        coupling_dim: int = 32,
        use_kuramoto: bool = False,
        use_cluster_readout: bool = True,
        use_higher_harmonics: bool = True,
        include_context: bool = False,
    ) -> None:
        super().__init__()
        self.lop = SymbolicLatentOscillatorPopulationClassifier(
            num_symbols=num_symbols,
            max_support_k=max_support_k,
            target_dim=target_dim,
            model_dim=model_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            rule_fourier_bands=rule_fourier_bands,
            oscillator_count=oscillator_count,
            oscillator_steps=oscillator_steps,
            oscillator_dt=oscillator_dt,
            harmonics=harmonics,
            cluster_count=cluster_count,
            coupling_dim=coupling_dim,
            use_kuramoto=use_kuramoto,
            use_cluster_readout=use_cluster_readout,
            use_higher_harmonics=use_higher_harmonics,
            use_rule_fourier=True,
        )
        for param in self.lop.head.parameters():
            param.requires_grad = False

        self.include_context = include_context
        self.use_cluster_readout = use_cluster_readout
        side_dim = model_dim * (2 if use_cluster_readout else 1)
        context_dim = 2 * model_dim if include_context else 0
        input_dim = side_dim + context_dim
        self.feature_norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, target_dim),
        )
        self.last_theta: Tensor | None = None
        self.last_lop_features: Tensor | None = None

    def features(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the pure LOP readout used by the classifier head."""

        query, support, global_state = self.lop.encode_support_query(batch)
        particles = self.lop.initialize_particles(global_state, support, batch["support_mask"].bool())
        theta = self.lop.evolve_phases(particles)
        self.last_theta = theta
        self.lop.last_theta = theta

        readout_parts: list[Tensor] = []
        if self.include_context:
            readout_parts.extend([query, global_state])
        readout_parts.append(self.lop.order_features(theta))
        if self.use_cluster_readout:
            readout_parts.append(self.lop.cluster_features(theta, particles))
        features = self.feature_norm(torch.cat(readout_parts, dim=-1))
        self.last_lop_features = features
        return features

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.head(self.features(batch))


class _SymbolicRuleTablePredictor(nn.Module):
    """Shared unique-function table and likelihood helpers for exact predictors."""

    def __init__(
        self,
        num_symbols: int = 10,
        rule_families: str = "affine,block,parity",
        target_dim: int = 10,
        support_noise: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        if target_dim != num_symbols:
            raise ValueError("Symbolic rule predictors expect target_dim == num_symbols.")
        if not 0.0 <= support_noise < 1.0:
            raise ValueError("support_noise must be in [0, 1).")
        rules = build_symbolic_rules(num_symbols=num_symbols, families=rule_families)
        representatives = symbolic_function_representatives(rules)
        function_ids = sorted(representatives)
        table = torch.tensor(
            [representatives[function_id].outputs for function_id in function_ids],
            dtype=torch.long,
        )
        self.register_buffer("rule_table", table, persistent=False)
        self.register_buffer("function_ids", torch.tensor(function_ids, dtype=torch.long), persistent=False)
        self.num_symbols = num_symbols
        self.support_noise = float(support_noise)
        self.dummy = nn.Parameter(torch.zeros(()))
        self.attribution_id = None
        self.uses_explicit_rule_features = False
        self.uses_kuramoto = False
        self.effective_model_dim = 0
        self.effective_ff_dim = 0
        self.population_size = 0

    def support_matches(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return [B, F, K] match indicators and valid-support mask."""

        table = self.rule_table.to(batch["query_digit"].device)
        support_digits = batch["support_digits"].long()
        support_labels = batch["support_labels"].long()
        support_mask = batch["support_mask"].bool().unsqueeze(1)
        predictions = table[:, support_digits].permute(1, 0, 2)
        return predictions.eq(support_labels.unsqueeze(1)), support_mask

    def support_match_counts(self, batch: dict[str, Tensor]) -> Tensor:
        """Count valid support observations matched by every candidate function."""

        matches, support_mask = self.support_matches(batch)
        return (matches & support_mask).sum(dim=-1)

    def support_log_likelihood(self, batch: dict[str, Tensor]) -> Tensor:
        """Return log p(noisy support | function) under symmetric label noise."""

        matches, support_mask = self.support_matches(batch)
        match_count = (matches & support_mask).sum(dim=-1)
        support_count = support_mask.sum(dim=-1)
        mismatch_count = support_count - match_count
        if self.support_noise == 0.0:
            zeros = torch.zeros_like(match_count, dtype=torch.float)
            impossible = torch.full_like(zeros, float("-inf"))
            return torch.where(mismatch_count.eq(0), zeros, impossible)

        log_correct = math.log1p(-self.support_noise)
        log_incorrect = math.log(self.support_noise / float(self.num_symbols - 1))
        return match_count.float() * log_correct + mismatch_count.float() * log_incorrect

    def query_outputs(self, batch: dict[str, Tensor]) -> Tensor:
        """Return [B, F] clean query labels for every candidate function."""

        table = self.rule_table.to(batch["query_digit"].device)
        return table[:, batch["query_digit"].long()].transpose(0, 1)

    def predictive_probs(self, batch: dict[str, Tensor], function_weights: Tensor) -> Tensor:
        """Marginalize candidate-function weights into query-label probabilities."""

        query_outputs = self.query_outputs(batch)
        probs = torch.zeros(
            query_outputs.shape[0],
            self.num_symbols,
            dtype=function_weights.dtype,
            device=query_outputs.device,
        )
        probs.scatter_add_(1, query_outputs, function_weights)
        return probs


class SymbolicBayesianPredictor(_SymbolicRuleTablePredictor):
    """Exact posterior predictive under a uniform prior over unique functions."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.readout_type = "exact_bayesian_predictor"
        self.predictor_type = "exact_bayesian"

    def posterior_function_probs(self, batch: dict[str, Tensor]) -> Tensor:
        """Return p(function | noisy support) for the benchmark generative model."""

        log_likelihood = self.support_log_likelihood(batch)
        finite_rows = torch.isfinite(log_likelihood).any(dim=1)
        if not finite_rows.all():
            fallback = self.support_match_counts(batch).float()
            log_likelihood = torch.where(finite_rows.unsqueeze(1), log_likelihood, fallback)
        return torch.softmax(log_likelihood, dim=1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        posterior = self.posterior_function_probs(batch)
        probs = self.predictive_probs(batch, posterior)
        return torch.log(probs.clamp_min(1.0e-12)) + self.dummy * 0.0


class SymbolicMaximumLikelihoodPredictor(_SymbolicRuleTablePredictor):
    """Maximum-likelihood function-set predictor with uniform tie voting."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.readout_type = "maximum_likelihood_predictor"
        self.predictor_type = "maximum_likelihood"

    def maximum_likelihood_mask(self, batch: dict[str, Tensor]) -> Tensor:
        """Return all candidate functions attaining maximum support likelihood."""

        log_likelihood = self.support_log_likelihood(batch)
        finite_rows = torch.isfinite(log_likelihood).any(dim=1)
        if not finite_rows.all():
            fallback = self.support_match_counts(batch).float()
            log_likelihood = torch.where(finite_rows.unsqueeze(1), log_likelihood, fallback)
        return log_likelihood.eq(log_likelihood.max(dim=1, keepdim=True).values)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        candidate_mask = self.maximum_likelihood_mask(batch)
        votes = self.predictive_probs(batch, candidate_mask.float())
        return torch.log(votes.clamp_min(1.0e-6)) + self.dummy * 0.0


class SymbolicRuleOracle(_SymbolicRuleTablePredictor):
    """Legacy max-match candidate-function voting oracle."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.readout_type = "legacy_max_match_predictor"
        self.predictor_type = "max_match"

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        scores = self.support_match_counts(batch)
        best_scores = scores.max(dim=1, keepdim=True).values
        candidate_mask = scores.eq(best_scores)
        votes = self.predictive_probs(batch, candidate_mask.float())
        return torch.log(votes.clamp_min(1.0e-6)) + self.dummy * 0.0
