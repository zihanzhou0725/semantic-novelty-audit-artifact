"""Frozen byte-level Transformer for the CLUTRR intervention."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


PAD_ID = 0
BYTE_OFFSET = 1
CLS_ID = 257
SEP_ID = 258
VOCAB_SIZE = 259
MAX_SEQUENCE_LENGTH = 512


def normalized_input(story: str, query: str) -> str:
    return f"{story} <sep> {query}"


def encode_parts(story: str, query: str) -> list[int]:
    story_bytes = story.encode("ascii")
    query_bytes = query.encode("ascii")
    return (
        [CLS_ID]
        + [value + BYTE_OFFSET for value in story_bytes]
        + [SEP_ID]
        + [value + BYTE_OFFSET for value in query_bytes]
    )


def encode_normalized_input(value: str) -> list[int]:
    parts = value.split(" <sep> ")
    if len(parts) != 2:
        raise ValueError("Normalized CLUTRR input must contain one <sep> token.")
    return encode_parts(parts[0], parts[1])


def collate_token_ids(
    sequences: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot collate an empty batch.")
    maximum = max(len(sequence) for sequence in sequences)
    if maximum > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Input length {maximum} exceeds {MAX_SEQUENCE_LENGTH}; "
            "truncation is forbidden."
        )
    input_ids = torch.full(
        (len(sequences), maximum),
        PAD_ID,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(sequences), maximum),
        dtype=torch.bool,
    )
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.as_tensor(
            sequence,
            dtype=torch.long,
        )
        attention_mask[index, :length] = True
    return input_ids, attention_mask


class ByteTransformerClassifier(nn.Module):
    """Four-layer learned-position Transformer with CLS pooling."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one.")
        self.num_classes = int(num_classes)
        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            128,
            padding_idx=PAD_ID,
        )
        self.position_embedding = nn.Embedding(MAX_SEQUENCE_LENGTH, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=4,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must be paired matrices.")
        if input_ids.shape[1] > MAX_SEQUENCE_LENGTH:
            raise ValueError("Sequence truncation is forbidden.")
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)
        states = self.token_embedding(input_ids)
        states = states + self.position_embedding(positions)
        states = self.encoder(
            states,
            src_key_padding_mask=~attention_mask.bool(),
        )
        return self.classifier(states[:, 0, :])


def inspect_architecture(num_classes: int) -> dict[str, Any]:
    model = ByteTransformerClassifier(num_classes)
    encoder_layers = list(model.encoder.layers)
    if len(encoder_layers) != 4:
        raise AssertionError("Registered encoder layer count changed.")
    first = encoder_layers[0]
    checks = {
        "attention_heads": first.self_attn.num_heads == 4,
        "classifier_output_classes": model.classifier.out_features
        == num_classes,
        "dropout": first.dropout.p == 0.1,
        "feed_forward_dimension": first.linear1.out_features == 512,
        "layers": len(encoder_layers) == 4,
        "maximum_sequence_length": model.position_embedding.num_embeddings
        == 512,
        "model_dimension": first.self_attn.embed_dim == 128,
        "position_embedding": "learned_absolute",
        "pooling": "CLS",
        "precision": next(model.parameters()).dtype == torch.float32,
        "vocabulary_size": model.token_embedding.num_embeddings == VOCAB_SIZE,
    }
    if not all(value is True or isinstance(value, str) for value in checks.values()):
        raise AssertionError(f"Model architecture mismatch: {checks}")
    return {
        "checks": checks,
        "num_classes": num_classes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
