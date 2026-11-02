from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


TASKS = ["context_permutation", "symbolic_rule_mapping"]
IMAGE_SOURCES = ["mnist", "fashion_mnist", "synthetic"]
CONTEXT_ENCODINGS = ["perm_matrix", "one_hot", "perm_vector"]
SYMBOLIC_SPLIT_MODES = ["strict_function", "legacy_identity"]
SYMBOLIC_RULE_FAMILIES = [
    "affine",
    "block",
    "parity",
    "quadratic",
    "piecewise_affine",
    "composition",
]
EXTENDED_SYMBOLIC_RULE_LIMIT = 400
COMPOSITION_RULE_COUNT = 400
COMPOSITION_RNG_SEED = 314159


@dataclass(frozen=True)
class DatasetBundle:
    """Container returned by task loader factories."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    image_shape: tuple[int, int, int]
    context_dim: int
    target_dim: int
    task_name: str
    task_type: Literal["classification"]
    metadata: dict[str, object]


@dataclass(frozen=True)
class SymbolicRule:
    """A deterministic digit-to-label rule used by symbolic benchmarks."""

    rule_id: int
    function_id: int
    family_id: int
    family: str
    params: tuple[int, ...]
    outputs: tuple[int, ...]


def group_symbolic_rules_by_function(rules: list[SymbolicRule]) -> dict[int, list[SymbolicRule]]:
    """Group parameter identities that implement the same complete mapping."""

    groups: dict[int, list[SymbolicRule]] = {}
    outputs_by_function: dict[int, tuple[int, ...]] = {}
    function_by_outputs: dict[tuple[int, ...], int] = {}
    for rule in rules:
        previous = outputs_by_function.setdefault(rule.function_id, rule.outputs)
        if previous != rule.outputs:
            raise ValueError(f"function_id={rule.function_id} maps to multiple output tables.")
        previous_function = function_by_outputs.setdefault(rule.outputs, rule.function_id)
        if previous_function != rule.function_id:
            raise ValueError(
                f"One output table maps to function_id={previous_function} and function_id={rule.function_id}."
            )
        groups.setdefault(rule.function_id, []).append(rule)
    return groups


def symbolic_function_representatives(rules: list[SymbolicRule]) -> dict[int, SymbolicRule]:
    """Return one deterministic representative identity for each unique function."""

    family_order = {family: index for index, family in enumerate(SYMBOLIC_RULE_FAMILIES)}
    return {
        function_id: min(
            aliases,
            key=lambda rule: (family_order[rule.family], rule.params, rule.rule_id),
        )
        for function_id, aliases in group_symbolic_rules_by_function(rules).items()
    }


def symbolic_function_split_checksum(
    rules: list[SymbolicRule],
    seen_function_ids: list[int],
    unseen_function_ids: list[int],
) -> str:
    """Hash the exact seen/unseen output tables for reproducibility audits."""

    representatives = symbolic_function_representatives(rules)
    payload = {
        "seen": [list(representatives[function_id].outputs) for function_id in sorted(seen_function_ids)],
        "unseen": [list(representatives[function_id].outputs) for function_id in sorted(unseen_function_ids)],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def symbolic_function_set_checksum(
    rules: list[SymbolicRule],
    function_ids: list[int],
) -> str:
    """Hash an ordered set of complete symbolic function mappings."""

    representatives = symbolic_function_representatives(rules)
    payload = [
        list(representatives[function_id].outputs)
        for function_id in sorted(function_ids)
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def symbolic_identity_split_checksum(
    rules: list[SymbolicRule],
    seen_rule_ids: list[int],
    unseen_rule_ids: list[int],
) -> str:
    """Hash a parameter-identity split without collapsing aliases."""

    payload = {
        "seen": [
            {
                "family": rules[rule_id].family,
                "params": list(rules[rule_id].params),
                "outputs": list(rules[rule_id].outputs),
            }
            for rule_id in sorted(seen_rule_ids)
        ],
        "unseen": [
            {
                "family": rules[rule_id].family,
                "params": list(rules[rule_id].params),
                "outputs": list(rules[rule_id].outputs),
            }
            for rule_id in sorted(unseen_rule_ids)
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def symbolic_function_id(outputs: tuple[int, ...]) -> int:
    """Return a stable positive int64 id derived only from a complete mapping."""

    encoded = json.dumps(list(outputs), separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


class SyntheticGlyphDataset(Dataset):
    """Small self-contained 28x28 digit-like dataset for smoke tests.

    This dataset is not intended as a publication benchmark. It exists so the
    platform can be tested without downloading MNIST.
    """

    def __init__(self, size: int, seed: int, train: bool = True) -> None:
        self.size = int(size)
        self.seed = int(seed) + (0 if train else 10_000)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rng = np.random.default_rng(self.seed + index)
        label = int(rng.integers(0, 10))
        image = np.zeros((28, 28), dtype=np.float32)
        row = 3 + (label * 2) % 20
        col = 3 + (label * 3) % 20
        image[row : row + 3, 4:24] = 1.0
        image[4:24, col : col + 3] = 1.0
        if label % 2:
            np.fill_diagonal(image[4:24, 4:24], 0.8)
        if label in {2, 5, 8}:
            image[8:20, 8:20] = np.maximum(image[8:20, 8:20], 0.45)
        image += 0.15 * rng.standard_normal(image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image[None, :, :]), label


def _load_torchvision_dataset(
    source: str,
    root: str,
    train: bool,
    download: bool,
) -> Dataset:
    """Load MNIST or FashionMNIST from torchvision."""

    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "torchvision is required for MNIST/FashionMNIST. "
            "Use --source synthetic for a no-download smoke test."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    if source == "mnist":
        return datasets.MNIST(root=root, train=train, transform=transform, download=download)
    if source == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)
    raise ValueError(f"Unsupported source: {source}")


def _subset_dataset(dataset: Dataset, limit: int | None, offset: int = 0) -> Dataset:
    """Return a deterministic subset when a limit is provided."""

    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(offset, min(offset + limit, len(dataset))))
    return Subset(dataset, indices)


def build_permutations(num_contexts: int, seed: int) -> np.ndarray:
    """Build deterministic label permutations for context-conditioned labels."""

    if num_contexts < 1:
        raise ValueError("num_contexts must be positive.")
    rng = np.random.default_rng(seed)
    permutations = [np.arange(10, dtype=np.int64)]
    for _ in range(1, num_contexts):
        perm = np.arange(10, dtype=np.int64)
        rng.shuffle(perm)
        permutations.append(perm)
    return np.stack(permutations, axis=0)


def encode_context(
    context_id: int,
    permutation: np.ndarray,
    num_contexts: int,
    encoding: str,
) -> np.ndarray:
    """Encode a context as a vector visible to every model."""

    if encoding == "one_hot":
        vec = np.zeros(num_contexts, dtype=np.float32)
        vec[context_id] = 1.0
        return vec
    if encoding == "perm_vector":
        return (permutation.astype(np.float32) / 9.0) * 2.0 - 1.0
    if encoding == "perm_matrix":
        matrix = np.zeros((10, 10), dtype=np.float32)
        for source_label, target_label in enumerate(permutation):
            matrix[source_label, int(target_label)] = 1.0
        return matrix.reshape(-1)
    raise ValueError(f"Unsupported context encoding: {encoding}")


class ContextPermutationDataset(Dataset):
    """Wrap an image classification dataset with context-conditioned labels."""

    def __init__(
        self,
        base: Dataset,
        permutations: np.ndarray,
        context_ids: list[int],
        context_encoding: str,
        seen_context_cutoff: int,
    ) -> None:
        self.base = base
        self.permutations = permutations
        self.context_ids = list(context_ids)
        self.context_encoding = context_encoding
        self.seen_context_cutoff = int(seen_context_cutoff)

    def __len__(self) -> int:
        return len(self.base) * len(self.context_ids)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        base_index = index // len(self.context_ids)
        context_id = self.context_ids[index % len(self.context_ids)]
        image, label = self.base[base_index]
        if not torch.is_tensor(image):
            image = torch.as_tensor(np.asarray(image), dtype=torch.float32).unsqueeze(0) / 255.0
        label_int = int(label)
        permutation = self.permutations[context_id]
        target = int(permutation[label_int])
        context = encode_context(
            context_id=context_id,
            permutation=permutation,
            num_contexts=int(self.permutations.shape[0]),
            encoding=self.context_encoding,
        )
        batch = {
            "image": image.float(),
            "context": torch.from_numpy(context).float(),
            "context_id": torch.tensor(context_id, dtype=torch.long),
            "base_label": torch.tensor(label_int, dtype=torch.long),
            "seen_context": torch.tensor(context_id < self.seen_context_cutoff, dtype=torch.bool),
        }
        return batch, torch.tensor(target, dtype=torch.long)


class SymbolicRuleMappingDataset(Dataset):
    """Variable-K symbolic rule inference dataset.

    Each example is (query_digit, support_pairs) -> query_label. Support
    pairs are padded to max_support_k and accompanied by support_mask, so K can
    vary across examples while the DataLoader still collates tensors normally.
    """

    def __init__(
        self,
        size: int,
        rules: list[SymbolicRule],
        active_function_ids: list[int],
        seen_function_ids: set[int],
        min_support_k: int,
        max_support_k: int,
        query_out_of_support_prob: float,
        support_noise: float,
        seed: int,
    ) -> None:
        if not rules:
            raise ValueError("rules must not be empty.")
        num_symbols = len(rules[0].outputs)
        if min_support_k < 1:
            raise ValueError("min_support_k must be positive.")
        if max_support_k < min_support_k:
            raise ValueError("max_support_k must be >= min_support_k.")
        if max_support_k > num_symbols:
            raise ValueError(f"max_support_k cannot exceed num_symbols={num_symbols}.")
        if not 0.0 <= query_out_of_support_prob <= 1.0:
            raise ValueError("query_out_of_support_prob must be in [0, 1].")
        if not 0.0 <= support_noise < 1.0:
            raise ValueError("support_noise must be in [0, 1).")

        self.size = int(size)
        self.rules = rules
        self.function_groups = group_symbolic_rules_by_function(rules)
        self.function_representatives = symbolic_function_representatives(rules)
        self.active_function_ids = sorted(set(active_function_ids))
        self.seen_function_ids = set(seen_function_ids)
        if not self.active_function_ids:
            raise ValueError("active_function_ids must not be empty.")
        unknown_function_ids = sorted(set(self.active_function_ids) - set(self.function_representatives))
        if unknown_function_ids:
            raise ValueError(f"Unknown active function ids: {unknown_function_ids}.")
        unknown_seen_ids = sorted(self.seen_function_ids - set(self.function_representatives))
        if unknown_seen_ids:
            raise ValueError(f"Unknown seen function ids: {unknown_seen_ids}.")
        self.min_support_k = int(min_support_k)
        self.max_support_k = int(max_support_k)
        self.query_out_of_support_prob = float(query_out_of_support_prob)
        self.support_noise = float(support_noise)
        self.seed = int(seed)
        self.num_symbols = num_symbols

    def __len__(self) -> int:
        return self.size

    def noisy_label(self, rng: np.random.Generator, clean_label: int) -> int:
        """Apply optional support-label corruption."""

        if self.support_noise <= 0.0 or rng.random() >= self.support_noise:
            return int(clean_label)
        candidates = [label for label in range(self.num_symbols) if label != int(clean_label)]
        return int(rng.choice(candidates))

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        function_id = int(rng.choice(self.active_function_ids))
        rule = self.function_representatives[function_id]
        seen_function = function_id in self.seen_function_ids
        return self._build_item(
            rng=rng,
            rule=rule,
            seen_identity=seen_function,
            seen_function=seen_function,
        )

    def _build_item(
        self,
        *,
        rng: np.random.Generator,
        rule: SymbolicRule,
        seen_identity: bool,
        seen_function: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Generate one episode from a selected rule identity."""

        function_id = rule.function_id
        support_k = int(rng.integers(self.min_support_k, self.max_support_k + 1))
        support_digits = rng.choice(self.num_symbols, size=support_k, replace=False).astype(np.int64)

        if rng.random() < self.query_out_of_support_prob and support_k < self.num_symbols:
            candidates = [digit for digit in range(self.num_symbols) if digit not in set(support_digits.tolist())]
            query_digit = int(rng.choice(candidates))
        else:
            query_digit = int(rng.integers(0, self.num_symbols))

        support_labels = np.array(
            [self.noisy_label(rng, rule.outputs[int(digit)]) for digit in support_digits],
            dtype=np.int64,
        )
        padded_digits = np.zeros(self.max_support_k, dtype=np.int64)
        padded_labels = np.zeros(self.max_support_k, dtype=np.int64)
        support_mask = np.zeros(self.max_support_k, dtype=bool)
        padded_digits[:support_k] = support_digits
        padded_labels[:support_k] = support_labels
        support_mask[:support_k] = True
        query_in_support = bool(query_digit in set(support_digits.tolist()))
        target = int(rule.outputs[query_digit])

        batch = {
            "query_digit": torch.tensor(query_digit, dtype=torch.long),
            "support_digits": torch.from_numpy(padded_digits).long(),
            "support_labels": torch.from_numpy(padded_labels).long(),
            "support_mask": torch.from_numpy(support_mask),
            "support_k": torch.tensor(support_k, dtype=torch.long),
            "rule_id": torch.tensor(rule.rule_id, dtype=torch.long),
            "identity_id": torch.tensor(rule.rule_id, dtype=torch.long),
            "function_id": torch.tensor(function_id, dtype=torch.long),
            "function_alias_count": torch.tensor(len(self.function_groups[function_id]), dtype=torch.long),
            "rule_family": torch.tensor(rule.family_id, dtype=torch.long),
            "context_id": torch.tensor(rule.family_id, dtype=torch.long),
            "seen_identity": torch.tensor(seen_identity, dtype=torch.bool),
            "seen_function": torch.tensor(seen_function, dtype=torch.bool),
            "seen_context": torch.tensor(seen_identity, dtype=torch.bool),
            "query_in_support": torch.tensor(query_in_support, dtype=torch.bool),
        }
        return batch, torch.tensor(target, dtype=torch.long)


class SymbolicIdentityRuleMappingDataset(SymbolicRuleMappingDataset):
    """Legacy identity-uniform episode sampler used only for split audits."""

    def __init__(
        self,
        size: int,
        rules: list[SymbolicRule],
        active_rule_ids: list[int],
        seen_rule_ids: set[int],
        seen_function_ids: set[int],
        min_support_k: int,
        max_support_k: int,
        query_out_of_support_prob: float,
        support_noise: float,
        seed: int,
    ) -> None:
        super().__init__(
            size=size,
            rules=rules,
            active_function_ids=sorted(symbolic_function_representatives(rules)),
            seen_function_ids=seen_function_ids,
            min_support_k=min_support_k,
            max_support_k=max_support_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed,
        )
        self.active_rule_ids = sorted(set(active_rule_ids))
        self.seen_rule_ids = set(seen_rule_ids)
        if not self.active_rule_ids:
            raise ValueError("active_rule_ids must not be empty.")
        if self.active_rule_ids[-1] >= len(rules) or self.active_rule_ids[0] < 0:
            raise ValueError("active_rule_ids contains an unknown identity.")
        if not self.seen_rule_ids.issubset(set(range(len(rules)))):
            raise ValueError("seen_rule_ids contains an unknown identity.")

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        rule_id = int(rng.choice(self.active_rule_ids))
        rule = self.rules[rule_id]
        return self._build_item(
            rng=rng,
            rule=rule,
            seen_identity=rule_id in self.seen_rule_ids,
            seen_function=rule.function_id in self.seen_function_ids,
        )


def context_dim(num_contexts: int, encoding: str) -> int:
    """Return context feature dimension."""

    if encoding == "one_hot":
        return num_contexts
    if encoding == "perm_vector":
        return 10
    if encoding == "perm_matrix":
        return 100
    raise ValueError(f"Unsupported context encoding: {encoding}")


def parse_rule_families(raw: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse and validate symbolic rule family names."""

    if isinstance(raw, str):
        families = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    else:
        families = [str(chunk).strip() for chunk in raw if str(chunk).strip()]
    unknown = [family for family in families if family not in SYMBOLIC_RULE_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown symbolic rule families: {unknown}.")
    if not families:
        raise ValueError("At least one symbolic rule family is required.")
    return families


def _unit_multipliers(num_symbols: int) -> list[int]:
    """Return invertible multipliers modulo num_symbols."""

    return [value for value in range(1, num_symbols) if math.gcd(value, num_symbols) == 1]


def _dedupe_rule_candidates(candidates: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Remove duplicate output tables while keeping deterministic order."""

    unique: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen_outputs: set[tuple[int, ...]] = set()
    for params, outputs in candidates:
        if outputs in seen_outputs:
            continue
        seen_outputs.add(outputs)
        unique.append((params, outputs))
    return unique


def _strided_limit_candidates(
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]],
    limit: int = EXTENDED_SYMBOLIC_RULE_LIMIT,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Keep a deterministic spread of candidates when a generated family is large."""

    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / float(limit)
    indices = [int(idx * step) for idx in range(limit)]
    return [candidates[index] for index in indices]


def affine_outputs(a: int, b: int, num_symbols: int) -> tuple[int, ...]:
    """Return outputs for y = a*x + b mod N."""

    return tuple(int((a * x + b) % num_symbols) for x in range(num_symbols))


def block_outputs(low_shift: int, high_shift: int, swap_blocks: int, num_symbols: int) -> tuple[int, ...]:
    """Return outputs for two half-block cyclic shifts with optional block swap."""

    half = num_symbols // 2
    outputs: list[int] = []
    for x in range(num_symbols):
        source_block = 0 if x < half else 1
        local = x % half
        shift = low_shift if source_block == 0 else high_shift
        target_block = source_block ^ int(bool(swap_blocks))
        outputs.append(int(target_block * half + ((local + shift) % half)))
    return tuple(outputs)


def parity_outputs(even_shift: int, odd_shift: int, parity_flip: int, num_symbols: int) -> tuple[int, ...]:
    """Return outputs for parity-group cyclic shifts with optional parity flip."""

    half = num_symbols // 2
    outputs: list[int] = []
    for x in range(num_symbols):
        source_parity = x % 2
        local = x // 2
        shift = even_shift if source_parity == 0 else odd_shift
        target_parity = source_parity ^ int(bool(parity_flip))
        outputs.append(int(2 * ((local + shift) % half) + target_parity))
    return tuple(outputs)


def quadratic_outputs(a: int, b: int, c: int, num_symbols: int) -> tuple[int, ...]:
    """Return outputs for y = a*x^2 + b*x + c mod N."""

    return tuple(int((a * x * x + b * x + c) % num_symbols) for x in range(num_symbols))


def piecewise_affine_outputs(
    base_slope_index: int,
    slope_step: int,
    base_shift: int,
    shift_step: int,
    reverse_segments: int,
    num_symbols: int,
) -> tuple[int, ...]:
    """Return outputs for segment-conditioned affine rules.

    The input domain is split into four segments when possible, or two segments
    for smaller even vocabularies. Each segment uses a related affine map, so
    the family is harder than block shifts but remains structured.
    """

    units = _unit_multipliers(num_symbols)
    if not units:
        raise ValueError("piecewise_affine requires at least one unit multiplier.")
    if num_symbols % 4 == 0:
        num_segments = 4
    elif num_symbols % 2 == 0:
        num_segments = 2
    else:
        raise ValueError("piecewise_affine requires an even num_symbols.")

    outputs: list[int] = []
    for x in range(num_symbols):
        segment = min(num_segments - 1, (x * num_segments) // num_symbols)
        slope_segment = (num_segments - 1 - segment) if reverse_segments else segment
        slope = units[(base_slope_index + slope_segment * slope_step) % len(units)]
        shift = (base_shift + segment * shift_step) % num_symbols
        outputs.append(int((slope * x + shift) % num_symbols))
    return tuple(outputs)


def affine_rule_candidates(num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate affine rule candidates."""

    return [
        ((a, b), affine_outputs(a, b, num_symbols))
        for a in _unit_multipliers(num_symbols)
        for b in range(num_symbols)
    ]


def block_rule_candidates(num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate block rule candidates."""

    half = num_symbols // 2
    return [
        ((low_shift, high_shift, swap_blocks), block_outputs(low_shift, high_shift, swap_blocks, num_symbols))
        for low_shift in range(half)
        for high_shift in range(half)
        for swap_blocks in (0, 1)
    ]


def parity_rule_candidates(num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate parity rule candidates."""

    half = num_symbols // 2
    return [
        ((even_shift, odd_shift, parity_flip), parity_outputs(even_shift, odd_shift, parity_flip, num_symbols))
        for even_shift in range(half)
        for odd_shift in range(half)
        for parity_flip in (0, 1)
    ]


def quadratic_rule_candidates(num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate a balanced subset of quadratic rule candidates."""

    units = _unit_multipliers(num_symbols)
    linear_terms = [0, *units]
    candidates = [
        ((a, b, c), quadratic_outputs(a, b, c, num_symbols))
        for a in units
        for b in linear_terms
        for c in range(num_symbols)
    ]
    return _strided_limit_candidates(_dedupe_rule_candidates(candidates))


def piecewise_affine_rule_candidates(num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate a balanced subset of piecewise-affine candidates."""

    units = _unit_multipliers(num_symbols)
    shift_steps = list(range(max(1, num_symbols // 4)))
    candidates = [
        (
            (units[base_slope_index], slope_step, base_shift, shift_step, reverse_segments),
            piecewise_affine_outputs(
                base_slope_index=base_slope_index,
                slope_step=slope_step,
                base_shift=base_shift,
                shift_step=shift_step,
                reverse_segments=reverse_segments,
                num_symbols=num_symbols,
            ),
        )
        for base_slope_index in range(len(units))
        for slope_step in range(len(units))
        for base_shift in range(num_symbols)
        for shift_step in shift_steps
        for reverse_segments in (0, 1)
    ]
    return _strided_limit_candidates(_dedupe_rule_candidates(candidates))


def primitive_composition_candidates(num_symbols: int) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Return primitive rules available to the composition family."""

    primitive_builders = [
        ("affine", affine_rule_candidates),
        ("block", block_rule_candidates),
        ("parity", parity_rule_candidates),
        ("quadratic", quadratic_rule_candidates),
        ("piecewise_affine", piecewise_affine_rule_candidates),
    ]
    primitives: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for family, builder in primitive_builders:
        for params, outputs in builder(num_symbols):
            primitives.append((family, params, outputs))
    return primitives


def composition_rule_candidates(num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Sample deterministic compositions y = g2(g1(x)) from primitive rules."""

    primitives = primitive_composition_candidates(num_symbols)
    if len(primitives) < 2:
        raise ValueError("composition requires at least two primitive rules.")

    rng = np.random.default_rng(COMPOSITION_RNG_SEED + num_symbols)
    sampled_pairs: set[tuple[int, int]] = set()
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen_outputs: set[tuple[int, ...]] = set()
    max_attempts = COMPOSITION_RULE_COUNT * 50
    attempts = 0
    while len(candidates) < COMPOSITION_RULE_COUNT and attempts < max_attempts:
        attempts += 1
        first_idx = int(rng.integers(0, len(primitives)))
        second_idx = int(rng.integers(0, len(primitives)))
        if (first_idx, second_idx) in sampled_pairs:
            continue
        sampled_pairs.add((first_idx, second_idx))
        first_outputs = primitives[first_idx][2]
        second_outputs = primitives[second_idx][2]
        outputs = tuple(int(second_outputs[first_outputs[x]]) for x in range(num_symbols))
        if outputs in seen_outputs:
            continue
        seen_outputs.add(outputs)
        candidates.append(((first_idx, second_idx), outputs))
    return candidates


def rule_candidates_for_family(family: str, num_symbols: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate candidate parameter/output pairs for one symbolic family."""

    if family == "affine":
        return affine_rule_candidates(num_symbols)
    if family == "block":
        return block_rule_candidates(num_symbols)
    if family == "parity":
        return parity_rule_candidates(num_symbols)
    if family == "quadratic":
        return quadratic_rule_candidates(num_symbols)
    if family == "piecewise_affine":
        return piecewise_affine_rule_candidates(num_symbols)
    if family == "composition":
        return composition_rule_candidates(num_symbols)
    raise ValueError(f"Unsupported symbolic rule family: {family}")


def build_symbolic_rules(
    num_symbols: int = 10,
    families: str | list[str] | tuple[str, ...] = "affine,block,parity",
) -> list[SymbolicRule]:
    """Enumerate structured digit-mapping rules for symbolic experiments."""

    if num_symbols < 2:
        raise ValueError("num_symbols must be at least 2.")
    selected = parse_rule_families(families)
    if num_symbols % 2 and any(family in {"block", "parity", "piecewise_affine", "composition"} for family in selected):
        raise ValueError("block, parity, piecewise_affine, and composition symbolic rule families require an even num_symbols.")
    rules: list[SymbolicRule] = []
    outputs_by_function_id: dict[int, tuple[int, ...]] = {}

    def add_rule(family_id: int, family: str, params: tuple[int, ...], outputs: tuple[int, ...]) -> None:
        function_id = symbolic_function_id(outputs)
        previous_outputs = outputs_by_function_id.setdefault(function_id, outputs)
        if previous_outputs != outputs:
            raise RuntimeError("SHA-256 function id collision detected.")
        rules.append(
            SymbolicRule(
                rule_id=len(rules),
                function_id=function_id,
                family_id=family_id,
                family=family,
                params=params,
                outputs=outputs,
            )
        )

    for family_id, family in enumerate(selected):
        for params, outputs in rule_candidates_for_family(family, num_symbols):
            add_rule(family_id, family, params, outputs)
    return rules


def build_symbolic_family_transfer_evaluation(
    *,
    num_symbols: int,
    family: str,
    excluded_families: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[SymbolicRule], list[int], dict[str, object]]:
    """Build one family test universe after excluding complete-map overlaps."""

    selected = parse_rule_families(family)
    if len(selected) != 1:
        raise ValueError("Family-transfer evaluation requires exactly one family.")
    family_name = selected[0]
    rules = build_symbolic_rules(num_symbols=num_symbols, families=family_name)
    representatives = symbolic_function_representatives(rules)
    all_function_ids = set(representatives)

    excluded_names: list[str] = []
    excluded_function_ids: set[int] = set()
    if excluded_families:
        excluded_names = parse_rule_families(excluded_families)
        if family_name in excluded_names:
            raise ValueError("The evaluated family cannot also be excluded.")
        excluded_rules = build_symbolic_rules(
            num_symbols=num_symbols,
            families=excluded_names,
        )
        excluded_function_ids = set(symbolic_function_representatives(excluded_rules))

    overlap_ids = sorted(all_function_ids & excluded_function_ids)
    eligible_function_ids = sorted(all_function_ids - excluded_function_ids)
    if not eligible_function_ids:
        raise ValueError(
            f"No eligible {family_name} functions remain after overlap exclusion."
        )
    metadata: dict[str, object] = {
        "family": family_name,
        "excluded_families": excluded_names,
        "num_family_functions": len(all_function_ids),
        "num_excluded_overlaps": len(overlap_ids),
        "num_eligible_functions": len(eligible_function_ids),
        "overlap_function_ids": overlap_ids,
        "eligible_function_checksum": symbolic_function_set_checksum(
            rules,
            eligible_function_ids,
        ),
    }
    return rules, eligible_function_ids, metadata


def split_symbolic_functions(
    rules: list[SymbolicRule],
    holdout_rule_frac: float,
    seed: int,
    stratify_by_family: bool = True,
) -> tuple[list[int], list[int]]:
    """Split unique functions so no complete mapping appears on both sides."""

    if not 0.0 < holdout_rule_frac < 1.0:
        raise ValueError("holdout_rule_frac must be in (0, 1).")
    rng = np.random.default_rng(seed)
    representatives = symbolic_function_representatives(rules)
    strata: dict[str, list[int]] = {}
    for function_id, rule in representatives.items():
        stratum = rule.family if stratify_by_family else "all"
        strata.setdefault(stratum, []).append(function_id)

    seen: list[int] = []
    holdout: list[int] = []
    for stratum in sorted(strata):
        function_ids = np.array(sorted(strata[stratum]), dtype=np.int64)
        rng.shuffle(function_ids)
        holdout_count = int(round(len(function_ids) * holdout_rule_frac))
        if len(function_ids) > 1:
            holdout_count = min(max(holdout_count, 1), len(function_ids) - 1)
        holdout.extend(int(idx) for idx in function_ids[:holdout_count])
        seen.extend(int(idx) for idx in function_ids[holdout_count:])
    seen.sort()
    holdout.sort()
    if not seen:
        raise ValueError("The symbolic function split left no seen functions.")
    return seen, holdout


def split_symbolic_rules(
    rules: list[SymbolicRule],
    holdout_rule_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Backward-compatible alias returning seen and unseen function ids."""

    return split_symbolic_functions(rules, holdout_rule_frac=holdout_rule_frac, seed=seed)


def split_symbolic_identities(
    rules: list[SymbolicRule],
    holdout_rule_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Reproduce the legacy unstratified parameter-identity split."""

    if not 0.0 < holdout_rule_frac < 1.0:
        raise ValueError("holdout_rule_frac must be in (0, 1).")
    rng = np.random.default_rng(seed)
    rule_ids = np.arange(len(rules), dtype=np.int64)
    rng.shuffle(rule_ids)
    holdout_count = max(1, int(round(len(rule_ids) * holdout_rule_frac)))
    holdout = sorted(int(rule_id) for rule_id in rule_ids[:holdout_count])
    seen = sorted(int(rule_id) for rule_id in rule_ids[holdout_count:])
    if not seen:
        raise ValueError("The symbolic identity split left no seen identities.")
    return seen, holdout


def get_context_permutation_loaders(
    source: str = "mnist",
    data_root: str = "kfwa_experiments/data",
    download: bool = False,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 0,
    num_contexts: int = 8,
    holdout_contexts: int = 2,
    context_encoding: str = "perm_matrix",
    train_limit: int | None = 20_000,
    val_limit: int | None = 5_000,
    test_limit: int | None = 10_000,
    quick: bool = False,
    **_: object,
) -> DatasetBundle:
    """Build loaders for Benchmark 1: context-conditioned label permutation."""

    if source not in IMAGE_SOURCES:
        raise ValueError(f"source must be one of {IMAGE_SOURCES}.")
    if context_encoding not in CONTEXT_ENCODINGS:
        raise ValueError(f"context_encoding must be one of {CONTEXT_ENCODINGS}.")
    if holdout_contexts < 0 or holdout_contexts >= num_contexts:
        raise ValueError("holdout_contexts must be in [0, num_contexts).")

    if quick:
        train_limit = min(train_limit or 512, 512)
        val_limit = min(val_limit or 128, 128)
        test_limit = min(test_limit or 256, 256)
        source = "synthetic" if source != "synthetic" else source

    if source == "synthetic":
        train_base_full: Dataset = SyntheticGlyphDataset(size=max((train_limit or 512) + (val_limit or 128), 640), seed=seed, train=True)
        test_base_full: Dataset = SyntheticGlyphDataset(size=test_limit or 256, seed=seed, train=False)
    else:
        train_base_full = _load_torchvision_dataset(source, data_root, train=True, download=download)
        test_base_full = _load_torchvision_dataset(source, data_root, train=False, download=download)

    train_limit_final = train_limit or max(1, len(train_base_full) - (val_limit or 0))
    val_limit_final = val_limit or min(5_000, max(1, len(train_base_full) - train_limit_final))
    train_base = _subset_dataset(train_base_full, train_limit_final, offset=0)
    val_base = _subset_dataset(train_base_full, val_limit_final, offset=train_limit_final)
    test_base = _subset_dataset(test_base_full, test_limit, offset=0)

    permutations = build_permutations(num_contexts=num_contexts, seed=seed + 123)
    seen_cutoff = num_contexts - holdout_contexts
    train_contexts = list(range(seen_cutoff))
    eval_contexts = list(range(num_contexts))

    train_set = ContextPermutationDataset(
        train_base,
        permutations=permutations,
        context_ids=train_contexts,
        context_encoding=context_encoding,
        seen_context_cutoff=seen_cutoff,
    )
    val_set = ContextPermutationDataset(
        val_base,
        permutations=permutations,
        context_ids=eval_contexts,
        context_encoding=context_encoding,
        seen_context_cutoff=seen_cutoff,
    )
    test_set = ContextPermutationDataset(
        test_base,
        permutations=permutations,
        context_ids=eval_contexts,
        context_encoding=context_encoding,
        seen_context_cutoff=seen_cutoff,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DatasetBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        image_shape=(1, 28, 28),
        context_dim=context_dim(num_contexts, context_encoding),
        target_dim=10,
        task_name="context_permutation",
        task_type="classification",
        metadata={
            "source": source,
            "num_contexts": num_contexts,
            "holdout_contexts": holdout_contexts,
            "seen_contexts": seen_cutoff,
            "context_encoding": context_encoding,
            "train_examples": len(train_set),
            "val_examples": len(val_set),
            "test_examples": len(test_set),
            "permutations": permutations.tolist(),
        },
    )


def get_symbolic_rule_mapping_loaders(
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 0,
    train_limit: int | None = 50_000,
    val_limit: int | None = 10_000,
    test_limit: int | None = 10_000,
    quick: bool = False,
    num_symbols: int = 10,
    support_min_k: int = 2,
    support_max_k: int = 6,
    rule_families: str = "affine,block,parity",
    holdout_rule_frac: float = 0.25,
    query_out_of_support_prob: float = 0.5,
    support_noise: float = 0.0,
    symbolic_split_mode: str = "strict_function",
    **_: object,
) -> DatasetBundle:
    """Build loaders for the symbolic variable-K rule mapping benchmark."""

    if symbolic_split_mode not in SYMBOLIC_SPLIT_MODES:
        raise ValueError(
            f"symbolic_split_mode must be one of {SYMBOLIC_SPLIT_MODES}."
        )
    if quick:
        train_limit = min(train_limit or 1024, 1024)
        val_limit = min(val_limit or 512, 512)
        test_limit = min(test_limit or 512, 512)
        batch_size = min(batch_size, 128)

    train_size = int(train_limit or 50_000)
    val_size = int(val_limit or 10_000)
    test_size = int(test_limit or 10_000)
    rules = build_symbolic_rules(num_symbols=num_symbols, families=rule_families)
    function_groups = group_symbolic_rules_by_function(rules)
    function_representatives = symbolic_function_representatives(rules)
    identity_split_checksum: str | None = None
    leaked_unseen_identities = 0
    leaked_unseen_identity_fraction = 0.0

    if symbolic_split_mode == "strict_function":
        seen_function_ids, unseen_function_ids = split_symbolic_functions(
            rules,
            holdout_rule_frac=holdout_rule_frac,
            seed=seed + 7919,
        )
        eval_function_ids = seen_function_ids + unseen_function_ids
        seen_function_set = set(seen_function_ids)
        seen_rule_ids = sorted(
            rule.rule_id for rule in rules if rule.function_id in seen_function_set
        )
        unseen_rule_ids = sorted(
            rule.rule_id for rule in rules if rule.function_id not in seen_function_set
        )
        train_set: Dataset = SymbolicRuleMappingDataset(
            size=train_size,
            rules=rules,
            active_function_ids=seen_function_ids,
            seen_function_ids=seen_function_set,
            min_support_k=support_min_k,
            max_support_k=support_max_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed + 10_000,
        )
        val_set: Dataset = SymbolicRuleMappingDataset(
            size=val_size,
            rules=rules,
            active_function_ids=eval_function_ids,
            seen_function_ids=seen_function_set,
            min_support_k=support_min_k,
            max_support_k=support_max_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed + 20_000,
        )
        test_set: Dataset = SymbolicRuleMappingDataset(
            size=test_size,
            rules=rules,
            active_function_ids=eval_function_ids,
            seen_function_ids=seen_function_set,
            min_support_k=support_min_k,
            max_support_k=support_max_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed + 30_000,
        )
        split_unit = "complete_function_mapping"
        split_stratification = "canonical_family"
        function_sampling = "uniform"
    else:
        seen_rule_ids, unseen_rule_ids = split_symbolic_identities(
            rules,
            holdout_rule_frac=holdout_rule_frac,
            seed=seed + 7919,
        )
        seen_rule_set = set(seen_rule_ids)
        seen_function_set = {
            rules[rule_id].function_id for rule_id in seen_rule_ids
        }
        heldout_function_set = {
            rules[rule_id].function_id for rule_id in unseen_rule_ids
        }
        all_function_ids = set(function_representatives)
        seen_function_ids = sorted(seen_function_set)
        unseen_function_ids = sorted(all_function_ids - seen_function_set)
        leaked_unseen_identities = sum(
            rules[rule_id].function_id in seen_function_set
            for rule_id in unseen_rule_ids
        )
        leaked_unseen_identity_fraction = (
            leaked_unseen_identities / len(unseen_rule_ids)
        )
        identity_split_checksum = symbolic_identity_split_checksum(
            rules,
            seen_rule_ids=seen_rule_ids,
            unseen_rule_ids=unseen_rule_ids,
        )
        all_rule_ids = seen_rule_ids + unseen_rule_ids
        train_set = SymbolicIdentityRuleMappingDataset(
            size=train_size,
            rules=rules,
            active_rule_ids=seen_rule_ids,
            seen_rule_ids=seen_rule_set,
            seen_function_ids=seen_function_set,
            min_support_k=support_min_k,
            max_support_k=support_max_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed + 10_000,
        )
        val_set = SymbolicIdentityRuleMappingDataset(
            size=val_size,
            rules=rules,
            active_rule_ids=all_rule_ids,
            seen_rule_ids=seen_rule_set,
            seen_function_ids=seen_function_set,
            min_support_k=support_min_k,
            max_support_k=support_max_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed + 20_000,
        )
        test_set = SymbolicIdentityRuleMappingDataset(
            size=test_size,
            rules=rules,
            active_rule_ids=all_rule_ids,
            seen_rule_ids=seen_rule_set,
            seen_function_ids=seen_function_set,
            min_support_k=support_min_k,
            max_support_k=support_max_k,
            query_out_of_support_prob=query_out_of_support_prob,
            support_noise=support_noise,
            seed=seed + 30_000,
        )
        split_unit = "parameter_identity"
        split_stratification = "none"
        function_sampling = "identity_uniform_alias_weighted"

    seen_outputs = {
        function_representatives[function_id].outputs
        for function_id in seen_function_set
    }
    heldout_identity_outputs = {
        rules[rule_id].outputs for rule_id in unseen_rule_ids
    }
    function_split_overlap = len(seen_outputs & heldout_identity_outputs)
    if symbolic_split_mode == "strict_function" and function_split_overlap:
        raise RuntimeError(
            "Function-disjoint split failed with "
            f"{function_split_overlap} overlapping mappings."
        )
    split_checksum = symbolic_function_split_checksum(
        rules,
        seen_function_ids=seen_function_ids,
        unseen_function_ids=unseen_function_ids,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DatasetBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        image_shape=(0, 0, 0),
        context_dim=0,
        target_dim=num_symbols,
        task_name="symbolic_rule_mapping",
        task_type="classification",
        metadata={
            "source": "symbolic",
            "num_symbols": num_symbols,
            "support_min_k": support_min_k,
            "support_max_k": support_max_k,
            "rule_families": parse_rule_families(rule_families),
            "num_rules": len(rules),
            "num_rule_identities": len(rules),
            "num_functions": len(function_groups),
            "seen_rules": len(seen_rule_ids),
            "unseen_rules": len(unseen_rule_ids),
            "seen_functions": len(seen_function_ids),
            "unseen_functions": len(unseen_function_ids),
            "holdout_rule_frac": holdout_rule_frac,
            "holdout_function_frac": holdout_rule_frac,
            "symbolic_split_mode": symbolic_split_mode,
            "split_unit": split_unit,
            "function_split_stratification": split_stratification,
            "function_sampling": function_sampling,
            "function_split_checksum": split_checksum,
            "function_split_overlap": function_split_overlap,
            "identity_split_checksum": identity_split_checksum,
            "leaked_unseen_identities": leaked_unseen_identities,
            "leaked_unseen_identity_fraction": leaked_unseen_identity_fraction,
            "query_out_of_support_prob": query_out_of_support_prob,
            "support_noise": support_noise,
            "train_examples": len(train_set),
            "val_examples": len(val_set),
            "test_examples": len(test_set),
            "seen_rule_ids": seen_rule_ids,
            "unseen_rule_ids": unseen_rule_ids,
            "seen_function_ids": seen_function_ids,
            "unseen_function_ids": unseen_function_ids,
            "function_groups": [
                {
                    "function_id": function_id,
                    "representative_rule_id": function_representatives[function_id].rule_id,
                    "outputs": list(function_representatives[function_id].outputs),
                    "alias_rule_ids": [rule.rule_id for rule in aliases],
                    "alias_families": [rule.family for rule in aliases],
                    "alias_params": [list(rule.params) for rule in aliases],
                }
                for function_id, aliases in sorted(function_groups.items())
            ],
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "identity_id": rule.rule_id,
                    "function_id": rule.function_id,
                    "family": rule.family,
                    "family_id": rule.family_id,
                    "params": list(rule.params),
                    "outputs": list(rule.outputs),
                }
                for rule in rules
            ],
        },
    )


def get_task_loaders(task: str, **kwargs: object) -> DatasetBundle:
    """Dispatch task loader creation."""

    if task == "context_permutation":
        return get_context_permutation_loaders(**kwargs)
    if task == "symbolic_rule_mapping":
        return get_symbolic_rule_mapping_loaders(**kwargs)
    raise ValueError(f"Unsupported task: {task}")
