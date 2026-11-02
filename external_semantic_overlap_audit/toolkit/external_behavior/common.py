"""Shared deterministic utilities for external-behavior-v1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL_ID = "external-behavior-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = PROJECT_ROOT / "protocols" / PROTOCOL_ID
MANIFEST_DIR = PROJECT_ROOT / "manifests" / PROTOCOL_ID
DATA_DIR = MANIFEST_DIR / "data"
UPSTREAM_DIR = PROJECT_ROOT / "upstream"
CLUTRR_ROOT = UPSTREAM_DIR / "external_behavior_v1_clutrr"
MATHEMATICS_ROOT = UPSTREAM_DIR / "external_behavior_v1_mathematics"
MODEL_SEEDS = tuple(range(8))
ARMS = ("A", "B")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(namespace: str) -> int:
    payload = f"{PROTOCOL_ID}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def arm_assignment_bit(model_seed: int, pair_index: int) -> int:
    return derive_seed(f"clutrr:arm:{model_seed}:{pair_index}") & 1


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="ascii") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def write_lines(path: Path, values: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="ascii") as handle:
        for value in values:
            handle.write(value)
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def require_pythonhashseed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "external-behavior-v1 materialization requires PYTHONHASHSEED=0"
        )


def verify_source_manifest() -> dict[str, Any]:
    manifest = read_json(PROTOCOL_DIR / "source_manifest.json")
    locations = {
        "clutrr": CLUTRR_ROOT,
        "mathematics": MATHEMATICS_ROOT,
    }
    checks: list[dict[str, Any]] = []
    for source_name, source in manifest["sources"].items():
        root = locations[source_name]
        revision = git_output(root, "rev-parse", "HEAD")
        status = git_output(root, "status", "--porcelain=v1")
        checks.append(
            {
                "check": f"{source_name}_revision",
                "expected": source["revision"],
                "observed": revision,
                "pass": revision == source["revision"],
            }
        )
        checks.append(
            {
                "check": f"{source_name}_clean",
                "observed": status,
                "pass": status == "",
            }
        )
        for relative, expected in source["files"].items():
            path = root / relative
            observed = sha256_file(path) if path.is_file() else None
            checks.append(
                {
                    "check": f"{source_name}:{relative}",
                    "expected": expected,
                    "observed": observed,
                    "pass": observed == expected,
                }
            )
    failed = [item for item in checks if not item["pass"]]
    if failed:
        raise RuntimeError(f"Pinned source validation failed: {failed}")
    return {
        "checks": checks,
        "source_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "source_manifest.json"
        ),
        "status": "pass",
    }


def checksums(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {
        path.resolve().relative_to(root.resolve()).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def write_checksum_manifest(directory: Path, paths: Iterable[Path]) -> Path:
    selected = sorted(path for path in paths if path.name != "SHA256SUMS")
    output = directory / "SHA256SUMS"
    output.write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.resolve().relative_to(directory.resolve()).as_posix()}\n"
            for path in selected
        ),
        encoding="ascii",
    )
    return output
