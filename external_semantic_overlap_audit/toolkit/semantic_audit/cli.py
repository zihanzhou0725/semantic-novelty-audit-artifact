"""Command-line interface for semantic-audit toolkit v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS, build_adapter_output
from .core import TOOL_VERSION, sha256_file, write_json
from .schema import DEFAULT_SCHEMA_PATH, validate_audit_output


DEFAULT_EXTERNAL_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILENAMES = {
    "occam-markov": "occam_markov.audit.json",
    "tensor-logic": "tensor_logic.audit.json",
}


def _summary(output: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "adapter_id": output["adapter"]["id"],
        "completeness_status": output["completeness_status"],
        "output_file": path.name,
        "output_sha256": sha256_file(path),
        "overlap_counts_and_denominators": output[
            "overlap_counts_and_denominators"
        ],
    }


def reproduce_outputs(
    output_dir: Path,
    *,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    for adapter_id in sorted(ADAPTERS):
        output = build_adapter_output(adapter_id, external_root)
        validate_audit_output(output)
        output_path = output_dir / OUTPUT_FILENAMES[adapter_id]
        write_json(output_path, output)
        output_paths.append(output_path)
        summaries.append(_summary(output, output_path))

    package_dir = Path(__file__).resolve().parent
    implementation_files = [
        external_root / "scripts" / "semantic_audit_v1.py",
        external_root / "scripts" / "validate_quality_p4_toolkit.py",
        external_root / "tests" / "test_p4_toolkit.py",
        external_root / "toolkit" / "README.md",
        package_dir / "__init__.py",
        package_dir / "adapters.py",
        package_dir / "bounds.py",
        package_dir / "cli.py",
        package_dir / "core.py",
        package_dir / "schema.py",
        DEFAULT_SCHEMA_PATH,
    ]
    manifest = {
        "adapters": summaries,
        "implementation_sha256": {
            str(path.relative_to(external_root)): sha256_file(path)
            for path in sorted(implementation_files)
        },
        "schema_file": str(DEFAULT_SCHEMA_PATH.relative_to(external_root)),
        "schema_sha256": sha256_file(DEFAULT_SCHEMA_PATH),
        "schema_version": "semantic-audit-reproduction-manifest-v1",
        "tool_version": TOOL_VERSION,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    output_paths.append(manifest_path)

    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(output_paths)
        )
        + "\n",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-audit-v1",
        description="Run validated semantic-audit adapters.",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT,
        help="external_semantic_overlap_audit root",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {TOOL_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Materialize all public adapters and a checksum manifest.",
    )
    reproduce.add_argument("--output-dir", type=Path, required=True)

    adapter = subparsers.add_parser(
        "adapter",
        help="Run and validate one public adapter.",
    )
    adapter.add_argument("adapter_id", choices=sorted(ADAPTERS))
    adapter.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate one or more common-schema outputs.",
    )
    validate.add_argument("outputs", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reproduce":
        manifest = reproduce_outputs(
            args.output_dir.resolve(),
            external_root=args.external_root.resolve(),
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0

    if args.command == "adapter":
        output = build_adapter_output(
            args.adapter_id,
            args.external_root.resolve(),
        )
        validate_audit_output(output)
        write_json(args.output.resolve(), output)
        print(
            json.dumps(
                {
                    "adapter_id": output["adapter"]["id"],
                    "output": str(args.output.resolve()),
                    "status": "validated",
                },
                sort_keys=True,
            )
        )
        return 0

    validated: list[str] = []
    for path in args.outputs:
        value = json.loads(path.read_text(encoding="ascii"))
        validate_audit_output(value)
        validated.append(str(path.resolve()))
    print(json.dumps({"status": "validated", "outputs": validated}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
