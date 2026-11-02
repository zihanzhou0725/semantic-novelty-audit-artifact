#!/usr/bin/env python3
"""Freeze the result-independent P3 candidate manifest before semantic audit."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quality_p1_common import EXTERNAL_ROOT, PROTOCOL_DIR, ROOT, sha256_file, write_json


SCHEMA_VERSION = "quality-enhancement-p3-candidates-v1"
P3_DIR = PROTOCOL_DIR / "p3"
DISCOVERY_DIR = P3_DIR / "discovery"
QUEUE_PATH = DISCOVERY_DIR / "deduplicated_screening_queue.csv"
DISCOVERY_MANIFEST_PATH = DISCOVERY_DIR / "discovery_manifest.json"

EXPECTED_QUEUE = (
    (
        "d50fe7355b7725a3",
        "Design of Adaptive E-Learning for Logic Operations",
    ),
    (
        "1efe72eea3cbbcfc",
        "How Do Position Encodings Affect Length Generalization? "
        "Case Studies On In-Context Function Learning",
    ),
    (
        "7b1fa5fbdb1da4ee",
        "Implementing Tensor Logic: Unifying Datalog and Neural Reasoning "
        "via Tensor Contraction",
    ),
    (
        "7b9ebd45a3528701",
        "In-Context Occam's Razor: How Transformers Prefer Simpler "
        "Hypotheses on the Fly",
    ),
    (
        "26452d6d102c8113",
        "SignReasoner: Compositional Reasoning for Complex Traffic Sign "
        "Understanding via Functional Structure Units",
    ),
    (
        "03cbb3deb603e909",
        "Transfer Learning Beyond Bounded Density Ratios",
    ),
    (
        "6b743d89805a72e3",
        "Understanding In-Context Learning in Transformers and LLMs by "
        "Learning to Learn Discrete Functions",
    ),
)


def fail(message: str) -> None:
    raise SystemExit(f"P3 candidate freeze failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = tuple((row["candidate_id"], row["title"]) for row in rows)
    require(observed == EXPECTED_QUEUE, "frozen screening queue changed")
    normalized = [row["normalized_title"] for row in rows]
    require(normalized == sorted(normalized), "screening queue is not title-sorted")
    return rows


def screening_decisions() -> dict[str, dict[str, Any]]:
    return {
        "d50fe7355b7725a3": {
            "behavioral_artifact_availability": "not_applicable",
            "code_revision": None,
            "dataset_revision": None,
            "decision": "excluded",
            "eligibility": {
                "generator_and_split_inspectable": False,
                "paper_and_evaluation_data_public": False,
                "semantic_claim_not_only_lexical": False,
                "sound_signature_or_executable_witness_available": False,
            },
            "evidence_urls": [],
            "paper_revision": "publication_metadata_only_2010",
            "reason": (
                "The title and available metadata describe adaptive e-learning "
                "for logic operations, not an evaluation benchmark making a "
                "semantic-novelty or compositional-generalization claim."
            ),
        },
        "1efe72eea3cbbcfc": {
            "behavioral_artifact_availability": (
                "public training and evaluation code; no frozen per-item "
                "prediction artifact identified"
            ),
            "code_revision": {
                "repository": "https://github.com/IKMLab/length-generalization-with-icl",
                "revision": "dc9146f671d623b304f98fec031e4a8e14a9b044",
            },
            "dataset_revision": (
                "generated online by the pinned code revision; no fixed dataset"
            ),
            "decision": "excluded",
            "eligibility": {
                "generator_and_split_inspectable": True,
                "paper_and_evaluation_data_public": True,
                "semantic_claim_not_only_lexical": True,
                "sound_signature_or_executable_witness_available": True,
            },
            "evidence_urls": [
                "https://doi.org/10.1609/aaai.v39i23.34637",
                "https://github.com/IKMLab/length-generalization-with-icl",
            ],
            "paper_revision": "AAAI-25 DOI 10.1609/aaai.v39i23.34637",
            "reason": (
                "Excluded by the frozen historical-case rule: the public "
                "repository states that it is based on dtsip/in-context-learning "
                "and Bhattamishra et al. supplementary material and reuses the "
                "same Boolean function families, so it is a generator-family "
                "variant rather than an independent public replication."
            ),
        },
        "7b1fa5fbdb1da4ee": {
            "behavioral_artifact_availability": (
                "public executable experiment code and aggregate reported "
                "metrics; no frozen per-item prediction artifact identified"
            ),
            "code_revision": {
                "repository": "https://github.com/sshah100-clt/tensor_logic_implementation",
                "revision": "9b497c27992843bfbfdd99801dab6881cebb465e",
            },
            "dataset_revision": {
                "base_dataset": "FB15k-237",
                "source_repository": (
                    "https://github.com/DeepGraphLearning/"
                    "KnowledgeGraphEmbedding"
                ),
                "source_revision": "2e440e0f9c687314d5ff67ead68ce985dc446e3a",
                "split_revision": (
                    "direct-edge-removal compositional split generated by "
                    "code revision 9b497c27992843bfbfdd99801dab6881cebb465e"
                ),
            },
            "decision": "included",
            "eligibility": {
                "generator_and_split_inspectable": True,
                "paper_and_evaluation_data_public": True,
                "semantic_claim_not_only_lexical": True,
                "sound_signature_or_executable_witness_available": True,
            },
            "evidence_urls": [
                "https://arxiv.org/abs/2601.17188v1",
                "https://github.com/sshah100-clt/tensor_logic_implementation",
                "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding",
            ],
            "paper_revision": "arXiv:2601.17188v1",
            "reason": (
                "Included as the first eligible title-sorted independent "
                "benchmark: public code and FB15k-237 data expose a held-out "
                "direct-edge compositional split with exact relation paths."
            ),
        },
        "7b9ebd45a3528701": {
            "behavioral_artifact_availability": (
                "public generator, training, and evaluation code plus paper "
                "aggregate results; no frozen per-item predictions identified"
            ),
            "code_revision": {
                "repository": "https://github.com/puneesh00/ICL-Bayes-Occam",
                "revision": "3b8fe0d69dd9bb56c39d933525bf0234feefcc39",
            },
            "dataset_revision": (
                "Markov-chain, linear-regression, and grammar episodes generated "
                "online by code revision "
                "3b8fe0d69dd9bb56c39d933525bf0234feefcc39"
            ),
            "decision": "included",
            "eligibility": {
                "generator_and_split_inspectable": True,
                "paper_and_evaluation_data_public": True,
                "semantic_claim_not_only_lexical": True,
                "sound_signature_or_executable_witness_available": True,
            },
            "evidence_urls": [
                "https://arxiv.org/abs/2506.19351v1",
                "https://openreview.net/forum?id=ZSMnX3LBva",
                "https://github.com/puneesh00/ICL-Bayes-Occam",
            ],
            "paper_revision": "arXiv:2506.19351v1; COLM 2025",
            "reason": (
                "Included as the second eligible title-sorted independent "
                "testbed: public code defines hierarchical Markov and regression "
                "task generators with exact latent task parameters."
            ),
        },
        "26452d6d102c8113": {
            "behavioral_artifact_availability": (
                "aggregate paper results only; no revision-pinnable public "
                "TrafficSignEval artifact located at screening time"
            ),
            "code_revision": None,
            "dataset_revision": None,
            "decision": "excluded",
            "eligibility": {
                "generator_and_split_inspectable": False,
                "paper_and_evaluation_data_public": False,
                "semantic_claim_not_only_lexical": True,
                "sound_signature_or_executable_witness_available": True,
            },
            "evidence_urls": [
                "https://arxiv.org/abs/2604.10436v1",
            ],
            "paper_revision": "arXiv:2604.10436v1",
            "reason": (
                "The paper describes releasing TrafficSignEval and evaluation "
                "code, but no public repository or dataset URL with a pinnable "
                "revision was present in the paper or discoverable by the frozen "
                "screening date; public-data eligibility therefore fails."
            ),
        },
        "03cbb3deb603e909": {
            "behavioral_artifact_availability": "not_applicable",
            "code_revision": None,
            "dataset_revision": None,
            "decision": "excluded",
            "eligibility": {
                "generator_and_split_inspectable": False,
                "paper_and_evaluation_data_public": True,
                "semantic_claim_not_only_lexical": False,
                "sound_signature_or_executable_witness_available": False,
            },
            "evidence_urls": [
                "https://arxiv.org/abs/2403.11963",
            ],
            "paper_revision": "arXiv:2403.11963",
            "reason": (
                "The record is a transfer-learning theory paper, not a public "
                "semantic-rule/function benchmark with a train/evaluation "
                "novelty claim."
            ),
        },
        "6b743d89805a72e3": {
            "behavioral_artifact_availability": (
                "public code and historical Boolean behavioral artifacts"
            ),
            "code_revision": None,
            "dataset_revision": None,
            "decision": "excluded",
            "eligibility": {
                "generator_and_split_inspectable": True,
                "paper_and_evaluation_data_public": True,
                "semantic_claim_not_only_lexical": True,
                "sound_signature_or_executable_witness_available": True,
            },
            "evidence_urls": [
                "https://arxiv.org/abs/2310.03016v1",
            ],
            "paper_revision": "arXiv:2310.03016v1",
            "reason": (
                "Explicitly excluded historical case: satwik77/incontext-bool "
                "was already audited under the Boolean behavioral case and "
                "cannot count as prospective P3 breadth."
            ),
        },
    }


def selected_candidates(
    queue_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in queue_rows:
        decision = decisions[row["candidate_id"]]
        if decision["decision"] != "included":
            continue
        if row["candidate_id"] == "7b1fa5fbdb1da4ee":
            audit = {
                "audited_claim": (
                    "The FB15k-237 direct-edge-removal evaluation measures "
                    "compositional inference on relation paths not directly "
                    "observed as edges during training."
                ),
                "generator_and_split_definition": (
                    "Use the pinned FB15k-237 source and the pinned experiment "
                    "code's compositional benchmark that removes direct edges "
                    "while retaining multi-hop relation paths."
                ),
                "semantic_signature_or_witness": (
                    "Canonical ordered relation-path signature plus endpoint "
                    "entity pair; an executable graph-path witness must derive "
                    "the held-out direct relation from retained training edges."
                ),
                "scope": (
                    "Experiment 3B compositional reasoning benchmark only; "
                    "standard link prediction and unrelated genealogy/country "
                    "experiments are outside the candidate audit."
                ),
            }
        elif row["candidate_id"] == "7b9ebd45a3528701":
            audit = {
                "audited_claim": (
                    "Transformers identify and infer held-out tasks across "
                    "hierarchical process-complexity classes rather than relying "
                    "only on surface sequence identity."
                ),
                "generator_and_split_definition": (
                    "Use the pinned code's online Markov-chain and linear-"
                    "regression generators and their frozen train/evaluation "
                    "complexity configurations; PCFG and pretrained-LLM case "
                    "studies are secondary and cannot replace this scope."
                ),
                "semantic_signature_or_witness": (
                    "Canonical latent task parameters: transition tensor/order "
                    "for Markov tasks and coefficient vector/subspace definition "
                    "for regression tasks, with an executable likelihood/output "
                    "witness for candidate collisions."
                ),
                "scope": (
                    "Synthetic Markov-chain and linear-regression hierarchical "
                    "ICL testbeds; no claim about all tasks or pretrained LLMs."
                ),
            }
        else:
            fail(f"unexpected selected candidate: {row['candidate_id']}")
        selected.append(
            {
                "audit_order": len(selected) + 1,
                "behavioral_artifact_availability": decision[
                    "behavioral_artifact_availability"
                ],
                "candidate_id": row["candidate_id"],
                "checkpoint_revision_if_any": None,
                "code_revision": decision["code_revision"],
                "dataset_revision": decision["dataset_revision"],
                "discovery_query_hits": json.loads(row["query_hits_json"]),
                "paper_revision": decision["paper_revision"],
                "result_independent_inclusion_or_exclusion_reason": decision[
                    "reason"
                ],
                "search_date": "2026-10-09",
                "search_strings": [
                    '"rule inference" benchmark unseen',
                    '"in-context learning" Boolean functions',
                    '"symbolic reasoning" benchmark held-out rules',
                    '"compositional generalization" unseen functions benchmark',
                ],
                "title": row["title"],
                **audit,
            }
        )
    return selected


def main() -> int:
    discovery_checksums = DISCOVERY_DIR / "SHA256SUMS"
    require(discovery_checksums.is_file(), "discovery is not frozen")
    for line in discovery_checksums.read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        path = DISCOVERY_DIR / filename
        require(path.is_file(), f"missing discovery artifact: {filename}")
        require(
            sha256_file(path) == digest,
            f"discovery artifact changed: {filename}",
        )
    discovery_manifest = json.loads(
        DISCOVERY_MANIFEST_PATH.read_text(encoding="ascii")
    )
    require(
        discovery_manifest["status"]
        == "raw_discovery_frozen_before_candidate_semantic_inspection",
        "discovery manifest status",
    )
    require(
        discovery_manifest["candidate_dataset_or_code_download_started"]
        is False
        and discovery_manifest["candidate_semantic_inspection_started"] is False,
        "candidate inspection started before selection",
    )

    preregistration = json.loads(
        (PROTOCOL_DIR / "preregistration.json").read_text(encoding="ascii")
    )
    p3 = preregistration["p3"]
    queue_rows = read_queue()
    decisions = screening_decisions()
    require(
        set(decisions) == {row["candidate_id"] for row in queue_rows},
        "screening decisions do not cover the complete queue",
    )
    selected = selected_candidates(queue_rows, decisions)
    require(
        p3["candidate_count"]["min"]
        <= len(selected)
        <= p3["candidate_count"]["max"],
        "selected candidate count violates P3 bounds",
    )
    require(
        [row["candidate_id"] for row in selected]
        == ["7b1fa5fbdb1da4ee", "7b9ebd45a3528701"],
        "selected candidates changed",
    )

    screening_rows: list[dict[str, Any]] = []
    for order, row in enumerate(queue_rows, start=1):
        decision = decisions[row["candidate_id"]]
        eligibility = decision["eligibility"]
        screening_rows.append(
            {
                "screening_order": order,
                "candidate_id": row["candidate_id"],
                "normalized_title": row["normalized_title"],
                "title": row["title"],
                "decision": decision["decision"],
                "paper_and_evaluation_data_public": eligibility[
                    "paper_and_evaluation_data_public"
                ],
                "semantic_claim_not_only_lexical": eligibility[
                    "semantic_claim_not_only_lexical"
                ],
                "generator_and_split_inspectable": eligibility[
                    "generator_and_split_inspectable"
                ],
                "sound_signature_or_executable_witness_available": eligibility[
                    "sound_signature_or_executable_witness_available"
                ],
                "reason": decision["reason"],
                "evidence_urls_json": json.dumps(
                    decision["evidence_urls"],
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
            }
        )

    manifest = {
        "candidate_count": len(selected),
        "candidate_dataset_download_started": False,
        "candidate_semantic_extraction_started": False,
        "candidates": selected,
        "discovery_checksum_manifest_sha256": sha256_file(
            discovery_checksums
        ),
        "discovery_manifest_sha256": sha256_file(DISCOVERY_MANIFEST_PATH),
        "freeze_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_cases_excluded": p3["historical_cases_excluded"],
        "result_dependent_selection_used": False,
        "schema_version": SCHEMA_VERSION,
        "screened_record_count": len(screening_rows),
        "selection_rule": p3["discovery"]["include_rule"],
        "status": "frozen_before_candidate_dataset_download_or_semantic_extraction",
    }
    manifest_path = P3_DIR / "candidate_manifest.json"
    screening_path = P3_DIR / "candidate_screening.csv"
    checksum_path = P3_DIR / "SHA256SUMS"
    for path in (manifest_path, screening_path, checksum_path):
        require(not path.exists(), f"P3 candidate artifact exists: {path}")
    P3_DIR.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    with screening_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(screening_rows[0]))
        writer.writeheader()
        writer.writerows(screening_rows)
    checksum_path.write_text(
        "\n".join(
            (
                f"{sha256_file(manifest_path)}  candidate_manifest.json",
                f"{sha256_file(screening_path)}  candidate_screening.csv",
            )
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "candidate_count": len(selected),
                "candidate_manifest_sha256": sha256_file(manifest_path),
                "selected_titles": [row["title"] for row in selected],
                "status": manifest["status"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
