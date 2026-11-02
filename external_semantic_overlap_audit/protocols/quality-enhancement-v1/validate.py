#!/usr/bin/env python3
"""Validate the frozen quality-enhancement protocol without running analyses."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "quality-enhancement-v1"
EXPECTED_FROZEN_FILES = {
    "README.md",
    "preregistration.json",
    "validate.py",
}


def fail(message: str) -> None:
    raise SystemExit(f"Validation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    require(isinstance(value, dict), f"{path.name} is not a JSON object")
    return value


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(),
        start=1,
    ):
        parts = line.split("  ", maxsplit=1)
        require(len(parts) == 2, f"invalid checksum line {line_number}")
        digest, filename = parts
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"invalid SHA-256 on checksum line {line_number}",
        )
        require(filename not in checksums, f"duplicate checksum entry: {filename}")
        checksums[filename] = digest
    return checksums


def validate_frozen_checksums(protocol_dir: Path) -> None:
    checksums = parse_checksums(protocol_dir / "SHA256SUMS")
    require(
        set(checksums) == EXPECTED_FROZEN_FILES,
        f"checksum file set is {set(checksums)}, expected {EXPECTED_FROZEN_FILES}",
    )
    for filename, expected in checksums.items():
        observed = sha256_file(protocol_dir / filename)
        require(
            observed == expected,
            f"frozen checksum mismatch for {filename}: {observed} != {expected}",
        )


def validate_source_pins(root: Path, protocol: dict[str, Any]) -> None:
    pins = protocol["source_pins"]
    require(len(pins) == 14, "expected exactly 14 source pins")
    paths = [pin["path"] for pin in pins]
    require(len(paths) == len(set(paths)), "source pin paths are not unique")
    for pin in pins:
        path = root / pin["path"]
        require(path.is_file(), f"missing pinned source: {pin['path']}")
        observed = sha256_file(path)
        require(
            observed == pin["sha256"],
            f"source pin mismatch for {pin['path']}: {observed} != {pin['sha256']}",
        )


def validate_top_level(protocol: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "freeze_date",
        "status",
        "execution_state_at_freeze",
        "registration_status",
        "prior_known_evidence",
        "global_rules",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "submission_gates",
        "stopping_and_order",
        "claim_boundary",
        "source_pins",
        "amendment_policy",
    }
    require(required.issubset(protocol), "missing required top-level fields")
    require(protocol["schema_version"] == SCHEMA_VERSION, "schema version mismatch")
    require(protocol["freeze_date"] == "2026-10-09", "freeze date mismatch")
    require(
        protocol["status"] == "frozen_before_new_confirmatory_execution",
        "protocol is not marked frozen before execution",
    )
    execution = protocol["execution_state_at_freeze"]
    require(
        execution
        == {
            "p1_complementary_training_started": False,
            "p1_manifests_materialized": False,
            "p2_registered_model_fit_run": False,
            "p3_candidate_result_inspection_started": False,
        },
        "execution state is not the frozen pre-execution state",
    )


def validate_global_thresholds(protocol: dict[str, Any]) -> None:
    rules = protocol["global_rules"]
    require(rules["sesoi_pp"] == 1.0, "SESOI must be 1.0 pp")
    require(
        rules["equivalence_margin_pp"] == [-0.5, 0.5],
        "equivalence margin must be [-0.5, 0.5] pp",
    )
    require(rules["report_all_registered_results"], "report-all rule is disabled")
    require(
        rules["primary_endpoints"]
        == [
            "p1.architecture_macro_exact_exposure_effect",
            "p2.adjusted_marginal_overlap_association",
        ],
        "primary endpoint family changed",
    )

    p1_gates = protocol["p1"]["gates"]
    require(
        p1_gates["positive_behavioral_go"]["macro_point_estimate_at_least_pp"]
        == rules["sesoi_pp"],
        "P1 positive threshold differs from global SESOI",
    )
    require(
        p1_gates["negative_behavioral_go"]["macro_point_estimate_at_most_pp"]
        == -rules["sesoi_pp"],
        "P1 negative threshold differs from global SESOI",
    )
    require(
        p1_gates["null_equivalence_go"]["macro_ci_contained_in_pp"]
        == rules["equivalence_margin_pp"],
        "P1 null margin differs from global margin",
    )

    p2_gates = protocol["p2"]["gates"]
    require(
        p2_gates["positive_association_go"]["point_estimate_at_least_pp"]
        == rules["sesoi_pp"],
        "P2 positive threshold differs from global SESOI",
    )
    require(
        p2_gates["negative_association_go"]["point_estimate_at_most_pp"]
        == -rules["sesoi_pp"],
        "P2 negative threshold differs from global SESOI",
    )
    require(
        p2_gates["null_association_go"]["ci_contained_in_pp"]
        == rules["equivalence_margin_pp"],
        "P2 null margin differs from global margin",
    )


def validate_p1_arithmetic(protocol: dict[str, Any]) -> None:
    p1 = protocol["p1"]
    census = p1["function_census"]
    require(census["identity_count"] == 560, "P1 identity count changed")
    require(census["unique_function_count"] == 518, "P1 function count changed")
    require(census["target_count"] == 40, "P1 target count changed")
    require(
        census["alias_signature_counts"]
        == {"affine+block+parity": 2, "affine+parity": 38},
        "P1 alias signature counts changed",
    )
    require(
        census["target_affine_multipliers"] == [1, 11],
        "P1 target multipliers changed",
    )

    stream = p1["stream"]
    require(
        stream["seed_derivation"]
        == "int.from_bytes(SHA256(('quality-enhancement-v1:20261009:' + namespace).encode('ascii')).digest()[0:8], 'big') & ((1<<63)-1)",
        "P1 seed derivation changed",
    )
    require(
        stream["arm_assignment"]
        == {
            "bit": "int.from_bytes(SHA256(payload_ascii).digest()[0:8],'big') & 1",
            "bit_0": "arm_A=f_(1,b),arm_B=f_(11,b)",
            "bit_1": "arm_A=f_(11,b),arm_B=f_(1,b)",
            "payload": "quality-enhancement-v1:20261009:p1-arm:{model_seed}:{intercept}",
        },
        "P1 arm assignment changed",
    )
    require(
        stream["extra_base_selection"]
        == {
            "count": 80,
            "input_order": "ascending_numeric_function_id",
            "namespace": "p1-extra:{model_seed}",
            "rule": "first_80_after_numpy_PCG64_permutation",
        },
        "P1 extra-base selection changed",
    )
    require(
        stream["slot_order"]["namespace"] == "p1-slots:{model_seed}"
        and stream["epoch_order"]["namespace"]
        == "p1-epoch:{model_seed}:{epoch_1_based}"
        and not stream["epoch_order"]["dataloader_shuffle"],
        "P1 slot or epoch order changed",
    )
    base_slots = (
        stream["base_function_count"] * stream["base_repetitions_per_function"]
        + stream["base_extra_episode_functions"]
    )
    target_slots = (
        stream["exposed_target_count_per_arm"]
        * stream["target_repetitions_per_exposed_function"]
    )
    require(
        base_slots + target_slots == stream["episodes_per_arm"] == 20000,
        "P1 stream arithmetic does not sum to 20,000",
    )
    require(
        stream["base_function_count"]
        + stream["exposed_target_count_per_arm"]
        == stream["training_function_count_per_arm"]
        == 498,
        "P1 per-arm function count is inconsistent",
    )

    models = p1["models"]
    expected_runs = (
        models["count"] * len(models["model_seeds"]) * len(stream["arms"])
    )
    require(expected_runs == models["run_count"] == 60, "P1 run count is not 60")
    require(
        len(models["architectures"]) == models["count"] == 6,
        "P1 architecture count is inconsistent",
    )

    evaluation = p1["evaluation"]
    require(
        evaluation["target_functions"] * evaluation["episodes_per_function"]
        == evaluation["n_points_total"]
        == 10000,
        "P1 evaluation count is inconsistent",
    )
    require(
        evaluation["common_across_all_arms_models_seeds"]
        and evaluation["same_tensors_between_arms"],
        "P1 evaluation manifest is not globally paired",
    )
    optimization = p1["optimization"]
    updates = optimization["epochs"] * math.ceil(
        stream["episodes_per_arm"] / optimization["batch_size"]
    )
    require(
        updates == optimization["optimizer_updates"] == 4740,
        "P1 optimizer update count is inconsistent",
    )
    require(
        optimization["checkpoint_selection"] == "fixed_final_epoch_60"
        and not optimization["validation_selection"],
        "P1 checkpoint selection is not fixed final epoch",
    )

    bootstrap = p1["analysis"]["bootstrap"]
    require(bootstrap["replicates"] == 20000, "P1 bootstrap count changed")
    require(
        bootstrap["model_seed_clusters"] == len(models["model_seeds"]),
        "P1 bootstrap seed clusters are inconsistent",
    )
    require(
        bootstrap["function_pair_clusters"] * 2 == census["target_count"],
        "P1 bootstrap pair clusters are inconsistent",
    )
    expected_confidence = 1.0 - 0.05 / models["count"]
    observed_confidence = p1["analysis"]["model_specific_interval"]["confidence"]
    require(
        abs(observed_confidence - expected_confidence) < 1e-12,
        "P1 simultaneous interval confidence is inconsistent",
    )


def validate_generator_census(root: Path, protocol: dict[str, Any]) -> None:
    package_root = (
        root
        / "semantic_novelty_audit_supplement"
        / "src"
        / "kfwa_experiments"
    )
    sys.path.insert(0, str(package_root))
    try:
        from kfwa.datasets import (  # type: ignore[import-not-found]
            build_symbolic_rules,
            group_symbolic_rules_by_function,
            symbolic_function_representatives,
        )
    except Exception as error:
        fail(f"could not import pinned symbolic generator: {error}")

    rules = build_symbolic_rules(
        num_symbols=20,
        families="affine,block,parity",
    )
    groups = group_symbolic_rules_by_function(rules)
    representatives = symbolic_function_representatives(rules)
    targets: list[tuple[int, int, tuple[str, ...]]] = []
    for function_id, aliases in groups.items():
        signature = tuple(sorted({rule.family for rule in aliases}))
        if len(signature) <= 1:
            continue
        representative = representatives[function_id]
        require(
            representative.family == "affine",
            f"target {function_id} is not canonically affine",
        )
        require(
            any(rule.family == "parity" for rule in aliases),
            f"target {function_id} has no parity alias",
        )
        multiplier, intercept = representative.params
        targets.append((multiplier, intercept, signature))

    require(len(rules) == 560, "generator does not produce 560 identities")
    require(len(groups) == 518, "generator does not produce 518 functions")
    require(len(targets) == 40, "generator does not produce 40 targets")
    require(
        Counter(signature for _, _, signature in targets)
        == Counter(
            {
                ("affine", "parity"): 38,
                ("affine", "block", "parity"): 2,
            }
        ),
        "generator alias signature census changed",
    )
    require(
        {(multiplier, intercept) for multiplier, intercept, _ in targets}
        == {
            (multiplier, intercept)
            for multiplier in (1, 11)
            for intercept in range(20)
        },
        "generator target affine parameter set changed",
    )
    for intercept in range(20):
        members = [
            multiplier
            for multiplier, candidate_intercept, _ in targets
            if candidate_intercept == intercept
        ]
        require(
            sorted(members) == [1, 11],
            f"pair for intercept {intercept} is incomplete",
        )

    census = protocol["p1"]["function_census"]
    require(census["identity_count"] == len(rules), "JSON/generator identity mismatch")
    require(
        census["unique_function_count"] == len(groups),
        "JSON/generator function mismatch",
    )
    require(census["target_count"] == len(targets), "JSON/generator target mismatch")


def validate_p2(protocol: dict[str, Any]) -> None:
    p2 = protocol["p2"]
    exposure = p2["exposure"]
    require(
        exposure["token_identical_rows"]
        + exposure["additional_sound_flat_alias_rows"]
        == exposure["witnessed_overlap_rows"]
        == 452,
        "P2 overlap union arithmetic changed",
    )
    require(
        exposure["witnessed_overlap_rows"] + exposure["comparator_rows"]
        == exposure["test_rows"]
        == 1000,
        "P2 test partition arithmetic changed",
    )
    common = p2["common_support"]
    require(
        common["propensity_bounds_inclusive"] == [0.05, 0.95],
        "P2 propensity bounds changed",
    )
    require(
        common["minimum_retained_fraction_per_group"] == 0.8
        and common["minimum_group_rows_after_trim"] == 100,
        "P2 common-support minimum changed",
    )
    require(
        p2["reproducibility_precondition"][
            "current_pinned_repository_output_eligible_for_paper_model_claim"
        ]
        is False,
        "P2 current-output claim boundary was relaxed",
    )
    require(
        p2["known_status_at_freeze"]["published_main_table_reproduced_by_pinned_output"]
        is False,
        "P2 known artifact mismatch was removed",
    )
    require(
        p2["known_status_at_freeze"]["identification_decision"]
        == "no_go_before_outcome_fit"
        and not p2["known_status_at_freeze"][
            "common_support_precheck_passes_80_percent_each_group"
        ],
        "P2 known common-support No-Go changed",
    )
    analysis = p2["analysis"]
    require(
        analysis["outcome_model"]["family"]
        == "statsmodels_GLM_Binomial_logit_unregularized_with_intercept"
        and analysis["outcome_model"]["spline_implementation"]
        == "patsy.cr(x,knots=(q25,q50,q75),lower_bound=min,upper_bound=max)"
        and analysis["outcome_model"]["fit"]
        == {
            "algorithm": "IRLS",
            "fallback_fit_allowed": False,
            "max_iterations": 100,
            "tolerance": 1e-08,
        },
        "P2 outcome-model implementation changed",
    )
    require(
        analysis["bootstrap"]["replicates"] == 10000
        and analysis["bootstrap"]["attempt_cap"] == 20000
        and analysis["bootstrap"]["resampling_strata"]
        == ["witnessed_overlap", "curriculum_id"],
        "P2 bootstrap implementation changed",
    )
    source = p2["source_selection"]
    require(
        source["confirmatory_model"] == "Llama-3.1-8B-IT-SLR"
        and source["confirmatory_result"] == "ACL 2026 final paper Table 2"
        and not source["different_model_substitution_allowed"],
        "P2 confirmatory source selection changed",
    )


def contains_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    width = len(sequence)
    return any(
        tuple(tokens[index : index + width]) == sequence
        for index in range(len(tokens) - width + 1)
    )


def validate_p2_covariates(root: Path, protocol: dict[str, Any]) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        fail(f"pyarrow is required to validate P2 covariates: {error}")

    parser_path = (
        root / "external_semantic_overlap_audit" / "scripts" / "audit_slr.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_audit_slr", parser_path)
    require(spec is not None and spec.loader is not None, "cannot load audit_slr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parquet_path = (
        root
        / "external_semantic_overlap_audit"
        / "upstream"
        / "slr_dataset"
        / "v1-All"
        / "test-00000-of-00001.parquet"
    )
    rows = pq.read_table(
        parquet_path,
        columns=[
            "ground-truth rule",
            "curriculum level",
            "rule sampling",
        ],
    ).to_pylist()
    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    structure_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    atom_counts: list[int] = []

    for row in rows:
        rule = row["ground-truth rule"]
        tokens_value = module.token_key(rule)
        require(tokens_value is not None, "P2 test rule tokenization failed")
        tokens = list(tokens_value)
        body_markers = [
            index
            for index in range(len(tokens) - 1)
            if tokens[index : index + 2] == [":", "-"]
        ]
        require(len(body_markers) == 1, "P2 test rule has no unique body marker")
        body = tokens[body_markers[0] + 2 :]
        callable_names = {
            body[index]
            for index in range(len(body) - 1)
            if identifier.fullmatch(body[index]) and body[index + 1] == "("
        }
        atom_count = sum(
            bool(identifier.fullmatch(body[index]))
            and index + 1 < len(body)
            and body[index + 1] == "("
            for index in range(len(body))
        )

        if callable_names & {"findall", "bagof", "setof", "forall"}:
            structure = "aggregate_or_quantified"
        elif "not" in callable_names or contains_sequence(body, ("\\", "+")):
            structure = "negated"
        elif ";" in body:
            structure = "disjunctive"
        elif any(
            contains_sequence(body, operator)
            for operator in (
                ("is",),
                (">",),
                ("<",),
                (">", "="),
                ("=", "<"),
                ("=", ":", "="),
                ("=", "\\", "="),
            )
        ):
            structure = "arithmetic_or_comparison"
        elif module.parse_flat_rule(rule) is not None:
            structure = "flat_positive"
        else:
            structure = "other_complex"

        require(row["curriculum level"] is not None, "missing curriculum level")
        require(row["rule sampling"] is not None, "missing rule sampling")
        structure_counts[structure] += 1
        token_lengths.append(len(tokens))
        atom_counts.append(atom_count)

    expected = protocol["p2"]["analysis"]["covariate_test_census"]
    require(len(rows) == expected["rows"] == 1000, "P2 covariate row count changed")
    require(
        dict(sorted(structure_counts.items()))
        == expected["logical_structure_counts"],
        "P2 logical-structure census changed",
    )
    require(
        [min(token_lengths), max(token_lengths)] == expected["token_length_range"],
        "P2 token-length range changed",
    )
    require(
        [min(atom_counts), max(atom_counts)] == expected["atom_count_range"],
        "P2 atom-count range changed",
    )

    witnessed_ids: set[int] = set()
    artifact_dir = root / "external_semantic_overlap_audit" / "artifacts"
    for filename in (
        "slr_semantic_overlap_witnesses.csv",
        "slr_token_overlap_witnesses.csv",
    ):
        with (artifact_dir / filename).open(newline="", encoding="utf-8") as handle:
            witnessed_ids.update(
                int(row["test_id"]) for row in csv.DictReader(handle)
            )
    require(len(witnessed_ids) == 452, "P2 witness-ID union changed")

    level_counts: Counter[tuple[int, bool]] = Counter(
        (int(row["curriculum level"]), int(row["id"]) in witnessed_ids)
        for row in pq.read_table(
            parquet_path,
            columns=["id", "curriculum level"],
        ).to_pylist()
    )
    levels = sorted({level for level, _ in level_counts})
    supported_levels = [
        level
        for level in levels
        if level_counts[(level, False)] > 0
        and level_counts[(level, True)] > 0
    ]
    witnessed_retained = sum(
        level_counts[(level, True)] for level in supported_levels
    )
    no_witness_retained = sum(
        level_counts[(level, False)] for level in supported_levels
    )
    support = protocol["p2"]["common_support"]["exact_curriculum_precheck"]
    require(
        supported_levels == support["levels_with_both_groups"],
        "P2 exact curriculum support levels changed",
    )
    require(
        witnessed_retained == support["witnessed_retained"] == 352
        and len(witnessed_ids) == support["witnessed_total"] == 452,
        "P2 witnessed support counts changed",
    )
    require(
        no_witness_retained == support["no_witness_retained"] == 498
        and len(rows) - len(witnessed_ids) == support["no_witness_total"] == 548,
        "P2 no-witness support counts changed",
    )
    minimum = protocol["p2"]["common_support"][
        "minimum_retained_fraction_per_group"
    ]
    require(
        witnessed_retained / len(witnessed_ids) < minimum,
        "P2 expected common-support No-Go no longer follows from pinned data",
    )


def validate_p3_p4_p5_and_submission(protocol: dict[str, Any]) -> None:
    p3 = protocol["p3"]
    require(
        p3["candidate_count"] == {"max": 5, "min": 2},
        "P3 candidate bounds changed",
    )
    require(
        p3["candidate_manifest_must_precede_semantic_extraction"],
        "P3 manifest is not required before extraction",
    )
    require(p3["result_dependent_selection_forbidden"], "P3 permits selection")
    require(
        p3["historical_cases_excluded"]
        == [
            "AIML-TUDA/SLR-Bench",
            "satwik77/incontext-bool",
            "internal_N20_affine_block_parity_generator",
            "variants_repackaging_the_same_generator_or_split",
        ],
        "P3 historical-case exclusions changed",
    )
    discovery = p3["discovery"]
    require(
        discovery["sources"] == ["Semantic Scholar", "arXiv"]
        and discovery["results_per_query_per_source"] == 100
        and len(discovery["queries"]) == 4
        and discovery["include_rule"]
        == "screen ascending normalized title; include first five eligible, or all if fewer",
        "P3 discovery protocol changed",
    )
    breadth = p3["gates"]["breadth_go"]
    require(
        breadth["complete_public_semantic_audits_at_least"] == 2
        and breadth["behavioral_evidence_benchmark_count_at_least"] == 1,
        "P3 Breadth Gate changed",
    )

    p4_gate = protocol["p4"]["gates"]["tool_contribution_go"]
    require(
        p4_gate["distinct_public_benchmark_adapters_at_least"] == 2,
        "P4 public-adapter minimum changed",
    )
    require(
        set(p4_gate["unit_canaries"])
        == {"exact_function", "sound_alias", "unresolved"},
        "P4 canary set changed",
    )

    p5_gate = protocol["p5"]["gates"]["theory_contribution_go"]
    require(
        p5_gate["instantiated_benchmarks_at_least"] == 2,
        "P5 instantiation minimum changed",
    )
    require(
        p5_gate["complete_proof"]
        and p5_gate["explicit_assumptions"]
        and p5_gate["falsifiable_rule_bound_or_estimator"],
        "P5 proof requirements were relaxed",
    )

    workshop = protocol["submission_gates"]["workshop_ready_go"]
    require(
        workshop["evidence_condition"]
        == "p1_decisive_go OR (p2_identification_go AND p3_breadth_go)",
        "workshop evidence condition changed",
    )
    main = protocol["submission_gates"]["ccf_a_main_track_go"]
    require(
        main["requires_workshop_ready_go"]
        and main["p3_breadth_go_required"]
        and main["p4_tool_contribution_go_required"]
        and main["credible_behavioral_result_required"],
        "main-track Gate was relaxed",
    )


def validate_known_evidence(root: Path, protocol: dict[str, Any]) -> None:
    internal = read_json(
        root
        / "semantic_novelty_audit_supplement"
        / "results"
        / "public_baseline_audit"
        / "analysis"
        / "claim_gates.json"
    )
    expected_internal = protocol["prior_known_evidence"]["internal_legacy_minus_strict"]
    require(
        abs(
            internal["architecture_macro_difference"]
            - expected_internal["effect_pp"] / 100.0
        )
        < 1e-12,
        "known internal contrast does not match the pinned result",
    )

    slr = read_json(
        root
        / "external_semantic_overlap_audit"
        / "artifacts"
        / "slr_overlap_results.json"
    )
    union = slr["sound_semantic_overlap_union"]
    expected_slr = protocol["prior_known_evidence"]["slr"]
    require(
        union["test_overlap_rows"] == expected_slr["semantic_overlap_lower_bound_rows"]
        and union["test_overlap_rate"]
        == expected_slr["semantic_overlap_lower_bound_rate"],
        "known SLR overlap does not match the pinned result",
    )

    slr_output = read_json(
        root
        / "external_semantic_overlap_audit"
        / "artifacts"
        / "slr_official_output_stratification.json"
    )
    require(
        slr_output["paper_crosscheck"]["behavioral_use_status"]
        == "exploratory_only_not_reported_main_table_behavior",
        "known SLR artifact mismatch status changed",
    )

    boolean = read_json(
        root
        / "external_semantic_overlap_audit"
        / "analysis"
        / "boolean-behavioral-v1"
        / "results.json"
    )
    require(boolean["gate"]["no_go"], "known Boolean No-Go changed")
    require(
        not boolean["gate"]["lstm_ranking_module_triggered"],
        "known Boolean LSTM expansion status changed",
    )


def validate_readme(protocol_dir: Path) -> None:
    text = (protocol_dir / "README.md").read_text(encoding="ascii")
    normalized_text = " ".join(text.split())
    required_fragments = [
        "Status: frozen before new confirmatory execution",
        "P1: Complementary Exact-Exposure Intervention",
        "P2: Registered SLR Adjusted Association",
        "P3: Prospective External Replication",
        "P4: Reproducible Audit Toolkit",
        "P5: Formal Contribution",
        "tau_macro",
        "Positive behavioral Go",
        "Null-equivalence Go",
        "quality-enhancement-v1:20261009:p1-arm:{model_seed}:{b}",
        "Llama-3.1-8B-IT-SLR",
        "\"rule inference\" benchmark unseen",
        "Workshop-ready Go",
        "CCF-A main-track Go",
        "No phase may turn overlap alone into evidence of memorization.",
    ]
    missing = [
        fragment
        for fragment in required_fragments
        if " ".join(fragment.split()) not in normalized_text
    ]
    require(not missing, f"README is missing frozen fragments: {missing}")


def main() -> None:
    protocol_dir = Path(__file__).resolve().parent
    root = protocol_dir.parents[2]
    protocol = read_json(protocol_dir / "preregistration.json")

    validate_frozen_checksums(protocol_dir)
    validate_top_level(protocol)
    validate_global_thresholds(protocol)
    validate_source_pins(root, protocol)
    validate_p1_arithmetic(protocol)
    validate_generator_census(root, protocol)
    validate_p2(protocol)
    validate_p2_covariates(root, protocol)
    validate_p3_p4_p5_and_submission(protocol)
    validate_known_evidence(root, protocol)
    validate_readme(protocol_dir)

    report = {
        "checks": {
            "frozen_checksums": "passed",
            "top_level_freeze_state": "passed",
            "global_threshold_consistency": "passed",
            "source_pins": "passed",
            "p1_arithmetic": "passed",
            "p1_generator_census": "passed",
            "p2_identification_and_counts": "passed",
            "p2_covariate_and_support_census": "passed",
            "p3_p4_p5_submission_gates": "passed",
            "known_evidence_crosscheck": "passed",
            "readme_machine_spec_alignment": "passed",
        },
        "checksum_manifest_sha256": sha256_file(protocol_dir / "SHA256SUMS"),
        "freeze_date": protocol["freeze_date"],
        "preregistration_sha256": sha256_file(
            protocol_dir / "preregistration.json"
        ),
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (protocol_dir / "validation_report.json").write_text(
        output,
        encoding="ascii",
    )
    print(output, end="")


if __name__ == "__main__":
    main()
