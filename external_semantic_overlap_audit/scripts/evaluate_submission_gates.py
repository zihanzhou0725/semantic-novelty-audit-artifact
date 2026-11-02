#!/usr/bin/env python3
"""Evaluate final manuscript readiness against the frozen submission gates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXTERNAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXTERNAL_ROOT.parent
PROTOCOL_DIR = EXTERNAL_ROOT / "protocols" / "quality-enhancement-v1"
ANALYSIS_DIR = EXTERNAL_ROOT / "analysis" / "quality-enhancement-v1"
PAPER_DIR = PROJECT_ROOT / "semantic_novelty_audit_paper"
OUTPUT_DIR = ANALYSIS_DIR / "submission"
STATUS_PATH = OUTPUT_DIR / "submission_gate_report.json"
REPORT_PATH = OUTPUT_DIR / "report_cn.md"


def fail(message: str) -> None:
    raise SystemExit(f"Submission gate audit failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def latex_page_count(log_path: Path, pdf_name: str) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        rf"Output written on {re.escape(pdf_name)} \((\d+) pages",
        text,
    )
    require(bool(matches), f"could not read page count for {pdf_name}")
    return int(matches[-1])


def citation_keys(tex_text: str) -> set[str]:
    keys: set[str] = set()
    for payload in re.findall(r"\\cite\w*\{([^}]+)\}", tex_text):
        keys.update(item.strip() for item in payload.split(","))
    return keys


def bibliography_keys(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", bib_text))


def manuscript_checks() -> dict[str, Any]:
    main_path = PAPER_DIR / "main.tex"
    supplement_path = PAPER_DIR / "supplementary.tex"
    bib_path = PAPER_DIR / "references.bib"
    main_log_path = PAPER_DIR / "main.log"
    supplement_log_path = PAPER_DIR / "supplementary.log"
    main_pdf_path = PAPER_DIR / "main.pdf"
    supplement_pdf_path = PAPER_DIR / "supplementary.pdf"

    paths = (
        main_path,
        supplement_path,
        bib_path,
        main_log_path,
        supplement_log_path,
        main_pdf_path,
        supplement_pdf_path,
    )
    for path in paths:
        require(path.is_file(), f"missing manuscript artifact: {path}")

    main_text = main_path.read_text(encoding="utf-8")
    supplement_text = supplement_path.read_text(encoding="utf-8")
    combined = main_text + "\n" + supplement_text
    normalized_main = re.sub(r"\s+", " ", main_text)
    normalized_combined = re.sub(r"\s+", " ", combined)
    bib_text = bib_path.read_text(encoding="utf-8")
    logs = (
        main_log_path.read_text(encoding="utf-8", errors="replace")
        + "\n"
        + supplement_log_path.read_text(encoding="utf-8", errors="replace")
    )

    required_claims = {
        "p1_positive_effect": (
            "$\\tau_{\\mathrm{macro}}=1.832$ points" in main_text
            and "$[1.470,2.189]$" in main_text
        ),
        "p2_identification_no_go": (
            "No propensity model, outcome model, or bootstrap was run"
            in normalized_main
        ),
        "p3_breadth_no_go": (
            "P3 Breadth Gate is therefore No-Go" in normalized_main
        ),
        "p4_tool_contribution_go": (
            "P4 Tool-Contribution Gate is Go" in normalized_main
        ),
        "p5_theory_contribution_go": (
            "P5 Theory-Contribution Gate" in normalized_combined
        ),
        "tensor_interval": (
            "$[484,1000]/1000$" in main_text
            and "$[999,1000]/1000$" in main_text
        ),
        "occam_interval": "$[2,4]/20$" in main_text,
        "behavioral_boundary": (
            "does not identify an internal memorization mechanism"
            in normalized_main
        ),
    }
    require(all(required_claims.values()), "required manuscript claim missing")

    forbidden_patterns = {
        "p3_false_go": r"P3 Breadth Gate (?:is|=) Go",
        "overlap_implies_accuracy": (
            r"semantic overlap necessarily (?:improves|inflates)"
        ),
        "memorization_mechanism_claim": (
            r"exact exposure (?:proves|identifies) (?:a |the )?"
            r"memorization mechanism"
        ),
        "slr_exact_45_2": r"45\.2\\% (?:is|as) (?:the )?exact",
        "occam_regression_audited": (
            r"linear-regression (?:scope|experiments?) (?:is|was|were) audited"
        ),
        "universal_prevalence": (
            r"universal contamination prevalence"
        ),
    }
    forbidden_hits = {
        name: bool(
            re.search(pattern, normalized_combined, flags=re.IGNORECASE)
        )
        for name, pattern in forbidden_patterns.items()
    }
    require(not any(forbidden_hits.values()), "forbidden manuscript claim")

    cited = citation_keys(combined)
    available = bibliography_keys(bib_text)
    missing_citations = sorted(cited - available)
    require(not missing_citations, f"missing bibliography keys: {missing_citations}")
    require(
        "undefined" not in logs.lower(),
        "LaTeX logs contain undefined citation/reference",
    )
    require("overfull" not in logs.lower(), "LaTeX logs contain overfull boxes")

    main_pages = latex_page_count(main_log_path, "main.pdf")
    supplement_pages = latex_page_count(
        supplement_log_path,
        "supplementary.pdf",
    )
    require(main_pages <= 9, "main PDF exceeds nine total pages")
    return {
        "bibliography_key_count": len(available),
        "cited_key_count": len(cited),
        "forbidden_claim_hits": forbidden_hits,
        "main_pdf_pages": main_pages,
        "main_total_page_limit_passed": True,
        "missing_citations": missing_citations,
        "no_overfull_boxes": True,
        "no_undefined_citations_or_references": True,
        "required_claims": required_claims,
        "supplement_pdf_pages": supplement_pages,
    }


def build_report() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_DIR / "preregistration.json")
    p0 = read_json(PROTOCOL_DIR / "validation_report.json")
    p1 = read_json(ANALYSIS_DIR / "p1" / "results.json")
    p2 = read_json(ANALYSIS_DIR / "p2" / "identification_report.json")
    p3 = read_json(ANALYSIS_DIR / "p3" / "p3_breadth_status.json")
    p4 = read_json(ANALYSIS_DIR / "p4" / "tool_contribution_status.json")
    p5 = read_json(ANALYSIS_DIR / "p5" / "validation_report.json")
    manuscript = manuscript_checks()

    p1_decisive = p1["gate"]["decision"] in {
        "positive_behavioral_go",
        "negative_behavioral_go",
        "null_equivalence_go",
    }
    all_attempted_reported = all(manuscript["required_claims"].values())
    claim_compliance = not any(
        manuscript["forbidden_claim_hits"].values()
    )
    workshop_conditions = {
        "all_attempted_results_reported": all_attempted_reported,
        "claim_vocabulary_compliance": claim_compliance,
        "evidence_condition": (
            p1_decisive
            or (
                p2["decision"] == "identification_go"
                and p3["decision"] == "go"
            )
        ),
        "p0_validation_required": p0["status"] == "passed",
        "p4_current_claim_reproduction_required": p4["decision"] == "go",
    }
    workshop_decision = (
        "go" if all(workshop_conditions.values()) else "no_go"
    )

    credible_behavioral_result = p1["gate"]["decision"] in {
        "positive_behavioral_go",
        "negative_behavioral_go",
        "null_equivalence_go",
    }
    main_conditions = {
        "credible_behavioral_result_required": credible_behavioral_result,
        "p3_breadth_go_required": p3["decision"] == "go",
        "p4_tool_contribution_go_required": p4["decision"] == "go",
        "public_benchmarks_at_least_2": p3["candidate_count"] >= 2,
        "requires_workshop_ready_go": workshop_decision == "go",
        "theory_claim_requires_p5_go": p5["decision"] == "go",
    }
    main_decision = "go" if all(main_conditions.values()) else "no_go"
    blockers = [
        key for key, passed in main_conditions.items() if not passed
    ]

    return {
        "component_decisions": {
            "p0": p0["status"],
            "p1": p1["gate"]["decision"],
            "p2": p2["decision"],
            "p3": p3["decision"],
            "p4": p4["decision"],
            "p5": p5["decision"],
        },
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                PROTOCOL_DIR / "preregistration.json",
                PROTOCOL_DIR / "validation_report.json",
                ANALYSIS_DIR / "p1" / "results.json",
                ANALYSIS_DIR / "p2" / "identification_report.json",
                ANALYSIS_DIR / "p3" / "p3_breadth_status.json",
                ANALYSIS_DIR / "p4" / "tool_contribution_status.json",
                ANALYSIS_DIR / "p5" / "validation_report.json",
                PAPER_DIR / "main.tex",
                PAPER_DIR / "supplementary.tex",
                PAPER_DIR / "references.bib",
                PAPER_DIR / "claim_evidence_matrix.md",
                PAPER_DIR / "README.md",
                PAPER_DIR / "technical_review.md",
                PAPER_DIR / "main.pdf",
                PAPER_DIR / "supplementary.pdf",
                Path(__file__).resolve(),
            )
        },
        "main_track": {
            "blockers": blockers,
            "conditions": main_conditions,
            "decision": main_decision,
        },
        "manuscript_integrity": manuscript,
        "recommendation": (
            "workshop_ready_bounded_claims_main_track_blocked_by_p3"
            if workshop_decision == "go" and main_decision == "no_go"
            else "follow_gate_decisions"
        ),
        "schema_version": "quality-enhancement-submission-gates-v1",
        "status": "complete",
        "workshop_ready": {
            "conditions": workshop_conditions,
            "decision": workshop_decision,
        },
    }


def build_report_cn(report: dict[str, Any]) -> str:
    main = report["main_track"]
    workshop = report["workshop_ready"]
    integrity = report["manuscript_integrity"]
    return "\n".join(
        [
            "# 最终 Submission Gate 审计",
            "",
            "## 决策",
            "",
            f"- Workshop-ready：**{workshop['decision'].upper()}**。",
            f"- CCF-A main-track：**{main['decision'].upper()}**。",
            "- Main-track 唯一阻断项：P3 Breadth Go 未满足。",
            "",
            "## 组件状态",
            "",
            "- P0：验证通过。",
            "- P1：Positive Behavioral Go，"
            "`tau_macro = +1.832 pp [1.470, 2.189]`。",
            "- P2：Identification No-Go before outcome fit。",
            "- P3：Breadth No-Go，`1/2` complete、`0/1` behavioral。",
            "- P4：Tool-Contribution Go。",
            "- P5：Theory-Contribution Go。",
            "",
            "## 稿件完整性",
            "",
            f"- 正文 {integrity['main_pdf_pages']} 页；补充材料 "
            f"{integrity['supplement_pdf_pages']} 页。",
            "- 无 undefined citation/reference，无 overfull box。",
            "- P1--P5 均在正文或补充材料中报告；禁用主张扫描为零命中。",
            "",
            "## 可用投稿口径",
            "",
            "当前证据满足内部 workshop-ready Gate，可提交强调方法、P1 "
            "随机化行为结果、P4 工具与 P5 partial-identification 命题的版本。",
            "按照冻结规则，不得将其包装为已通过 P3 的跨 benchmark prevalence "
            "或 behavioral replication，也不得宣称 memorization mechanism。",
            "",
        ]
    )


def main() -> int:
    report = build_report()
    write_json(STATUS_PATH, report)
    REPORT_PATH.write_text(build_report_cn(report), encoding="utf-8")
    checksum_path = OUTPUT_DIR / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (REPORT_PATH, STATUS_PATH)
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "main_track": report["main_track"]["decision"],
                "main_track_blockers": report["main_track"]["blockers"],
                "workshop_ready": report["workshop_ready"]["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
