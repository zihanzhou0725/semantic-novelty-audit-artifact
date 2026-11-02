#!/usr/bin/env python3
"""Freeze P3 Semantic Scholar and arXiv search results before candidate audit."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi

from quality_p1_common import EXTERNAL_ROOT, PROTOCOL_DIR, ROOT, sha256_file, write_json


SCHEMA_VERSION = "quality-enhancement-p3-discovery-v1"
P3_DIR = PROTOCOL_DIR / "p3"
DISCOVERY_DIR = P3_DIR / "discovery"
RAW_DIR = DISCOVERY_DIR / "raw"
QUERIES = (
    '"rule inference" benchmark unseen',
    '"in-context learning" Boolean functions',
    '"symbolic reasoning" benchmark held-out rules',
    '"compositional generalization" unseen functions benchmark',
)
ARXIV_QUERIES = (
    'all:"rule inference" AND all:benchmark AND all:unseen',
    'all:"in-context learning" AND all:"Boolean functions"',
    'all:"symbolic reasoning" AND all:benchmark AND all:"held-out rules"',
    'all:"compositional generalization" AND all:"unseen functions" AND all:benchmark',
)
SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,year,authors,venue,externalIds,url,"
    "openAccessPdf,publicationTypes"
)
USER_AGENT = (
    "SemanticNoveltyAudit/1.0 "
    "(prospective benchmark discovery; contact: local-research-agent)"
)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fail(message: str) -> None:
    raise SystemExit(f"P3 discovery failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character if character.isalnum() else " " for character in normalized
    )
    return normalize_space(normalized)


def arxiv_base_id(value: str | None) -> str:
    if not value:
        return ""
    candidate = value.rsplit("/", maxsplit=1)[-1]
    return re.sub(r"v\d+$", "", candidate, flags=re.IGNORECASE).lower()


class Fetcher:
    def __init__(self) -> None:
        self.last_request: dict[str, float] = {}

    def fetch(self, url: str, provider: str) -> tuple[bytes, dict[str, str]]:
        minimum_interval = 6.0 if provider == "semantic_scholar" else 3.0
        elapsed = time.monotonic() - self.last_request.get(provider, 0.0)
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        last_error: Exception | None = None
        for attempt in range(1, 7):
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": (
                        "application/json"
                        if provider == "semantic_scholar"
                        else "application/atom+xml"
                    ),
                    "Accept-Encoding": "identity",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                self.last_request[provider] = time.monotonic()
                with urllib.request.urlopen(
                    request,
                    timeout=90,
                    context=SSL_CONTEXT,
                ) as response:
                    body = response.read()
                    headers = {
                        key.lower(): value
                        for key, value in response.headers.items()
                    }
                    return body, headers
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code != 429 or attempt == 6:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 10.0 * attempt
                time.sleep(min(delay, 60.0))
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
                if attempt == 6:
                    raise
                time.sleep(5.0 * attempt)
        raise RuntimeError(f"Unreachable fetch failure: {last_error}")


def semantic_scholar_url(query: str) -> str:
    parameters = urllib.parse.urlencode(
        {
            "query": query,
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
    )
    return (
        "https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
        + parameters
    )


def arxiv_url(query: str) -> str:
    parameters = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": 100,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    return "https://export.arxiv.org/api/query?" + parameters


def parse_semantic_scholar(
    body: bytes,
    query_index: int,
) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(body)
    data = payload.get("data", [])
    require(isinstance(data, list), "Semantic Scholar data is not a list")
    records: list[dict[str, Any]] = []
    for rank, paper in enumerate(data[:100], start=1):
        external_ids = paper.get("externalIds") or {}
        open_access_pdf = paper.get("openAccessPdf") or {}
        records.append(
            {
                "abstract": normalize_space(paper.get("abstract")),
                "arxiv_id": arxiv_base_id(external_ids.get("ArXiv")),
                "authors": [
                    normalize_space(author.get("name"))
                    for author in paper.get("authors") or []
                ],
                "doi": normalize_space(external_ids.get("DOI")).lower(),
                "external_ids": external_ids,
                "paper_id": normalize_space(paper.get("paperId")),
                "pdf_url": normalize_space(open_access_pdf.get("url")),
                "provider": "semantic_scholar",
                "provider_rank": rank,
                "publication_types": paper.get("publicationTypes") or [],
                "query": QUERIES[query_index - 1],
                "query_index": query_index,
                "title": normalize_space(paper.get("title")),
                "url": normalize_space(paper.get("url")),
                "venue": normalize_space(paper.get("venue")),
                "year": paper.get("year"),
            }
        )
    return records, int(payload.get("total", len(data)))


def child_text(node: ET.Element, name: str, namespaces: dict[str, str]) -> str:
    child = node.find(name, namespaces)
    return normalize_space(child.text if child is not None else "")


def parse_arxiv(
    body: bytes,
    query_index: int,
) -> tuple[list[dict[str, Any]], int]:
    root = ET.fromstring(body)
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    }
    total_text = child_text(root, "opensearch:totalResults", namespaces)
    total = int(total_text or 0)
    records: list[dict[str, Any]] = []
    for rank, entry in enumerate(
        root.findall("atom:entry", namespaces)[:100],
        start=1,
    ):
        entry_id = child_text(entry, "atom:id", namespaces)
        links = [
            {
                "href": normalize_space(link.attrib.get("href")),
                "rel": normalize_space(link.attrib.get("rel")),
                "type": normalize_space(link.attrib.get("type")),
                "title": normalize_space(link.attrib.get("title")),
            }
            for link in entry.findall("atom:link", namespaces)
        ]
        pdf_url = next(
            (
                link["href"]
                for link in links
                if link["title"].lower() == "pdf"
                or link["type"] == "application/pdf"
            ),
            "",
        )
        alternate_url = next(
            (
                link["href"]
                for link in links
                if link["rel"] == "alternate"
            ),
            entry_id,
        )
        doi = child_text(entry, "arxiv:doi", namespaces).lower()
        records.append(
            {
                "abstract": child_text(entry, "atom:summary", namespaces),
                "arxiv_id": arxiv_base_id(entry_id),
                "authors": [
                    child_text(author, "atom:name", namespaces)
                    for author in entry.findall("atom:author", namespaces)
                ],
                "categories": [
                    normalize_space(category.attrib.get("term"))
                    for category in entry.findall("atom:category", namespaces)
                ],
                "comment": child_text(entry, "arxiv:comment", namespaces),
                "doi": doi,
                "journal_reference": child_text(
                    entry,
                    "arxiv:journal_ref",
                    namespaces,
                ),
                "paper_id": arxiv_base_id(entry_id),
                "pdf_url": pdf_url,
                "provider": "arxiv",
                "provider_rank": rank,
                "publication_types": ["Preprint"],
                "published": child_text(entry, "atom:published", namespaces),
                "query": QUERIES[query_index - 1],
                "query_index": query_index,
                "title": child_text(entry, "atom:title", namespaces),
                "updated": child_text(entry, "atom:updated", namespaces),
                "url": alternate_url,
                "venue": "arXiv",
                "year": int(
                    child_text(entry, "atom:published", namespaces)[:4] or 0
                ),
            }
        )
    return records, total


def merge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    doi_map: dict[str, int] = {}
    arxiv_map: dict[str, int] = {}
    title_map: dict[str, int] = {}
    for record in records:
        title_key = normalize_title(record["title"])
        candidates = []
        if record["doi"] and record["doi"] in doi_map:
            candidates.append(doi_map[record["doi"]])
        if record["arxiv_id"] and record["arxiv_id"] in arxiv_map:
            candidates.append(arxiv_map[record["arxiv_id"]])
        if title_key and title_key in title_map:
            candidates.append(title_map[title_key])
        if candidates:
            group_index = min(candidates)
            group = groups[group_index]
        else:
            group_index = len(groups)
            group = {
                "abstracts": [],
                "arxiv_ids": [],
                "authors": [],
                "dois": [],
                "hits": [],
                "pdf_urls": [],
                "provider_ids": [],
                "publication_types": [],
                "titles": [],
                "urls": [],
                "venues": [],
                "years": [],
            }
            groups.append(group)
        group["hits"].append(
            {
                "provider": record["provider"],
                "provider_rank": record["provider_rank"],
                "query_index": record["query_index"],
            }
        )
        for key, value in (
            ("abstracts", record["abstract"]),
            ("arxiv_ids", record["arxiv_id"]),
            ("dois", record["doi"]),
            ("pdf_urls", record["pdf_url"]),
            ("provider_ids", f"{record['provider']}:{record['paper_id']}"),
            ("publication_types", record["publication_types"]),
            ("titles", record["title"]),
            ("urls", record["url"]),
            ("venues", record["venue"]),
            ("years", record["year"]),
        ):
            if value and value not in group[key]:
                group[key].append(value)
        for author in record["authors"]:
            if author and author not in group["authors"]:
                group["authors"].append(author)
        if record["doi"]:
            doi_map[record["doi"]] = group_index
        if record["arxiv_id"]:
            arxiv_map[record["arxiv_id"]] = group_index
        if title_key:
            title_map[title_key] = group_index

    output: list[dict[str, Any]] = []
    for group in groups:
        title = sorted(
            group["titles"],
            key=lambda value: (len(value), value.casefold()),
        )[0]
        normalized = normalize_title(title)
        stable_key = (
            f"doi:{sorted(group['dois'])[0]}"
            if group["dois"]
            else (
                f"arxiv:{sorted(group['arxiv_ids'])[0]}"
                if group["arxiv_ids"]
                else f"title:{normalized}"
            )
        )
        output.append(
            {
                "abstract": max(group["abstracts"], key=len, default=""),
                "arxiv_ids_json": canonical_json(sorted(group["arxiv_ids"])),
                "authors_json": canonical_json(group["authors"]),
                "candidate_id": hashlib.sha256(
                    stable_key.encode("utf-8")
                ).hexdigest()[:16],
                "deduplication_key": stable_key,
                "dois_json": canonical_json(sorted(group["dois"])),
                "normalized_title": normalized,
                "pdf_urls_json": canonical_json(sorted(group["pdf_urls"])),
                "provider_ids_json": canonical_json(
                    sorted(group["provider_ids"])
                ),
                "query_hits_json": canonical_json(
                    sorted(
                        group["hits"],
                        key=lambda row: (
                            row["query_index"],
                            row["provider"],
                            row["provider_rank"],
                        ),
                    )
                ),
                "screening_status": "not_screened",
                "title": title,
                "urls_json": canonical_json(sorted(group["urls"])),
                "venues_json": canonical_json(sorted(group["venues"])),
                "years_json": canonical_json(
                    sorted(
                        {int(value) for value in group["years"] if value}
                    )
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (row["normalized_title"], row["candidate_id"]),
    )


def validate_parent() -> dict[str, Any]:
    validation = json.loads(
        (PROTOCOL_DIR / "validation_report.json").read_text(encoding="ascii")
    )
    preregistration_path = PROTOCOL_DIR / "preregistration.json"
    require(validation.get("status") == "passed", "parent validation failed")
    require(
        validation.get("preregistration_sha256")
        == sha256_file(preregistration_path),
        "parent validation refers to another preregistration",
    )
    preregistration = json.loads(
        preregistration_path.read_text(encoding="ascii")
    )
    p3 = preregistration["p3"]
    require(list(QUERIES) == p3["discovery"]["queries"], "P3 queries changed")
    require(
        p3["discovery"]["sources"] == ["Semantic Scholar", "arXiv"],
        "P3 sources changed",
    )
    require(
        p3["discovery"]["results_per_query_per_source"] == 100,
        "P3 result limit changed",
    )
    require(
        p3["discovery"]["deduplication_keys_in_order"]
        == ["doi", "arxiv_id", "normalized_title"],
        "P3 deduplication keys changed",
    )
    return preregistration


def main() -> int:
    preregistration = validate_parent()
    output_paths = [
        DISCOVERY_DIR / "discovery_manifest.json",
        DISCOVERY_DIR / "search_results.jsonl",
        DISCOVERY_DIR / "deduplicated_screening_queue.csv",
        DISCOVERY_DIR / "SHA256SUMS",
    ]
    output_paths.extend(
        RAW_DIR / f"{provider}_q{index}.{extension}"
        for provider, extension in (
            ("semantic_scholar", "json"),
            ("arxiv", "xml"),
        )
        for index in range(1, len(QUERIES) + 1)
    )
    for path in output_paths:
        require(not path.exists(), f"discovery output already exists: {path}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetcher = Fetcher()
    records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for query_index, query in enumerate(QUERIES, start=1):
        url = semantic_scholar_url(query)
        body, headers = fetcher.fetch(url, "semantic_scholar")
        raw_path = RAW_DIR / f"semantic_scholar_q{query_index}.json"
        raw_path.write_bytes(body)
        parsed, total = parse_semantic_scholar(body, query_index)
        records.extend(parsed)
        requests.append(
            {
                "api_mode": (
                    "official_graph_bulk_search_fallback_after_public_"
                    "relevance_endpoint_returned_http_429"
                ),
                "provider": "Semantic Scholar",
                "provider_reported_total": total,
                "query": query,
                "query_index": query_index,
                "request_url": url,
                "response_content_type": headers.get("content-type", ""),
                "response_sha256": sha256_file(raw_path),
                "results_retained_in_provider_order": len(parsed),
            }
        )

    for query_index, query in enumerate(ARXIV_QUERIES, start=1):
        url = arxiv_url(query)
        body, headers = fetcher.fetch(url, "arxiv")
        raw_path = RAW_DIR / f"arxiv_q{query_index}.xml"
        raw_path.write_bytes(body)
        parsed, total = parse_arxiv(body, query_index)
        records.extend(parsed)
        requests.append(
            {
                "api_mode": "official_atom_api_relevance_descending",
                "provider": "arXiv",
                "provider_query": query,
                "provider_reported_total": total,
                "query": QUERIES[query_index - 1],
                "query_index": query_index,
                "request_url": url,
                "response_content_type": headers.get("content-type", ""),
                "response_sha256": sha256_file(raw_path),
                "results_retained_in_provider_order": len(parsed),
            }
        )

    require(len(requests) == 8, "did not execute all eight frozen searches")
    results_path = DISCOVERY_DIR / "search_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    queue = merge_records(records)
    queue_path = DISCOVERY_DIR / "deduplicated_screening_queue.csv"
    require(queue, "discovery returned no candidate records")
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)

    manifest = {
        "candidate_dataset_or_code_download_started": False,
        "candidate_semantic_inspection_started": False,
        "deduplicated_record_count": len(queue),
        "deduplication": {
            "keys_in_order": ["doi", "arxiv_id", "normalized_title"],
            "normalized_title": (
                "Unicode NFKC, casefold, replace every non-alphanumeric "
                "character with space, collapse whitespace"
            ),
        },
        "discovery_script_sha256": sha256_file(Path(__file__).resolve()),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_case_exclusions": preregistration["p3"][
            "historical_cases_excluded"
        ],
        "include_rule": preregistration["p3"]["discovery"]["include_rule"],
        "parsed_result_count_before_deduplication": len(records),
        "requests": sorted(
            requests,
            key=lambda row: (row["query_index"], row["provider"]),
        ),
        "schema_version": SCHEMA_VERSION,
        "screening_order": "ascending_normalized_title_then_candidate_id",
        "status": "raw_discovery_frozen_before_candidate_semantic_inspection",
    }
    manifest_path = DISCOVERY_DIR / "discovery_manifest.json"
    write_json(manifest_path, manifest)

    checksum_files = [
        path
        for path in output_paths
        if path.name != "SHA256SUMS"
    ]
    checksum_path = DISCOVERY_DIR / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(DISCOVERY_DIR).as_posix()}"
            for path in sorted(checksum_files)
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
