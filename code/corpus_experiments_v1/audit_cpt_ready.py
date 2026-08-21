from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REFERENCE_LIKE_RE = re.compile(
    r"^(?:\[?\d{1,3}\]?\s*[.)]|references?\b|bibliography\b)|\bdoi\s*:\s*10\.\d{4,9}/",
    re.IGNORECASE,
)
MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â€", "ï¬", "\x00")


def normalized_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _token_lengths(tokenizer: Any, texts: list[str], batch_size: int = 512) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            truncation=False,
            return_length=True,
            verbose=False,
        )
        if "length" in encoded:
            lengths.extend(int(value) for value in encoded["length"])
        else:
            lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def frozen_duplicate_context(
    project_root: Path,
    member_ids: set[str],
    manifest_db: Path,
) -> dict[str, Any]:
    gate0_path = project_root / "code/14_source_selection/output/gate0_duplicate_validation.json"
    features_path = project_root / "code/14_source_selection/output/paper_features.jsonl"
    gate0 = json.loads(gate0_path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = defaultdict(list)
    with features_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["paper_id"] not in member_ids:
                continue
            title = re.sub(r"\W+", "", unicodedata.normalize("NFKC", row.get("title") or "").casefold())
            if title:
                groups[title].append(row["paper_id"])
    collision_groups = [members for members in groups.values() if len(members) > 1]

    con = sqlite3.connect(f"file:{manifest_db.resolve().as_posix()}?mode=ro", uri=True)
    version_groups: dict[str, list[str]] = defaultdict(list)
    for paper_id, group in con.execute(
        "select paper_id, version_group from source_pool_members where pool_id='S2K'"
    ):
        version_groups[group].append(paper_id)
    con.close()
    reliable_multi = [members for members in version_groups.values() if len(members) > 1]
    signals = gate0["duplicate_candidate_signals"]
    return {
        "frozen_gate0_scope_papers": gate0["input"]["paper_count"],
        "frozen_gate0_exact_normalized_title_collision_groups": signals["exact_normalized_title_collision_groups"],
        "frozen_gate0_papers_in_title_collision_groups": signals["papers_in_exact_normalized_title_collision_groups"],
        "frozen_gate0_semantic_threshold": signals["semantic_nearest_neighbour"]["threshold_applied"],
        "s2k_exact_normalized_title_collision_groups": len(collision_groups),
        "s2k_papers_in_title_collision_groups": sum(len(group) for group in collision_groups),
        "s2k_largest_title_collision_group": max((len(group) for group in collision_groups), default=0),
        "s2k_reliable_version_duplicate_groups": len(reliable_multi),
        "s2k_papers_in_reliable_version_duplicate_groups": sum(len(group) for group in reliable_multi),
        "interpretation": "Frozen candidate signals reused; no semantic threshold was invented. S2K title collisions remain candidates, not asserted duplicates.",
    }


def audit_parquet(
    parquet_path: Path,
    tokenizer_snapshot: Path,
    tokenizer_meta: dict[str, Any],
    project_root: Path,
    manifest_db: Path,
    manifest_strata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True, use_fast=True)
    parquet = pq.ParquetFile(parquet_path)
    documents: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(batch_size=16):
        documents.extend(batch.to_pylist())

    raw_texts: list[str] = []
    clean_texts: list[str] = []
    owners: list[str] = []
    fragment_groups: dict[str, dict[str, Any]] = {}
    unicode_counts = Counter()
    contamination = Counter()
    repeated_candidates = 0
    suspected = 0
    excluded = Counter()
    section_assigned = 0
    paragraph_count = 0

    for document in documents:
        paper_id = document["paper_id"]
        excluded.update(json.loads(document["excluded_counts_json"]))
        repeated: dict[str, set[int]] = defaultdict(set)
        for paragraph in document["paragraphs"]:
            raw = paragraph["raw_text"]
            clean = paragraph["clean_text"]
            raw_texts.append(raw)
            clean_texts.append(clean)
            owners.append(paper_id)
            paragraph_count += 1
            section_assigned += int(paragraph.get("section_id") is not None)
            suspected += int(bool(paragraph.get("suspected_word_split")))
            norm = normalized_fragment(clean)
            if len(norm) >= 200:
                digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                group = fragment_groups.setdefault(digest, {"papers": set(), "paragraphs": 0})
                group["papers"].add(paper_id)
                group["paragraphs"] += 1
            if 20 <= len(norm) <= 200 and paragraph.get("page") is not None:
                repeated[norm].add(int(paragraph["page"]))
            if REFERENCE_LIKE_RE.search(clean):
                contamination["reference_like_body_paragraphs"] += 1
            if any(marker in raw for marker in MOJIBAKE_MARKERS):
                unicode_counts["replacement_nul_or_mojibake_paragraphs"] += 1
            if unicodedata.normalize("NFKC", raw) != raw:
                unicode_counts["nfkc_changed_paragraphs"] += 1
            if any(unicodedata.category(char) in {"Cc", "Cf"} for char in raw):
                unicode_counts["raw_control_or_format_character_paragraphs"] += 1
            if any(unicodedata.category(char) in {"Cc", "Cf"} for char in clean):
                unicode_counts["clean_control_or_format_character_paragraphs"] += 1
        repeated_candidates += sum(len(pages) >= 3 for pages in repeated.values())

    raw_lengths = _token_lengths(tokenizer, raw_texts)
    clean_lengths = _token_lengths(tokenizer, clean_texts)
    per_paper: dict[str, dict[str, Any]] = defaultdict(lambda: {"body_raw_tokens": 0, "body_clean_tokens": 0})
    for paper_id, raw_count, clean_count in zip(owners, raw_lengths, clean_lengths):
        per_paper[paper_id]["body_raw_tokens"] += raw_count
        per_paper[paper_id]["body_clean_tokens"] += clean_count
    for values in per_paper.values():
        raw_count = values["body_raw_tokens"]
        values["clean_token_retention_rate"] = values["body_clean_tokens"] / raw_count if raw_count else None

    cross_document = [
        value for value in fragment_groups.values() if len(value["papers"]) > 1
    ]
    affected_papers = set().union(*(value["papers"] for value in cross_document)) if cross_document else set()
    duplicates = frozen_duplicate_context(project_root, set(per_paper), manifest_db)
    raw_tokens = sum(raw_lengths)
    clean_tokens = sum(clean_lengths)
    summary = {
        "tokenizer": {
            "repository_id": tokenizer_meta["repository_id"],
            "snapshot": str(tokenizer_snapshot),
            "revision": tokenizer_meta["revision"],
            "tokenizer_bundle_sha256": tokenizer_meta["tokenizer_bundle_sha256"],
            "local_files_only": True,
            "add_special_tokens": False,
        },
        "document_count": len(documents),
        "body_paragraph_count": paragraph_count,
        "body_raw_tokens": raw_tokens,
        "body_clean_tokens": clean_tokens,
        "clean_token_retention_rate": clean_tokens / raw_tokens if raw_tokens else None,
        "section_structure_coverage": section_assigned / paragraph_count if paragraph_count else None,
        "suspected_word_split_paragraphs": suspected,
        "unicode_ocr_anomalies": dict(sorted(unicode_counts.items())),
        "contamination_candidates": {
            **dict(sorted(contamination.items())),
            "within_paper_repeated_short_body_fragments_on_3plus_pages": repeated_candidates,
            "explicitly_excluded_by_parser": dict(sorted(excluded.items())),
        },
        "paragraph_fragment_similarity": {
            "definition": "NFKC+casefold+whitespace exact match for body fragments with at least 200 characters; cross-document only",
            "cross_document_groups": len(cross_document),
            "paragraph_members": sum(value["paragraphs"] for value in cross_document),
            "affected_papers": len(affected_papers),
            "largest_group_paragraph_members": max((value["paragraphs"] for value in cross_document), default=0),
        },
        "paper_duplicate_context": duplicates,
        "stratified_distributions": manifest_strata,
    }
    return summary, dict(per_paper)
