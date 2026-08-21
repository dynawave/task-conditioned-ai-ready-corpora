#!/usr/bin/env python3
"""Frozen corpus_v1 RAG engineering audit and public-gold retrieval experiment."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import platform
import random
import re
import sqlite3
import statistics
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


EXPERIMENT_ID = "corpus_v1_rag_public_gold_v1"
SEED = 20260811
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 10_000
MODEL = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
SUPPLEMENTARY_TOP_K = 10
TOKEN_BUDGET = 1024
CANONICAL_SHA256 = "2c72b3615d5299534d6336c9d4a4b4db0982853011718babeabc2aafa18e007b"
S2K_MEMBERSHIP_SHA256 = "4381084194802c40c2a55fa77a32fb0fcc9e6b44dbd6a9e8fbfd38dc8fd108f4"
QASPER_ARCHIVE_SHA256 = "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"
QASPER_GOLD_SHA256 = "5f44034796f9689fa70c1d18e6ea6e39ef6576efa9dce863728b9885500ca994"
QASPER_VERSION = "v0.3"
QASPER_SPLIT = "test"
MAPPING_GATE = 0.95

LABEL_RE = re.compile(
    r"\b(fig(?:ure)?|table|eq(?:uation)?)\.?\s*\(?([A-Za-z]?\d+[A-Za-z]?)\)?",
    re.IGNORECASE,
)
FORMULA_CUE_RE = re.compile(r"\b(?:eq(?:uation)?\.?\s*\(?\d+\)?|equation)\b", re.IGNORECASE)
TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def normalize_title(value: Any) -> str:
    return " ".join(TITLE_TOKEN_RE.findall(normalize_text(value).lower()))


def percentile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def token_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": int(min(values)) if values else 0,
        "mean": float(statistics.fmean(values)) if values else 0.0,
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": int(max(values)) if values else 0,
    }


def json_bytes(record: dict[str, Any]) -> int:
    return len((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def canonical_paragraphs(document: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    section_map = {section["section_id"]: section for section in document.get("sections") or []}
    paragraphs: list[dict[str, Any]] = []
    for position, raw in enumerate(document.get("paragraphs") or []):
        text = normalize_text(raw.get("clean_text"))
        if not text:
            continue
        section = section_map.get(raw.get("section_id"), {})
        paragraphs.append(
            {
                "id": raw["paragraph_id"],
                "paper_id": document["paper_id"],
                "position": position,
                "section_id": raw.get("section_id"),
                "section_path": list(section.get("section_path") or []),
                "text": text,
                "tokens": tokenizer.encode(text, add_special_tokens=False),
                "page": raw.get("page"),
                "bbox": list(raw.get("bbox") or []),
                "provenance": raw.get("provenance") or {},
            }
        )
    return paragraphs


def qasper_paragraphs(paper_id: str, paper: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    position = 0
    for section_index, section in enumerate(paper.get("full_text") or []):
        section_name = normalize_text(section.get("section_name"))
        path = [item.strip() for item in section_name.split(" ::: ") if item.strip()]
        for paragraph_index, raw_text in enumerate(section.get("paragraphs") or []):
            text = normalize_text(raw_text)
            if not text:
                continue
            paragraphs.append(
                {
                    "id": f"{paper_id}::s{section_index:03d}::p{paragraph_index:03d}",
                    "paper_id": paper_id,
                    "position": position,
                    "section_id": f"{paper_id}::s{section_index:03d}",
                    "section_path": path,
                    "text": text,
                    "tokens": tokenizer.encode(text, add_special_tokens=False),
                    "page": None,
                    "bbox": [],
                    "provenance": {
                        "source_file": "qasper-test-v0.3.json",
                        "source_section_index": section_index,
                        "source_paragraph_index": paragraph_index,
                    },
                }
            )
            position += 1
    return paragraphs


def canonical_objects(document: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    result = []
    for raw in document.get("objects") or []:
        kind = str(raw.get("object_type") or "").lower()
        text = normalize_text(raw.get("caption") or raw.get("object_text") or raw.get("formula_latex"))
        if not text:
            continue
        result.append(
            {
                "id": raw["object_id"],
                "kind": kind,
                "text": text,
                "tokens": tokenizer.encode(text, add_special_tokens=False),
                "page": raw.get("page"),
                "bbox": list(raw.get("bbox") or []),
                "provenance": raw.get("provenance") or {},
            }
        )
    return result


def qasper_objects(paper_id: str, paper: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    result = []
    for index, raw in enumerate(paper.get("figures_and_tables") or []):
        text = normalize_text(raw.get("caption"))
        if not text:
            continue
        label = object_label(text)
        kind = label[0] if label else ("table" if "table" in str(raw.get("file") or "").lower() else "figure")
        result.append(
            {
                "id": f"{paper_id}::figure_or_table::{index:03d}",
                "kind": kind,
                "text": text,
                "tokens": tokenizer.encode(text, add_special_tokens=False),
                "page": None,
                "bbox": [],
                "provenance": {
                    "source_file": "qasper-test-v0.3.json",
                    "source_figure_or_table_index": index,
                    "asset_file": raw.get("file"),
                },
            }
        )
    return result


def normalize_label_kind(value: str) -> str:
    value = value.lower()
    if value.startswith("fig"):
        return "figure"
    if value.startswith("tab"):
        return "table"
    return "formula"


def object_label(text: str) -> tuple[str, str] | None:
    match = LABEL_RE.search(text)
    if not match:
        return None
    return normalize_label_kind(match.group(1)), match.group(2).lower()


def referenced_labels(text: str) -> set[tuple[str, str]]:
    return {(normalize_label_kind(match.group(1)), match.group(2).lower()) for match in LABEL_RE.finditer(text)}


def bbox_vertical_gap(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return float("inf")
    if left[3] < right[1]:
        return float(right[1] - left[3])
    if right[3] < left[1]:
        return float(left[1] - right[3])
    return 0.0


def linked_objects(anchor: dict[str, Any], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = referenced_labels(anchor["text"])
    linked: list[dict[str, Any]] = []
    for obj in objects:
        label = object_label(obj["text"])
        explicit = label is not None and label in labels
        near_formula = (
            obj["kind"] == "formula"
            and anchor.get("page") is not None
            and anchor.get("page") == obj.get("page")
            and bool(FORMULA_CUE_RE.search(anchor["text"]))
            and bbox_vertical_gap(anchor.get("bbox") or [], obj.get("bbox") or []) <= 120.0
        )
        if explicit or near_formula:
            linked.append(obj)
    seen: set[str] = set()
    return [obj for obj in linked if not (obj["id"] in seen or seen.add(obj["id"]))]


def provenance_complete(value: dict[str, Any]) -> bool:
    return bool(value and value.get("source_file"))


def build_fixed_units(
    paper_id: str, paragraphs: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    usable = [paragraph for paragraph in paragraphs if paragraph["tokens"]]
    if not usable:
        return []
    starts, ends, stream = [], [], []
    for paragraph in usable:
        starts.append(len(stream))
        stream.extend(paragraph["tokens"])
        ends.append(len(stream))
    units = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for unit_index, window_start in enumerate(range(0, len(stream), step)):
        window_end = min(window_start + CHUNK_SIZE, len(stream))
        first = max(0, bisect.bisect_right(ends, window_start))
        segments: dict[str, list[int]] = {}
        members = []
        member_index = first
        while member_index < len(usable) and starts[member_index] < window_end:
            paragraph = usable[member_index]
            local_start = max(0, window_start - starts[member_index])
            local_end = min(len(paragraph["tokens"]), window_end - starts[member_index])
            if local_end > local_start:
                segments[paragraph["id"]] = [local_start, local_end, len(paragraph["tokens"])]
                members.append(paragraph)
            member_index += 1
        complete_ids = [pid for pid, (a, b, total) in segments.items() if a == 0 and b == total]
        units.append(
            {
                "unit_id": f"{paper_id}::B1::{unit_index:05d}",
                "paper_id": paper_id,
                "view": "B1",
                "text": tokenizer.decode(stream[window_start:window_end], skip_special_tokens=True),
                "token_count": window_end - window_start,
                "atomic_ids": complete_ids,
                "segments": segments,
                "section_ids": list(dict.fromkeys(p.get("section_id") for p in members if p.get("section_id"))),
                "section_paths": [p.get("section_path") or [] for p in members],
                "pages": sorted({p["page"] for p in members if p.get("page") is not None}),
                "bbox_complete": all(len(p.get("bbox") or []) == 4 for p in members),
                "page_complete": all(p.get("page") is not None for p in members),
                "provenance_complete": all(provenance_complete(p.get("provenance") or {}) for p in members),
                "section_path_complete": all(bool(p.get("section_path")) for p in members),
                "object_ids": [],
                "support_ids": [],
            }
        )
        if window_end == len(stream):
            break
    return units


def section_prefix(paragraph: dict[str, Any]) -> str:
    path = " > ".join(paragraph.get("section_path") or [])
    return f"[Section] {path}\n" if path else ""


def build_atomic_units(paper_id: str, paragraphs: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    units = []
    for index, paragraph in enumerate(paragraphs):
        rendered_text = f"{section_prefix(paragraph)}[Paragraph] {paragraph['text']}"
        units.append(
            {
                "unit_id": f"{paper_id}::B2::{index:05d}",
                "paper_id": paper_id,
                "view": "B2",
                "text": rendered_text,
                "token_count": len(tokenizer.encode(rendered_text, add_special_tokens=False)),
                "atomic_ids": [paragraph["id"]],
                "segments": {paragraph["id"]: [0, len(paragraph["tokens"]), len(paragraph["tokens"])]},
                "section_ids": [paragraph["section_id"]] if paragraph.get("section_id") else [],
                "section_paths": [paragraph.get("section_path") or []],
                "pages": [paragraph["page"]] if paragraph.get("page") is not None else [],
                "bbox_complete": len(paragraph.get("bbox") or []) == 4,
                "page_complete": paragraph.get("page") is not None,
                "provenance_complete": provenance_complete(paragraph.get("provenance") or {}),
                "section_path_complete": bool(paragraph.get("section_path")),
                "object_ids": [],
                "support_ids": [],
            }
        )
    return units


def build_package_units(
    paper_id: str,
    paragraphs: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paragraph in paragraphs:
        section_key = str(paragraph.get("section_id") or f"unsectioned::{paragraph['id']}")
        paragraph["_package_section_key"] = section_key
        by_section[section_key].append(paragraph)
    section_positions = {
        paragraph["id"]: index
        for members in by_section.values()
        for index, paragraph in enumerate(members)
    }
    units = []
    for index, anchor in enumerate(paragraphs):
        section_members = by_section[anchor["_package_section_key"]]
        position = section_positions[anchor["id"]]
        members = []
        if position > 0:
            members.append(section_members[position - 1])
        members.append(anchor)
        if position + 1 < len(section_members):
            members.append(section_members[position + 1])
        members = list({paragraph["id"]: paragraph for paragraph in members}.values())
        members.sort(key=lambda paragraph: paragraph["position"])
        linked = linked_objects(anchor, objects)
        support_ids = [paragraph["id"] for paragraph in members if paragraph["id"] != anchor["id"]]
        rendered = [section_prefix(anchor).rstrip(), *(
            f"[{'Anchor' if paragraph['id'] == anchor['id'] else 'Support'}] {paragraph['text']}"
            for paragraph in members
        )]
        rendered.extend(f"[Object:{obj['kind']}] {obj['text']}" for obj in linked)
        text = "\n".join(item for item in rendered if item)
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        all_entities = members + linked
        units.append(
            {
                "unit_id": f"{paper_id}::B3::{index:05d}",
                "paper_id": paper_id,
                "view": "B3",
                "text": text,
                "token_count": token_count,
                "atomic_ids": [paragraph["id"] for paragraph in members] + [obj["id"] for obj in linked],
                "segments": {
                    paragraph["id"]: [0, len(paragraph["tokens"]), len(paragraph["tokens"])]
                    for paragraph in members
                },
                "anchor_id": anchor["id"],
                "support_ids": support_ids,
                "object_ids": [obj["id"] for obj in linked],
                "object_kinds": [obj["kind"] for obj in linked],
                "section_ids": [anchor["section_id"]] if anchor.get("section_id") else [],
                "section_paths": [anchor.get("section_path") or []],
                "pages": sorted({entity["page"] for entity in all_entities if entity.get("page") is not None}),
                "bbox_complete": all(len(entity.get("bbox") or []) == 4 for entity in all_entities),
                "page_complete": all(entity.get("page") is not None for entity in all_entities),
                "provenance_complete": all(provenance_complete(entity.get("provenance") or {}) for entity in all_entities),
                "section_path_complete": bool(anchor.get("section_path")),
            }
        )
    return units


def summarize_units(view: str, units: Iterable[dict[str, Any]], paper_count: int, elapsed: float) -> dict[str, Any]:
    rows = list(units)
    tokens = [int(row["token_count"]) for row in rows]
    object_counts = Counter(kind for row in rows for kind in row.get("object_kinds") or [])
    return {
        "view": view,
        "unit_count": len(rows),
        "mean_units_per_paper": len(rows) / paper_count if paper_count else 0.0,
        "token_length": token_summary(tokens),
        "provenance_complete_rate": statistics.fmean(row["provenance_complete"] for row in rows) if rows else 0.0,
        "section_path_complete_rate": statistics.fmean(row["section_path_complete"] for row in rows) if rows else 0.0,
        "page_complete_rate": statistics.fmean(row["page_complete"] for row in rows) if rows else 0.0,
        "bbox_complete_rate": statistics.fmean(row["bbox_complete"] for row in rows) if rows else 0.0,
        "object_context_unit_rate": statistics.fmean(bool(row.get("object_ids")) for row in rows) if rows else 0.0,
        "object_context_counts": {
            "figure": int(object_counts["figure"]),
            "table": int(object_counts["table"]),
            "formula": int(object_counts["formula"]),
        },
        "mean_support_paragraphs": statistics.fmean(len(row.get("support_ids") or []) for row in rows) if rows else 0.0,
        "build_seconds": elapsed,
        "logical_jsonl_size_bytes": sum(json_bytes(row) for row in rows),
        "logical_size_note": "Materialized UTF-8 JSONL byte count computed in memory; RAG views were not persisted as extra files.",
    }


def build_s2k_summary(canonical_path: Path, tokenizer: Any) -> dict[str, Any]:
    documents = pq.read_table(canonical_path).to_pylist()
    accumulators: dict[str, list[dict[str, Any]]] = {"B1": [], "B2": [], "B3": []}
    elapsed = Counter()
    for document in documents:
        paragraphs = canonical_paragraphs(document, tokenizer)
        objects = canonical_objects(document, tokenizer)
        for view, builder in (
            ("B1", lambda: build_fixed_units(document["paper_id"], paragraphs, tokenizer)),
            ("B2", lambda: build_atomic_units(document["paper_id"], paragraphs, tokenizer)),
            ("B3", lambda: build_package_units(document["paper_id"], paragraphs, objects, tokenizer)),
        ):
            started = time.perf_counter()
            accumulators[view].extend(builder())
            elapsed[view] += time.perf_counter() - started
    summaries = {
        view: summarize_units(view, rows, len(documents), float(elapsed[view]))
        for view, rows in accumulators.items()
    }
    return {
        "scope": "S2K_engineering_audit_only",
        "paper_count": len(documents),
        "canonical_path": canonical_path.as_posix(),
        "canonical_sha256": sha256_file(canonical_path),
        "s2k_membership_sha256": S2K_MEMBERSHIP_SHA256,
        "views": summaries,
        "formal_retrieval_metrics_computed": False,
        "synthetic_queries_used": False,
    }


def load_qasper_archive(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("qasper-test-v0.3.json"))
        return json.load(archive.extractfile(member))


def load_qasper_gold(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == QASPER_SPLIT and row["retrieval_eligible"]]
    if len(selected) != 1309:
        raise ValueError(f"Frozen QASPER test cohort changed: expected 1309, got {len(selected)}")
    return selected


def verify_mapping_item(item: dict[str, Any], atomic_text: dict[str, str]) -> tuple[bool, str, list[list[str]]]:
    paragraph_ids = list(item.get("paragraph_ids") or [])
    caption_ids = list(item.get("caption_ids") or [])
    all_ids = paragraph_ids + caption_ids
    if not all_ids:
        return False, "no_explicit_paragraph_or_caption_mapping", []
    missing = [atomic_id for atomic_id in all_ids if atomic_id not in atomic_text]
    if missing:
        return False, "mapped_atomic_id_absent_from_frozen_source", []
    evidence = normalize_text(item.get("original_evidence"))
    alternatives: list[list[str]] = []
    if paragraph_ids:
        joined = normalize_text(" ".join(atomic_text[atomic_id] for atomic_id in paragraph_ids))
        method = item.get("match_method")
        valid = (
            (method == "normalized_exact" and evidence == joined)
            or (method == "evidence_in_paragraph" and evidence in joined)
            or (method == "paragraph_in_evidence" and joined in evidence)
            or (str(method).startswith("adjacent_") and (evidence in joined or joined in evidence))
            or method == "high_confidence_fuzzy"
        )
        if not valid:
            return False, "frozen_text_mapping_failed_revalidation", []
        alternatives.append(paragraph_ids)
    for caption_id in caption_ids:
        caption = normalize_text(atomic_text[caption_id])
        if not (evidence == caption or (len(evidence) >= 20 and evidence in caption) or (len(caption) >= 20 and caption in evidence)):
            return False, "frozen_caption_mapping_failed_revalidation", []
        alternatives.append([caption_id])
    return True, "mapped_and_revalidated", alternatives


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def map_public_gold(
    gold_rows: list[dict[str, Any]],
    paragraphs: dict[str, list[dict[str, Any]]],
    objects: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    atomic_text = {
        item["id"]: item["text"]
        for collection in (paragraphs, objects)
        for rows in collection.values()
        for item in rows
    }
    mapping_rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    method_counts = Counter()
    for row in gold_rows:
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in row.get("mapping") or []:
            valid, reason, alternatives = verify_mapping_item(item, atomic_text)
            annotation_index = int(item["annotation_index"])
            evidence_index = int(item["evidence_index"])
            record = {
                "dataset": "QASPER",
                "version": QASPER_VERSION,
                "split": QASPER_SPLIT,
                "paper_id": row["paper_id"],
                "question_id": row["question_id"],
                "annotation_index": annotation_index,
                "evidence_index": evidence_index,
                "evidence_sha256": hashlib.sha256(normalize_text(item.get("original_evidence")).encode("utf-8")).hexdigest(),
                "paragraph_ids": json.dumps(item.get("paragraph_ids") or [], ensure_ascii=False),
                "caption_ids": json.dumps(item.get("caption_ids") or [], ensure_ascii=False),
                "atomic_alternatives": json.dumps(alternatives, ensure_ascii=False),
                "match_method": item.get("match_method"),
                "match_score": item.get("match_score"),
                "status": "mapped" if valid else "unmatched",
                "reason": reason,
            }
            mapping_rows.append(record)
            method_counts[str(item.get("match_method"))] += 1
            annotations[annotation_index].append({"valid": valid, "alternatives": alternatives})
            if not valid:
                unmatched.append(record)
        complete_sets = []
        for annotation_index in sorted(annotations):
            items = annotations[annotation_index]
            if items and all(item["valid"] for item in items):
                complete_sets.append(
                    {
                        "annotation_index": annotation_index,
                        "evidence_elements": [item["alternatives"] for item in items],
                    }
                )
        if complete_sets:
            evaluation_rows.append(row | {"complete_gold_sets": complete_sets})
    fields = [
        "dataset", "version", "split", "paper_id", "question_id", "annotation_index", "evidence_index",
        "evidence_sha256", "paragraph_ids", "caption_ids", "atomic_alternatives", "match_method", "match_score",
        "status", "reason",
    ]
    write_csv(output_dir / "public_gold_mapping.csv", mapping_rows, fields)
    if unmatched:
        write_csv(output_dir / "unmatched_gold.csv", unmatched, fields)
    mapped_count = len(mapping_rows) - len(unmatched)
    mapping_rate = mapped_count / len(mapping_rows) if mapping_rows else 0.0
    summary = {
        "dataset": "QASPER",
        "version": QASPER_VERSION,
        "split": QASPER_SPLIT,
        "evaluation_role": "reused_formal_public_gold_not_blind",
        "frozen_eligible_question_count": len(gold_rows),
        "formal_question_count": len(evaluation_rows),
        "excluded_questions_without_complete_mapped_annotation_set": len(gold_rows) - len(evaluation_rows),
        "total_gold_evidence_strings": len(mapping_rows),
        "mapped_gold_evidence_strings": mapped_count,
        "unmatched_gold_evidence_strings": len(unmatched),
        "mapping_rate": mapping_rate,
        "mapping_gate": MAPPING_GATE,
        "mapping_gate_passed": mapping_rate >= MAPPING_GATE,
        "match_method_counts": dict(sorted(method_counts.items())),
        "gold_semantics": "Each annotator evidence list is a complete set; a mapped evidence string may have alternative atomic locations. CES succeeds if any complete annotator set is recovered.",
    }
    return evaluation_rows, summary


def title_tokens(value: str) -> set[str]:
    return set(normalize_title(value).split())


def leakage_audit(
    qasper_papers: dict[str, Any], canonical_path: Path, prep_db: Path
) -> dict[str, Any]:
    canonical = pq.read_table(canonical_path, columns=["paper_id", "source_id", "title"]).to_pylist()
    local_ids = {str(row.get("paper_id")) for row in canonical} | {str(row.get("source_id")) for row in canonical}
    benchmark_ids = set(qasper_papers)
    source_id_overlap = sorted(benchmark_ids & local_ids)
    local_titles = defaultdict(list)
    for row in canonical:
        local_titles[normalize_title(row.get("title"))].append(row["paper_id"])
    exact_title = []
    near_title = []
    for paper_id, paper in qasper_papers.items():
        normalized = normalize_title(paper.get("title"))
        if normalized and normalized in local_titles:
            exact_title.append({"benchmark_paper_id": paper_id, "local_paper_ids": local_titles[normalized]})
            continue
        left = title_tokens(normalized)
        if len(left) < 5:
            continue
        best_score, best_id = 0.0, None
        for row in canonical:
            right = title_tokens(row.get("title") or "")
            if len(right) < 5:
                continue
            score = len(left & right) / len(left | right)
            if score > best_score:
                best_score, best_id = score, row["paper_id"]
        if best_score >= 0.90:
            near_title.append({"benchmark_paper_id": paper_id, "local_paper_id": best_id, "title_token_jaccard": best_score})
    prep_counts = None
    if prep_db.exists():
        connection = sqlite3.connect(prep_db)
        prep_counts = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(public_eval_source_overlap),0), COALESCE(SUM(public_eval_title_overlap),0) "
            "FROM source_pool_members WHERE pool_id='S2K'"
        ).fetchone()
        connection.close()
    return {
        "benchmark_paper_count": len(benchmark_ids),
        "canonical_s2k_source_id_overlap_count": len(source_id_overlap),
        "canonical_s2k_source_id_overlaps": source_id_overlap,
        "canonical_s2k_normalized_exact_title_overlap_count": len(exact_title),
        "canonical_s2k_near_title_overlap_count": len(near_title),
        "near_title_rule": "token Jaccard >=0.90 with at least 5 normalized title tokens",
        "near_title_matches": near_title,
        "frozen_prep_s2k_audit": {
            "member_count": int(prep_counts[0]),
            "public_eval_source_overlap": int(prep_counts[1]),
            "public_eval_title_overlap": int(prep_counts[2]),
        } if prep_counts else None,
        "historical_benchmark_use": [
            "QASPER validation was used for earlier retrieval/SEP configuration development.",
            "QASPER test was consumed by an earlier frozen retrieval experiment.",
            "This run is formal public-gold reuse, not blind evaluation and not independent confirmation.",
        ],
        "gold_used_for_current_parameter_tuning": False,
    }


@torch.inference_mode()
def encode_texts(
    texts: list[str], tokenizer: Any, model: Any, device: torch.device, batch_size: int, query: bool = False
) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        if query:
            batch = [QUERY_PREFIX + text for text in batch]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        output = model(**inputs).last_hidden_state[:, 0]
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        vectors.append(output.float().cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, model.config.hidden_size), dtype=np.float32)


def merge_intervals(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    left, right = ordered[0]
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def recovered_ids(units: list[dict[str, Any]], view: str) -> set[str]:
    if view != "B1":
        return {atomic_id for unit in units for atomic_id in unit["atomic_ids"]}
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    totals: dict[str, int] = {}
    for unit in units:
        for atomic_id, (start, end, total) in unit["segments"].items():
            intervals[atomic_id].append((int(start), int(end)))
            totals[atomic_id] = int(total)
    return {
        atomic_id
        for atomic_id, spans in intervals.items()
        if totals[atomic_id] > 0 and merge_intervals(spans) >= totals[atomic_id]
    }


def gold_scores(gold_sets: list[dict[str, Any]], recovered: set[str]) -> tuple[float, float, set[str]]:
    best_recall = 0.0
    complete = 0.0
    recovered_gold_ids: set[str] = set()
    for annotation in gold_sets:
        elements = annotation["evidence_elements"]
        hits = 0
        for alternatives in elements:
            element_hit = False
            for alternative in alternatives:
                required = set(alternative)
                if required and required <= recovered:
                    element_hit = True
                    recovered_gold_ids.update(required)
                    break
            hits += int(element_hit)
        recall = hits / len(elements) if elements else 0.0
        best_recall = max(best_recall, recall)
        complete = max(complete, float(recall == 1.0 and bool(elements)))
    return best_recall, complete, recovered_gold_ids


def evaluate_ranking(row: dict[str, Any], ranked: list[dict[str, Any]], view: str) -> dict[str, Any]:
    top1 = ranked[:1]
    top5 = ranked[:TOP_K]
    top10 = ranked[:SUPPLEMENTARY_TOP_K]
    recovered1 = recovered_ids(top1, view)
    recovered5 = recovered_ids(top5, view)
    recovered10 = recovered_ids(top10, view)
    recall1, _, _ = gold_scores(row["complete_gold_sets"], recovered1)
    recall5, ces5, gold5 = gold_scores(row["complete_gold_sets"], recovered5)
    recall10, _, _ = gold_scores(row["complete_gold_sets"], recovered10)
    hit1 = float(recall1 > 0)
    hit5 = float(recall5 > 0)
    first_rank = 0
    for rank in range(1, len(ranked) + 1):
        recall, _, _ = gold_scores(row["complete_gold_sets"], recovered_ids(ranked[:rank], view))
        if recall > 0:
            first_rank = rank
            break
    budget_units, budget_tokens = [], 0
    for unit in ranked:
        count = int(unit["token_count"])
        if budget_tokens + count > TOKEN_BUDGET:
            break
        budget_units.append(unit)
        budget_tokens += count
    budget_recovered = recovered_ids(budget_units, view)
    budget_recall, budget_ces, budget_gold = gold_scores(row["complete_gold_sets"], budget_recovered)
    return {
        "hit@1": hit1,
        "hit@5": hit5,
        "recall@5": recall5,
        "recall@10": recall10,
        "ces@5": ces5,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "budget_recall@1024": budget_recall,
        "budget_ces@1024": budget_ces,
        "top5_recovered_gold_ids": sorted(gold5),
        "budget_recovered_gold_ids": sorted(budget_gold),
        "budget_units": budget_units,
        "budget_tokens": budget_tokens,
    }


def build_public_views(
    papers: dict[str, Any], tokenizer: Any, needed: set[str]
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    paragraphs = {paper_id: qasper_paragraphs(paper_id, papers[paper_id], tokenizer) for paper_id in sorted(needed)}
    objects = {paper_id: qasper_objects(paper_id, papers[paper_id], tokenizer) for paper_id in sorted(needed)}
    views = {
        "B1": {paper_id: build_fixed_units(paper_id, paragraphs[paper_id], tokenizer) for paper_id in sorted(needed)},
        "B2": {paper_id: build_atomic_units(paper_id, paragraphs[paper_id], tokenizer) for paper_id in sorted(needed)},
        "B3": {paper_id: build_package_units(paper_id, paragraphs[paper_id], objects[paper_id], tokenizer) for paper_id in sorted(needed)},
    }
    return views, paragraphs, objects


def run_retrieval(
    rows: list[dict[str, Any]],
    views: dict[str, dict[str, list[dict[str, Any]]]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_vectors = encode_texts([row["question"] for row in rows], tokenizer, model, device, batch_size, query=True)
    result_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    view_details = {}
    metric_names = [
        "ces@5", "budget_ces@1024", "hit@5", "recall@5", "mrr", "budget_recall@1024", "hit@1", "recall@10",
    ]
    for view in ("B1", "B2", "B3"):
        flat_units = [unit for paper_id in sorted(views[view]) for unit in views[view][paper_id]]
        vectors = encode_texts([unit["text"] for unit in flat_units], tokenizer, model, device, batch_size)
        paper_vectors = {}
        cursor = 0
        for paper_id in sorted(views[view]):
            count = len(views[view][paper_id])
            paper_vectors[paper_id] = vectors[cursor:cursor + count]
            cursor += count
        per_metric: dict[str, list[float]] = defaultdict(list)
        for index, row in enumerate(rows):
            units = views[view][row["paper_id"]]
            scores = paper_vectors[row["paper_id"]] @ query_vectors[index]
            order = np.argsort(-scores, kind="mergesort")
            ranked = [units[int(item)] for item in order]
            evaluated = evaluate_ranking(row, ranked, view)
            for metric in metric_names:
                per_metric[metric].append(float(evaluated[metric]))
            top = ranked[:SUPPLEMENTARY_TOP_K]
            result_rows.append(
                {
                    "dataset": "QASPER",
                    "version": QASPER_VERSION,
                    "split": QASPER_SPLIT,
                    "evaluation_role": "reused_formal_public_gold_not_blind",
                    "question_id": row["question_id"],
                    "paper_id": row["paper_id"],
                    "question": row["question"],
                    "view": view,
                    "gold_sets_json": json.dumps(row["complete_gold_sets"], ensure_ascii=False, separators=(",", ":")),
                    "top_unit_ids": [unit["unit_id"] for unit in top],
                    "top_scores": [float(scores[int(item)]) for item in order[:SUPPLEMENTARY_TOP_K]],
                    "top_token_counts": [int(unit["token_count"]) for unit in top],
                    "top_atomic_ids_json": json.dumps([unit["atomic_ids"] for unit in top], ensure_ascii=False),
                    "top5_recovered_gold_ids": evaluated["top5_recovered_gold_ids"],
                    "budget_unit_ids": [unit["unit_id"] for unit in evaluated["budget_units"]],
                    "budget_tokens": int(evaluated["budget_tokens"]),
                    "budget_recovered_gold_ids": evaluated["budget_recovered_gold_ids"],
                    **{metric.replace("@", "_at_"): float(evaluated[metric]) for metric in metric_names},
                }
            )
        aggregates = {metric: float(statistics.fmean(values)) for metric, values in per_metric.items()}
        aggregates["mean_top5_tokens"] = float(statistics.fmean(sum(r["top_token_counts"][:5]) for r in result_rows if r["view"] == view))
        aggregates["mean_budget_tokens"] = float(statistics.fmean(r["budget_tokens"] for r in result_rows if r["view"] == view))
        metric_rows.append({"dataset": "QASPER", "view": view, "question_count": len(rows), **aggregates})
        view_details[view] = {
            "unit_count": len(flat_units),
            "token_length": token_summary([unit["token_count"] for unit in flat_units]),
            "object_context_unit_rate": statistics.fmean(bool(unit.get("object_ids")) for unit in flat_units),
            "per_question": dict(per_metric),
        }
        del vectors, paper_vectors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result_rows, metric_rows, view_details


def bootstrap_statistics(view_details: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = ("ces@5", "budget_ces@1024")
    views = ("B1", "B2", "B3")
    n = len(view_details["B1"]["per_question"]["ces@5"])
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n), dtype=np.int32)
    bootstrap_means: dict[tuple[str, str], np.ndarray] = {}
    for metric in metrics:
        for view in views:
            values = np.asarray(view_details[view]["per_question"][metric], dtype=np.float64)
            means = values[indices].mean(axis=1)
            bootstrap_means[(view, metric)] = means
            rows.append(
                {
                    "row_type": "view_estimate",
                    "metric": metric,
                    "view_a": view,
                    "view_b": "",
                    "estimate": float(values.mean()),
                    "ci_low": float(np.quantile(means, 0.025)),
                    "ci_high": float(np.quantile(means, 0.975)),
                    "raw_p": "",
                    "holm_p": "",
                    "significant_0_05": "",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    comparisons = []
    for metric in metrics:
        for left, right in (("B2", "B1"), ("B3", "B1"), ("B3", "B2")):
            left_values = np.asarray(view_details[left]["per_question"][metric], dtype=np.float64)
            right_values = np.asarray(view_details[right]["per_question"][metric], dtype=np.float64)
            difference = left_values - right_values
            boot = bootstrap_means[(left, metric)] - bootstrap_means[(right, metric)]
            negative = (np.count_nonzero(boot <= 0.0) + 1) / (BOOTSTRAP_REPLICATES + 1)
            positive = (np.count_nonzero(boot >= 0.0) + 1) / (BOOTSTRAP_REPLICATES + 1)
            raw_p = min(1.0, 2.0 * min(negative, positive))
            comparisons.append(
                {
                    "row_type": "pairwise_difference",
                    "metric": metric,
                    "view_a": left,
                    "view_b": right,
                    "estimate": float(difference.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "raw_p": raw_p,
                    "holm_p": None,
                    "significant_0_05": None,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    ordered = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["raw_p"])
    running = 0.0
    count = len(comparisons)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, comparisons[index]["raw_p"] * (count - rank))
        running = max(running, adjusted)
        comparisons[index]["holm_p"] = running
        comparisons[index]["significant_0_05"] = running < 0.05
    rows.extend(comparisons)
    return rows


def write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files)
    (output_dir / "checksums.sha256").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=Path("aicorpus-derived/experiments/corpus_v1/pilot2k/run_02/canonical_documents.parquet"))
    parser.add_argument("--qasper-archive", type=Path, default=Path("aicorpus-data/public_benchmarks/qasper-test-and-evaluator-v0.3.tgz"))
    parser.add_argument("--qasper-gold", type=Path, default=Path("aicorpus-derived/qasper/qasper_retrieval_gold.jsonl"))
    parser.add_argument("--prep-db", type=Path, default=Path("aicorpus-derived/e2_prep_v25/e2_prep_v25.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("aicorpus-derived/experiments/corpus_v1/rag"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen FP16 embedding run requires CUDA")
    input_hashes = {
        "canonical_documents.parquet": sha256_file(args.canonical),
        "qasper-test-and-evaluator-v0.3.tgz": sha256_file(args.qasper_archive),
        "qasper_retrieval_gold.jsonl": sha256_file(args.qasper_gold),
    }
    expected = {
        "canonical_documents.parquet": CANONICAL_SHA256,
        "qasper-test-and-evaluator-v0.3.tgz": QASPER_ARCHIVE_SHA256,
        "qasper_retrieval_gold.jsonl": QASPER_GOLD_SHA256,
    }
    if input_hashes != expected:
        raise ValueError(f"Frozen input hash mismatch: {input_hashes}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    s2k_summary = build_s2k_summary(args.canonical, tokenizer)
    (args.output_dir / "s2k_engineering_summary.json").write_text(
        json.dumps(s2k_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qasper_papers = load_qasper_archive(args.qasper_archive)
    gold_rows = load_qasper_gold(args.qasper_gold)
    needed = {row["paper_id"] for row in gold_rows}
    views, paragraphs, objects = build_public_views(qasper_papers, tokenizer, needed)
    evaluation_rows, mapping_summary = map_public_gold(gold_rows, paragraphs, objects, args.output_dir)
    leakage = leakage_audit(qasper_papers, args.canonical, args.prep_db)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_formal_retrieval",
        "seed": SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "inputs": {
            "canonical": args.canonical.as_posix(),
            "s2k_membership_sha256": S2K_MEMBERSHIP_SHA256,
            "qasper_archive": args.qasper_archive.as_posix(),
            "qasper_gold": args.qasper_gold.as_posix(),
            "sha256": input_hashes,
        },
        "public_gold": {
            "dataset": "QASPER",
            "version": QASPER_VERSION,
            "split": QASPER_SPLIT,
            "source_paper_ids": sorted(needed),
            "selection": "historical frozen retrieval_eligible test cohort; formal comparison requires >=1 fully mapped complete annotator evidence set",
            "evaluation_role": "reused_formal_public_gold_not_blind",
            "peerqa_role": "not_used_no_local_executable_paper_level_evidence_mapping",
        },
        "retrieval": {
            "scope": "paper_scoped",
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dtype": "float16",
            "query_prefix": QUERY_PREFIX,
            "passage_prefix": None,
            "pooling": "CLS",
            "normalized_embeddings": True,
            "similarity": "dot_product",
            "embedding_max_tokens": 512,
            "top_k": TOP_K,
            "supplementary_top_k": SUPPLEMENTARY_TOP_K,
            "token_budget": TOKEN_BUDGET,
            "budget_rule": "ranked prefix; stop before first whole unit that would exceed 1024; no cross-unit dedup",
        },
        "views": {
            "B1": {"name": "fixed_chunks", "chunk_tokens": CHUNK_SIZE, "overlap_tokens": CHUNK_OVERLAP, "source": "body paragraphs", "gold_coverage": "100% union of paragraph token intervals"},
            "B2": {"name": "section_aware_atomic", "unit": "atomic paragraph", "section_path_in_retrieval_text": True, "cross_section_merge": False},
            "B3": {"name": "one_hop_evidence_package", "anchor": "atomic paragraph", "supports": "same-section previous and next valid paragraphs", "max_supports": 2, "object_link": "explicit figure/table label; or same-page equation cue plus bbox vertical gap <=120 for formulas", "within_package_dedup": True},
        },
        "metrics": {
            "primary": ["CES@5", "BudgetCES@1024"],
            "auxiliary": ["Hit@5", "Recall@5", "MRR", "BudgetRecall@1024", "Hit@1", "Recall@10"],
            "multi_reference_rule": "maximum recall across complete annotator evidence sets; CES if any complete set is recovered",
        },
        "mapping_gate": MAPPING_GATE,
        "parameter_tuning_in_this_run": False,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if not mapping_summary["mapping_gate_passed"]:
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "status": "BLOCKER",
            "reason": "public_gold_mapping_below_95_percent",
            "mapping": mapping_summary,
            "leakage_audit": leakage,
            "formal_retrieval_run": False,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_checksums(args.output_dir)
        raise RuntimeError("BLOCKER: public gold mapping coverage below 95%; formal comparison was not run")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True).to(device=device, dtype=torch.float16).eval()
    retrieval_started = time.time()
    result_rows, metrics_rows, view_details = run_retrieval(
        evaluation_rows, views, tokenizer, model, device, args.batch_size
    )
    retrieval_seconds = time.time() - retrieval_started
    pq.write_table(pa.Table.from_pylist(result_rows), args.output_dir / "retrieval_results.parquet", compression="zstd")
    metric_fields = [
        "dataset", "view", "question_count", "ces@5", "budget_ces@1024", "hit@5", "recall@5", "mrr",
        "budget_recall@1024", "hit@1", "recall@10", "mean_top5_tokens", "mean_budget_tokens",
    ]
    write_csv(args.output_dir / "metrics.csv", metrics_rows, metric_fields)
    bootstrap_rows = bootstrap_statistics(view_details)
    bootstrap_fields = [
        "row_type", "metric", "view_a", "view_b", "estimate", "ci_low", "ci_high", "raw_p", "holm_p",
        "significant_0_05", "bootstrap_replicates",
    ]
    write_csv(args.output_dir / "bootstrap.csv", bootstrap_rows, bootstrap_fields)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "RAG_RESULT_READY",
        "s2k_engineering": s2k_summary,
        "public_gold_mapping": mapping_summary,
        "leakage_audit": leakage,
        "public_corpus": {
            view: {key: value for key, value in details.items() if key != "per_question"}
            for view, details in view_details.items()
        },
        "metrics": metrics_rows,
        "bootstrap": bootstrap_rows,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pyarrow": pa.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "dtype": "float16",
            "batch_size": args.batch_size,
        },
        "runtime": {
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": time.time() - started,
        },
        "blockers": [],
        "interpretation_boundary": "Observed retrieval differences are specific to this frozen QASPER setup and do not by themselves establish a universally optimal corpus construction strategy.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_checksums(args.output_dir)
    print(json.dumps({
        "status": summary["status"],
        "questions": len(evaluation_rows),
        "mapping_rate": mapping_summary["mapping_rate"],
        "metrics": metrics_rows,
        "runtime": summary["runtime"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
