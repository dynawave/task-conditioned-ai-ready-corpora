from __future__ import annotations

"""Development-only Adaptive Chunking 2026 representation feasibility smoke."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.rag import run_rag_experiment as rag  # noqa: E402


OUTPUT = (
    ROOT
    / "aicorpus-derived/experiments/corpus_v1/rag/adaptive_chunking_smoke_v1"
    / "adaptive_chunking_feasibility.json"
)
QASPER_ARCHIVE = ROOT / "aicorpus-data/public_benchmarks/qasper-test-and-evaluator-v0.3.tgz"
QASPER_GOLD = ROOT / "aicorpus-derived/qasper/qasper_retrieval_gold.jsonl"
EXPECTED_SOURCE_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
METHODS = {"our_recurs_600": 600, "our_recurs_1100": 1100}
SEPARATORS = [
    r"(?<=\n)#{1}\s", r"(?<=\n)#{2}\s", r"(?<=\n)#{3}\s", r"(?<=\n)#{4}\s",
    r"(?<=\n)#{5}\s", r"(?<=\n)#{6}\s", r"(?<=\n)\s*\(?[A-Za-z0-9]{1,4}[.)]\s+",
    r"(?<=\n)\s*[-*·•●▪◦‣▸▹○◯‒–—]\s+", r"\n{2,}", r"\n", r"[.!?]\s+", r",\s+", r"\s+", r"",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_text(paragraphs: list[dict[str, Any]]) -> tuple[str, dict[str, tuple[int, int]]]:
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    previous_section: str | None = None
    offset = 0
    for paragraph in paragraphs:
        section_id = str(paragraph["section_id"])
        if section_id != previous_section:
            title = " > ".join(paragraph["section_path"]) or "Untitled section"
            heading = f"## {title}\n\n"
            parts.append(heading)
            offset += len(heading)
            previous_section = section_id
        text = str(paragraph["text"])
        start = offset
        parts.append(text)
        offset += len(text)
        spans[str(paragraph["id"])] = (start, offset)
        parts.append("\n\n")
        offset += 2
    return "".join(parts), spans


def mapping_class(
    item: dict[str, Any],
    atomic_text: dict[str, str],
    paragraph_spans: dict[str, tuple[int, int]],
    chunk_spans: list[tuple[int, int]],
) -> str:
    valid, _reason, alternatives = rag.verify_mapping_item(item, atomic_text)
    if not valid:
        return "unmapped"
    for alternative in alternatives:
        spans = [paragraph_spans[atomic_id] for atomic_id in alternative if atomic_id in paragraph_spans]
        if len(spans) != len(alternative) or not spans:
            continue
        evidence_start = min(start for start, _end in spans)
        evidence_end = max(end for _start, end in spans)
        if any(start <= evidence_start and end >= evidence_end for start, end in chunk_spans):
            return "fully_mapped"
        covering = {
            index
            for index, (start, end) in enumerate(chunk_spans)
            if any(not (end <= evidence_start_i or start >= evidence_end_i) for evidence_start_i, evidence_end_i in spans)
        }
        if len(covering) > 1:
            return "multi_chunk"
    return "unmapped"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-source", type=Path, required=True)
    parser.add_argument("--adaptive-deps", type=Path, required=True)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise RuntimeError(f"Output exists; refuse to overwrite: {OUTPUT}")

    source = args.adaptive_source.resolve()
    deps = args.adaptive_deps.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError(f"Adaptive source commit drift: {commit}")
    sys.path[:0] = [str(deps), str(source / "src")]
    import tiktoken  # type: ignore
    from adaptive_chunking.chunking_utils import count_tokens
    from adaptive_chunking.postprocessing import (
        check_chunk_gaps,
        find_chunks_start_and_end,
        merge_small_chunks_to_neighbours,
    )
    from adaptive_chunking.splitters import RecursiveSplitter

    if sha256(QASPER_ARCHIVE) != rag.QASPER_ARCHIVE_SHA256 or sha256(QASPER_GOLD) != rag.QASPER_GOLD_SHA256:
        raise RuntimeError("Frozen QASPER input hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        rag.MODEL, revision=rag.MODEL_REVISION, local_files_only=True, use_fast=True
    )
    papers = rag.load_qasper_archive(QASPER_ARCHIVE)
    gold_rows = rag.load_qasper_gold(QASPER_GOLD)
    paper_ids = sorted({str(row["paper_id"]) for row in gold_rows})[:10]
    rows_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        if row["paper_id"] in paper_ids:
            rows_by_paper[str(row["paper_id"])].append(row)

    splitters = {
        method: RecursiveSplitter(
            separators=SEPARATORS,
            chunk_size=size,
            chunk_overlap=0,
            is_separator_regex=True,
            attach_separator_to="start",
            length_function=lambda text: count_tokens(text, model="gpt-4o"),
            merging="to_chunk_size",
            merging_order="forward",
        )
        for method, size in METHODS.items()
    }

    paper_records: list[dict[str, Any]] = []
    mapping_totals: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    generation_failures: list[dict[str, str]] = []
    for paper_id in paper_ids:
        paragraphs = rag.qasper_paragraphs(paper_id, papers[paper_id], tokenizer)
        objects = rag.qasper_objects(paper_id, papers[paper_id], tokenizer)
        atomic_text = {item["id"]: item["text"] for item in [*paragraphs, *objects]}
        full_text, paragraph_spans = build_text(paragraphs)
        method_records: dict[str, Any] = {}
        for method, splitter in splitters.items():
            try:
                chunks = splitter.split_text(full_text)
                chunks = merge_small_chunks_to_neighbours(
                    chunks,
                    lambda text: count_tokens(text, model="gpt-4o"),
                    min_limit=100,
                    max_limit=1150,
                    merge_to="next",
                )
                if not chunks or not check_chunk_gaps(chunks, full_text):
                    raise RuntimeError("official chunk gap/coverage check failed")
                chunk_spans = find_chunks_start_and_end(chunks, full_text)
                method_tokens = [int(count_tokens(chunk, model="gpt-4o")) for chunk in chunks]
                bge_tokens = [len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks]
                evidence_counts: Counter[str] = Counter()
                for row in rows_by_paper[paper_id]:
                    for item in row.get("mapping") or []:
                        evidence_counts[mapping_class(item, atomic_text, paragraph_spans, chunk_spans)] += 1
                mapping_totals[method].update(evidence_counts)
                method_records[method] = {
                    "number_of_chunks": len(chunks),
                    "average_chunk_token_length": float(statistics.fmean(bge_tokens)),
                    "total_chunks_tokens": int(sum(bge_tokens)),
                    "official_gpt4o_average_chunk_tokens": float(statistics.fmean(method_tokens)),
                    "official_gpt4o_total_chunk_tokens": int(sum(method_tokens)),
                    "bge_chunks_over_512": int(sum(value > 512 for value in bge_tokens)),
                    "bge_tokens_beyond_512": int(sum(max(0, value - 512) for value in bge_tokens)),
                    "generation": "success",
                    "evidence_mapping": {
                        "fully_mapped_count": int(evidence_counts["fully_mapped"]),
                        "multi_chunk_count": int(evidence_counts["multi_chunk"]),
                        "unmapped_count": int(evidence_counts["unmapped"]),
                        "total": int(sum(evidence_counts.values())),
                    },
                }
            except Exception as error:
                generation_failures.append({"paper_id": paper_id, "method": method, "error": str(error)})
                method_records[method] = {
                    "number_of_chunks": None,
                    "average_chunk_token_length": None,
                    "total_chunks_tokens": None,
                    "generation": "failure",
                    "error": str(error),
                }
        paper_records.append({
            "paper_id": paper_id,
            "body_paragraphs": len(paragraphs),
            "eligible_questions": len(rows_by_paper[paper_id]),
            "methods": method_records,
        })

    aggregate: dict[str, Any] = {}
    for method in METHODS:
        records = [row["methods"][method] for row in paper_records if row["methods"][method]["generation"] == "success"]
        aggregate[method] = {
            "papers_successful": len(records),
            "papers_failed": len(paper_ids) - len(records),
            "total_chunks": int(sum(row["number_of_chunks"] for row in records)),
            "average_chunk_token_length": float(
                sum(row["total_chunks_tokens"] for row in records) / sum(row["number_of_chunks"] for row in records)
            ) if records else None,
            "total_chunks_tokens": int(sum(row["total_chunks_tokens"] for row in records)),
            "bge_chunks_over_512": int(sum(row["bge_chunks_over_512"] for row in records)),
            "bge_tokens_beyond_512": int(sum(row["bge_tokens_beyond_512"] for row in records)),
        }

    mapping_summary = {
        method: {
            "fully_mapped_count": int(counts["fully_mapped"]),
            "multi_chunk_count": int(counts["multi_chunk"]),
            "unmapped_count": int(counts["unmapped"]),
            "total": int(sum(counts.values())),
            "mapping_basis": "frozen RQ1 QASPER atomic alternatives against same body-paragraph source",
        }
        for method, counts in mapping_totals.items()
    }
    all_candidates_pass = not generation_failures and all(aggregate[m]["papers_successful"] == 10 for m in METHODS)
    payload = {
        "experiment": "adaptive_chunking_2026_feasibility_smoke_v1",
        "status": "DEVELOPMENT_ONLY_NOT_FORMAL_RQ1",
        "environment_status": {
            "official_core_candidate_generation": "PASS" if all_candidates_pass else "FAIL",
            "full_adaptive_selection": "NOT_RUN",
            "reason_full_selector_not_run": [
                "QASPER JSON has no PDF page metadata required by the official page candidate",
                "official semantic candidate requires Qwen/Qwen3-Embedding-0.6B and flash-attention",
                "official LLM-regex candidate requires GPT-4o API calls",
                "five-metric selector requires Jina embeddings and coreference mentions not present in frozen RQ1",
                "official package requires Python >=3.11 while frozen RQ1 runtime is Python 3.10",
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tiktoken": tiktoken.__version__,
            "bge_model_unchanged": f"{rag.MODEL}@{rag.MODEL_REVISION}",
            "retrieval_run": False,
            "ces_or_budgetces_run": False,
        },
        "method_version_source": {
            "paper": "Adaptive Chunking: Optimizing Chunking-Method Selection for RAG",
            "paper_version": "LREC 2026 publication; supplementary preprint arXiv:2603.25333v1",
            "official_repository": "https://github.com/ekimetrics/adaptive-chunking",
            "official_commit": commit,
            "package_version": "0.1.0",
            "source_files": {
                "splitters.py_sha256": sha256(source / "src/adaptive_chunking/splitters.py"),
                "postprocessing.py_sha256": sha256(source / "src/adaptive_chunking/postprocessing.py"),
            },
            "smoke_scope": "official our_recurs_600 and our_recurs_1100 candidate representation generation only; not the final adaptive selector",
        },
        "selection": {
            "rule": "first 10 lexicographically sorted paper_ids in frozen retrieval-eligible QASPER v0.3 test cohort",
            "outcome_independent": True,
        },
        "tested_papers": paper_records,
        "chunk_statistics": aggregate,
        "evidence_mapping_statistics": mapping_summary,
        "generation_failures": generation_failures,
        "feasibility_judgment": {
            "judgment": "CONDITIONAL" if all_candidates_pass else "FAIL",
            "reasons": [
                "The two official deterministic recursive candidates generated gap-free representations for all 10 papers." if all_candidates_pass else "At least one official deterministic candidate failed representation generation.",
                "A final Adaptive representation cannot be selected without adding method-specific candidates, models, metrics, and parser metadata absent from frozen RQ1.",
                "Official 600/1100-token candidates can exceed the unchanged BGE 512-token embedding limit, creating unencoded chunk tails unless a formal compatibility rule is frozen.",
                "Therefore this smoke supports engineering feasibility only, not a fair ready-to-run RQ1 baseline.",
            ],
        },
        "frozen_inputs": {
            "qasper_archive_sha256": sha256(QASPER_ARCHIVE),
            "qasper_gold_sha256": sha256(QASPER_GOLD),
            "split": "QASPER v0.3 test",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"judgment": payload["feasibility_judgment"]["judgment"], "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
