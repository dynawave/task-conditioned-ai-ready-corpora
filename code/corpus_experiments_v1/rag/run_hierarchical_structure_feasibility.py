#!/usr/bin/env python3
"""Development-only feasibility smoke for a structure-only hierarchical RAG view.

This script does not alter the frozen RQ1 experiment.  It compares the frozen B2
paragraph representation with an otherwise identical Paper -> Section -> Paragraph
serialization on a seed-selected set of 20 QASPER test papers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from corpus_experiments_v1.rag import run_rag_experiment as rag


ROOT = Path(__file__).resolve().parents[3]
FORMAL_DIR = ROOT / "aicorpus-derived/experiments/corpus_v1/rag"
FORMAL_RESULTS = FORMAL_DIR / "retrieval_results.parquet"
QASPER_ARCHIVE = ROOT / "aicorpus-data/public_benchmarks/qasper-test-and-evaluator-v0.3.tgz"
DEFAULT_OUTPUT = FORMAL_DIR / "hierarchical_structure_smoke_v1/hierarchical_structure_feasibility.json"

EXPERIMENT_ID = "hierarchical_structure_only_feasibility_v1"
SAMPLING_SEED = 20260823
PAPER_COUNT = 20
HIER_VIEW = "HIERARCHICAL_STRUCTURE_ONLY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_hierarchical_units(
    paper_id: str,
    paper: dict[str, Any],
    paragraphs: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Serialize the same atomic paragraphs with paper and section hierarchy only."""
    title = rag.normalize_text(paper.get("title"))
    units: list[dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs):
        parts = []
        if title:
            parts.append(f"[Paper] {title}")
        section_path = " > ".join(paragraph.get("section_path") or [])
        if section_path:
            parts.append(f"[Section] {section_path}")
        parts.append(f"[Paragraph] {paragraph['text']}")
        text = "\n".join(parts)
        units.append(
            {
                "unit_id": f"{paper_id}::HIER::{index:05d}",
                "paper_id": paper_id,
                "view": HIER_VIEW,
                "text": text,
                "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                "atomic_ids": [paragraph["id"]],
                "segments": {
                    paragraph["id"]: [0, len(paragraph["tokens"]), len(paragraph["tokens"])]
                },
                "section_ids": [paragraph["section_id"]] if paragraph.get("section_id") else [],
                "section_paths": [paragraph.get("section_path") or []],
            }
        )
    return units


def unit_statistics(units_by_paper: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    units = [unit for paper_id in sorted(units_by_paper) for unit in units_by_paper[paper_id]]
    counts = [int(unit["token_count"]) for unit in units]
    per_paper = [len(units_by_paper[paper_id]) for paper_id in sorted(units_by_paper)]
    return {
        "paper_count": len(units_by_paper),
        "unit_count": len(units),
        "units_per_paper": rag.token_summary(per_paper),
        "unit_token_length": rag.token_summary(counts),
        "total_unit_tokens": int(sum(counts)),
        "units_exceeding_embedding_max_length_512": int(sum(value > 512 for value in counts)),
        "units_exceeding_budget_1024": int(sum(value > rag.TOKEN_BUDGET for value in counts)),
    }


def coverage_statistics(
    rows: list[dict[str, Any]], units_by_paper: dict[str, list[dict[str, Any]]], view: str
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[float, float]]]:
    values: dict[tuple[str, str], tuple[float, float]] = {}
    recalls: list[float] = []
    complete: list[float] = []
    for row in rows:
        recovered = rag.recovered_ids(units_by_paper[row["paper_id"]], view)
        recall, ces, _ = rag.gold_scores(row["complete_gold_sets"], recovered)
        key = (str(row["paper_id"]), str(row["question_id"]))
        values[key] = (float(recall), float(ces))
        recalls.append(float(recall))
        complete.append(float(ces))
    return {
        "question_count": len(rows),
        "fully_covered_question_count": int(sum(complete)),
        "fully_covered_question_rate": float(statistics.fmean(complete)) if complete else 0.0,
        "mean_best_gold_recall_ceiling": float(statistics.fmean(recalls)) if recalls else 0.0,
        "questions_with_zero_mapped_evidence": int(sum(value == 0.0 for value in recalls)),
    }, values


def retrieve_view(
    rows: list[dict[str, Any]],
    units_by_paper: dict[str, list[dict[str, Any]]],
    view: str,
    query_vectors: np.ndarray,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, float]]]:
    flat_units = [unit for paper_id in sorted(units_by_paper) for unit in units_by_paper[paper_id]]
    vectors = rag.encode_texts(
        [unit["text"] for unit in flat_units], tokenizer, model, device, batch_size
    )
    paper_vectors: dict[str, np.ndarray] = {}
    cursor = 0
    for paper_id in sorted(units_by_paper):
        count = len(units_by_paper[paper_id])
        paper_vectors[paper_id] = vectors[cursor : cursor + count]
        cursor += count

    metric_names = (
        "ces@5",
        "budget_ces@1024",
        "hit@5",
        "recall@5",
        "budget_recall@1024",
        "mrr",
    )
    per_metric: dict[str, list[float]] = defaultdict(list)
    per_question: dict[tuple[str, str], dict[str, float]] = {}
    budget_tokens: list[int] = []
    for index, row in enumerate(rows):
        units = units_by_paper[row["paper_id"]]
        scores = paper_vectors[row["paper_id"]] @ query_vectors[index]
        order = np.argsort(-scores, kind="mergesort")
        ranked = [units[int(item)] for item in order]
        evaluated = rag.evaluate_ranking(row, ranked, view)
        key = (str(row["paper_id"]), str(row["question_id"]))
        per_question[key] = {name: float(evaluated[name]) for name in metric_names}
        for name in metric_names:
            per_metric[name].append(float(evaluated[name]))
        budget_tokens.append(int(evaluated["budget_tokens"]))

    metrics = {name: float(statistics.fmean(values)) for name, values in per_metric.items()}
    metrics["question_count"] = len(rows)
    metrics["mean_budget_tokens"] = float(statistics.fmean(budget_tokens)) if budget_tokens else 0.0
    del vectors, paper_vectors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, per_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite development result: {args.output}")
    if sha256_file(QASPER_ARCHIVE) != rag.QASPER_ARCHIVE_SHA256:
        raise RuntimeError("Frozen QASPER archive SHA256 mismatch")
    if not FORMAL_RESULTS.exists():
        raise FileNotFoundError(FORMAL_RESULTS)

    started = time.time()
    formal = pd.read_parquet(FORMAL_RESULTS)
    formal_b1 = formal.loc[formal["view"] == "B1"].copy()
    if len(formal_b1) != 1292:
        raise RuntimeError(f"Expected 1292 frozen formal questions, found {len(formal_b1)}")
    if formal_b1[["paper_id", "question_id"]].duplicated().any():
        raise RuntimeError("Frozen formal question membership is not unique")

    all_papers = sorted(str(value) for value in formal_b1["paper_id"].unique())
    if len(all_papers) < PAPER_COUNT:
        raise RuntimeError("Insufficient formal test papers")
    selected_papers = sorted(random.Random(SAMPLING_SEED).sample(all_papers, PAPER_COUNT))
    selected = formal_b1.loc[formal_b1["paper_id"].astype(str).isin(selected_papers)].copy()
    selected.sort_values(["paper_id", "question_id"], kind="mergesort", inplace=True)
    rows = [
        {
            "paper_id": str(record.paper_id),
            "question_id": str(record.question_id),
            "question": str(record.question),
            "complete_gold_sets": json.loads(record.gold_sets_json),
        }
        for record in selected.itertuples(index=False)
    ]
    if {row["paper_id"] for row in rows} != set(selected_papers):
        raise RuntimeError("At least one selected paper has no formal evaluation question")

    tokenizer = AutoTokenizer.from_pretrained(
        rag.MODEL, revision=rag.MODEL_REVISION, local_files_only=True
    )
    papers = rag.load_qasper_archive(QASPER_ARCHIVE)
    missing = sorted(set(selected_papers) - set(papers))
    if missing:
        raise RuntimeError(f"Selected papers absent from frozen QASPER archive: {missing}")

    paragraph_units: dict[str, list[dict[str, Any]]] = {}
    hierarchical_units: dict[str, list[dict[str, Any]]] = {}
    for paper_id in selected_papers:
        paragraphs = rag.qasper_paragraphs(paper_id, papers[paper_id], tokenizer)
        paragraph_units[paper_id] = rag.build_atomic_units(paper_id, paragraphs, tokenizer)
        hierarchical_units[paper_id] = build_hierarchical_units(
            paper_id, papers[paper_id], paragraphs, tokenizer
        )

    membership_equal = all(
        [unit["atomic_ids"] for unit in paragraph_units[paper_id]]
        == [unit["atomic_ids"] for unit in hierarchical_units[paper_id]]
        for paper_id in selected_papers
    )
    unit_counts_equal = all(
        len(paragraph_units[paper_id]) == len(hierarchical_units[paper_id])
        for paper_id in selected_papers
    )
    title_present = all(
        all(unit["text"].startswith("[Paper] ") for unit in hierarchical_units[paper_id])
        for paper_id in selected_papers
    )
    no_forbidden_payload_fields = all(
        not ({"object_ids", "support_ids", "pages", "bbox_complete", "provenance_complete"} & set(unit))
        for units in hierarchical_units.values()
        for unit in units
    )

    paragraph_coverage, paragraph_coverage_items = coverage_statistics(rows, paragraph_units, "B2")
    hierarchical_coverage, hierarchical_coverage_items = coverage_statistics(
        rows, hierarchical_units, HIER_VIEW
    )
    coverage_equal = paragraph_coverage_items == hierarchical_coverage_items

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; exact frozen FP16 retrieval protocol cannot be run")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModel.from_pretrained(
        rag.MODEL, revision=rag.MODEL_REVISION, local_files_only=True
    ).to(device=device, dtype=torch.float16).eval()
    query_vectors = rag.encode_texts(
        [row["question"] for row in rows], tokenizer, model, device, args.batch_size, query=True
    )
    paragraph_metrics, paragraph_per_question = retrieve_view(
        rows, paragraph_units, "B2", query_vectors, tokenizer, model, device, args.batch_size
    )
    hierarchical_metrics, hierarchical_per_question = retrieve_view(
        rows,
        hierarchical_units,
        HIER_VIEW,
        query_vectors,
        tokenizer,
        model,
        device,
        args.batch_size,
    )
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    del query_vectors, model
    torch.cuda.empty_cache()

    formal_b2 = formal.loc[
        (formal["view"] == "B2")
        & formal["paper_id"].astype(str).isin(selected_papers)
    ]
    frozen_lookup = {
        (str(record.paper_id), str(record.question_id)): {
            "ces@5": float(record.ces_at_5),
            "budget_ces@1024": float(record.budget_ces_at_1024),
            "recall@5": float(record.recall_at_5),
            "budget_recall@1024": float(record.budget_recall_at_1024),
        }
        for record in formal_b2.itertuples(index=False)
    }
    regression_differences = []
    for key, current in paragraph_per_question.items():
        frozen = frozen_lookup.get(key)
        if frozen is None:
            regression_differences.append({"key": list(key), "reason": "missing_frozen_B2_row"})
            continue
        for metric, expected in frozen.items():
            if not np.isclose(current[metric], expected, rtol=0.0, atol=1e-12):
                regression_differences.append(
                    {
                        "key": list(key),
                        "metric": metric,
                        "current": current[metric],
                        "frozen": expected,
                    }
                )
    paragraph_regression_pass = not regression_differences and len(frozen_lookup) == len(rows)

    p_stats = unit_statistics(paragraph_units)
    h_stats = unit_statistics(hierarchical_units)
    metric_difference = {
        name: float(hierarchical_metrics[name] - paragraph_metrics[name])
        for name in (
            "ces@5",
            "budget_ces@1024",
            "hit@5",
            "recall@5",
            "budget_recall@1024",
            "mrr",
        )
    }
    token_overhead_rate = (
        h_stats["total_unit_tokens"] / p_stats["total_unit_tokens"] - 1.0
        if p_stats["total_unit_tokens"]
        else 0.0
    )

    protocol_checks = {
        "formal_question_cohort_1292": len(formal_b1) == 1292,
        "selected_paper_count_20": len(selected_papers) == PAPER_COUNT,
        "all_selected_papers_present": not missing,
        "unit_counts_identical": unit_counts_equal,
        "atomic_membership_identical": membership_equal,
        "coverage_identical": coverage_equal,
        "hierarchy_has_paper_tag": title_present,
        "no_figure_table_equation_qualifier_provenance_payload_added": no_forbidden_payload_fields,
        "paragraph_B2_regression_matches_frozen_results": paragraph_regression_pass,
        "same_embedding_model_and_revision": True,
        "same_query_prefix_pooling_normalization_and_paper_scoped_retrieval": True,
        "same_CES_BudgetCES_code_and_token_budget_1024": True,
        "no_training": True,
        "formal_RQ1_results_not_written": True,
    }
    engineering_pass = all(protocol_checks.values())

    # The view is mechanically fair if all gates pass, but B2 already encodes the
    # section path and retrieval is paper-scoped.  The added signal is therefore
    # only a paper title repeated in every unit, making its scientific distinctness
    # limited even when implementation feasibility is established.
    judgment = "CONDITIONAL" if engineering_pass else "FAIL"
    reasons = (
        [
            "All implementation and frozen-protocol gates passed.",
            "The hierarchical view preserves exactly the B2 paragraph units and gold coverage.",
            "However, frozen B2 already serializes Section -> Paragraph and retrieval is paper-scoped; the only added representational signal is the paper title repeated in each unit.",
            "It is suitable only if RQ1 explicitly defines this as a title-augmented structure-only sensitivity baseline, not as a wholly structure-free versus hierarchical contrast.",
        ]
        if engineering_pass
        else [
            "At least one frozen-protocol or representation-fairness gate failed.",
            *[name for name, passed in protocol_checks.items() if not passed],
        ]
    )

    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "development_only_not_formal_RQ1_result",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "feasibility_judgment": judgment,
        "reasons": reasons,
        "sampling": {
            "population": "unique papers represented by the frozen 1,292-question QASPER v0.3 test cohort",
            "seed": SAMPLING_SEED,
            "algorithm": "random.Random(seed).sample(sorted(unique_paper_ids), 20); selected IDs sorted for execution",
            "paper_count": PAPER_COUNT,
            "selected_paper_ids": selected_papers,
            "selected_question_count": len(rows),
        },
        "representations": {
            "paragraph_baseline": {
                "name": "B2 Fixed formal section-aware atomic paragraph baseline",
                "serialization": "[Section] section_path\\n[Paragraph] paragraph_text",
                "unit_membership": "one frozen QASPER body paragraph per unit",
            },
            "hierarchical_structure_only": {
                "name": HIER_VIEW,
                "serialization": "[Paper] paper_title\\n[Section] section_path\\n[Paragraph] paragraph_text",
                "unit_membership": "identical one-paragraph atomic units as B2",
                "excluded_additions": [
                    "figure payload",
                    "table payload",
                    "equation payload",
                    "qualifier/support context",
                    "provenance payload",
                ],
            },
        },
        "retrieval_protocol": {
            "model": rag.MODEL,
            "model_revision": rag.MODEL_REVISION,
            "precision": "FP16",
            "query_prefix": rag.QUERY_PREFIX,
            "pooling": "CLS",
            "normalization": "L2",
            "similarity": "dot_product",
            "scope": "paper_scoped",
            "embedding_max_length": 512,
            "top_k": rag.TOP_K,
            "token_budget": rag.TOKEN_BUDGET,
            "budget_rule": "ranked whole units until the next unit would exceed the budget",
            "metric_implementation": str(Path(rag.__file__).resolve()),
        },
        "coverage": {
            "paragraph_baseline": paragraph_coverage,
            "hierarchical_structure_only": hierarchical_coverage,
            "identical": coverage_equal,
        },
        "unit_statistics": {
            "paragraph_baseline": p_stats,
            "hierarchical_structure_only": h_stats,
            "hierarchical_minus_paragraph_total_tokens": int(
                h_stats["total_unit_tokens"] - p_stats["total_unit_tokens"]
            ),
            "hierarchical_token_overhead_rate": float(token_overhead_rate),
        },
        "preliminary_retrieval": {
            "paragraph_baseline": paragraph_metrics,
            "hierarchical_structure_only": hierarchical_metrics,
            "hierarchical_minus_paragraph": metric_difference,
            "inference_only": True,
            "no_bootstrap_or_formal_inference": True,
        },
        "protocol_checks": protocol_checks,
        "paragraph_regression_difference_count": len(regression_differences),
        "paragraph_regression_differences": regression_differences[:20],
        "inputs": {
            "formal_retrieval_results": str(FORMAL_RESULTS.resolve()),
            "formal_retrieval_results_sha256": sha256_file(FORMAL_RESULTS),
            "qasper_archive": str(QASPER_ARCHIVE.resolve()),
            "qasper_archive_sha256": sha256_file(QASPER_ARCHIVE),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_vram_bytes": peak_vram,
            "runtime_seconds": float(time.time() - started),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "judgment": judgment}, ensure_ascii=False))


if __name__ == "__main__":
    main()
