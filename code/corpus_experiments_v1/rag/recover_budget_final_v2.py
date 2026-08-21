#!/usr/bin/env python3
"""Recover frozen RAG embeddings/rankings and finalize budget evaluation v2.

This is an asset-recovery run authorized after the original formal RAG result
failed to persist embeddings or rankings beyond top-10. It keeps the prior
dataset, questions, frozen gold sets, unit-construction code, tokenizer,
embedding model, FP16 encoding, pooling, normalization, and paper-scoped dot
product ranking unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


SEED = 20260811
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 10_000
MODEL = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
TOP_K = 5
MAX_BUDGET = 2048
BUDGETS = (512, 1024, 2048)
VIEWS = ("B1", "B2", "B3")
REGRESSION_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )


def load_original_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_rag_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen RAG runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_queries(retrieval_path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(
        retrieval_path,
        columns=["view", "question_id", "paper_id", "question", "gold_sets_json"],
    ).to_pylist()
    b1 = [row for row in rows if row["view"] == "B1"]
    if len(b1) != 1292 or len({row["question_id"] for row in b1}) != 1292:
        raise ValueError("Frozen B1 query ordering is not the expected 1,292 unique questions")
    return [
        {
            "question_id": row["question_id"],
            "paper_id": row["paper_id"],
            "question": row["question"],
            "complete_gold_sets": json.loads(row["gold_sets_json"]),
        }
        for row in b1
    ]


def persist_unit_metadata(output_dir: Path, view: str, units: list[dict[str, Any]]) -> None:
    rows = []
    for index, unit in enumerate(units):
        rows.append(
            {
                "embedding_index": index,
                "unit_id": unit["unit_id"],
                "paper_id": unit["paper_id"],
                "unit_tokens": int(unit["token_count"]),
                "atomic_ids_json": json_compact(unit["atomic_ids"]),
                "segments_json": json_compact(unit.get("segments") or {}),
                "text_sha256": hashlib.sha256(unit["text"].encode("utf-8")).hexdigest(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{view}_unit_ids.parquet", compression="zstd")


def persist_query_metadata(output_dir: Path, queries: list[dict[str, Any]]) -> None:
    rows = [
        {
            "embedding_index": index,
            "question_id": row["question_id"],
            "paper_id": row["paper_id"],
            "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            "gold_sets_json": json_compact(row["complete_gold_sets"]),
        }
        for index, row in enumerate(queries)
    ]
    pq.write_table(pa.Table.from_pylist(rows), output_dir / "query_ids.parquet", compression="zstd")


def budget_scores(runner: Any, row: dict[str, Any], ranked: list[dict[str, Any]], view: str, budget: int) -> tuple[float, float, int, int]:
    selected, used = [], 0
    for unit in ranked:
        token_count = int(unit["token_count"])
        if used + token_count > budget:
            break
        selected.append(unit)
        used += token_count
    recovered = runner.recovered_ids(selected, view)
    recall, ces, _ = runner.gold_scores(row["complete_gold_sets"], recovered)
    return recall, ces, used, len(selected)


def rank_depth_for_budget(ranked: list[dict[str, Any]], budget: int) -> tuple[int, bool]:
    cumulative = 0
    for rank, unit in enumerate(ranked, start=1):
        cumulative += int(unit["token_count"])
        if cumulative >= budget:
            return rank, False
    return len(ranked), True


def aggregate(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def bootstrap_budget(per_question: dict[str, dict[str, list[float]]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[float, float]]]:
    n = len(per_question["B1"]["budget_ces@512"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n), dtype=np.int32)
    metrics = [
        *[f"budget_ces@{budget}" for budget in BUDGETS],
        *[f"budget_recall@{budget}" for budget in BUDGETS],
    ]
    means: dict[tuple[str, str], np.ndarray] = {}
    view_intervals: dict[tuple[str, str], tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for view in VIEWS:
            values = np.asarray(per_question[view][metric], dtype=np.float64)
            sampled = values[indices].mean(axis=1)
            means[(view, metric)] = sampled
            ci = (float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)))
            view_intervals[(view, metric)] = ci
            rows.append(
                {
                    "row_type": "view_estimate",
                    "budget": int(metric.rsplit("@", 1)[1]),
                    "metric": "BudgetCES" if metric.startswith("budget_ces") else "BudgetRecall",
                    "view_a": view,
                    "view_b": "",
                    "estimate": float(values.mean()),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "raw_p": "",
                    "holm_p": "",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "statistical_role": "primary" if metric.startswith("budget_ces") else "auxiliary",
                }
            )
    primary = []
    for budget in BUDGETS:
        metric = f"budget_ces@{budget}"
        for left, right in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
            left_values = np.asarray(per_question[left][metric], dtype=np.float64)
            right_values = np.asarray(per_question[right][metric], dtype=np.float64)
            differences = left_values - right_values
            sampled = means[(left, metric)] - means[(right, metric)]
            negative = (np.count_nonzero(sampled <= 0.0) + 1) / (BOOTSTRAP_REPLICATES + 1)
            positive = (np.count_nonzero(sampled >= 0.0) + 1) / (BOOTSTRAP_REPLICATES + 1)
            primary.append(
                {
                    "row_type": "pairwise_difference",
                    "budget": budget,
                    "metric": "BudgetCES",
                    "view_a": left,
                    "view_b": right,
                    "estimate": float(differences.mean()),
                    "ci_low": float(np.quantile(sampled, 0.025)),
                    "ci_high": float(np.quantile(sampled, 0.975)),
                    "raw_p": min(1.0, 2.0 * min(negative, positive)),
                    "holm_p": None,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "statistical_role": "primary",
                }
            )
    order = sorted(range(len(primary)), key=lambda index: primary[index]["raw_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, primary[index]["raw_p"] * (len(primary) - rank))
        running = max(running, adjusted)
        primary[index]["holm_p"] = running
    rows.extend(primary)
    for budget in BUDGETS:
        metric = f"budget_recall@{budget}"
        for left, right in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
            differences = np.asarray(per_question[left][metric]) - np.asarray(per_question[right][metric])
            sampled = means[(left, metric)] - means[(right, metric)]
            rows.append(
                {
                    "row_type": "pairwise_difference",
                    "budget": budget,
                    "metric": "BudgetRecall",
                    "view_a": left,
                    "view_b": right,
                    "estimate": float(differences.mean()),
                    "ci_low": float(np.quantile(sampled, 0.025)),
                    "ci_high": float(np.quantile(sampled, 0.975)),
                    "raw_p": "",
                    "holm_p": "",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "statistical_role": "auxiliary_ci_only",
                }
            )
    return rows, view_intervals


def main() -> None:
    root = Path("aicorpus-derived/experiments/corpus_v1/rag")
    output_dir = root / "budget_final_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_path = Path("code/corpus_experiments_v1/rag/run_rag_experiment.py")
    retrieval_path = root / "retrieval_results.parquet"
    config_path = root / "config.json"
    archive_path = Path("aicorpus-data/public_benchmarks/qasper-test-and-evaluator-v0.3.tgz")
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 recovery embedding requires CUDA")
    runner = load_original_runner(runner_path)
    if runner.MODEL != MODEL or runner.MODEL_REVISION != MODEL_REVISION or runner.SEED != SEED:
        raise ValueError("Frozen runner model/revision/seed mismatch")
    queries = load_frozen_queries(retrieval_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    papers = runner.load_qasper_archive(archive_path)
    needed = {row["paper_id"] for row in queries}
    views, _, _ = runner.build_public_views(papers, tokenizer, needed)
    for view in VIEWS:
        if any(not views[view][row["paper_id"]] for row in queries):
            raise ValueError(f"Empty candidate set in {view}")
    persist_query_metadata(output_dir, queries)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True).to(device=device, dtype=torch.float16).eval()
    query_vectors = runner.encode_texts([row["question"] for row in queries], tokenizer, model, device, 128, query=True)
    np.save(output_dir / "query_embeddings.npy", query_vectors)
    ranking_rows: list[dict[str, Any]] = []
    per_question: dict[str, dict[str, list[float]]] = {view: defaultdict(list) for view in VIEWS}
    topk_rows: list[dict[str, Any]] = []
    depth_accumulator: dict[str, dict[str, list[int] | int]] = {
        view: {"needed_depths": [], "top256_needs_expansion": 0, "exhausted_before_2048": 0}
        for view in VIEWS
    }
    original_metrics = {row["view"]: row for row in csv.DictReader((root / "metrics.csv").open(encoding="utf-8"))}
    for view in VIEWS:
        flat_units = [unit for paper_id in sorted(views[view]) for unit in views[view][paper_id]]
        persist_unit_metadata(output_dir, view, flat_units)
        vectors = runner.encode_texts([unit["text"] for unit in flat_units], tokenizer, model, device, 128)
        np.save(output_dir / f"{view}_corpus_embeddings.npy", vectors)
        by_paper_vectors: dict[str, np.ndarray] = {}
        cursor = 0
        for paper_id in sorted(views[view]):
            count = len(views[view][paper_id])
            by_paper_vectors[paper_id] = vectors[cursor:cursor + count]
            cursor += count
        for query_index, row in enumerate(queries):
            units = views[view][row["paper_id"]]
            scores = by_paper_vectors[row["paper_id"]] @ query_vectors[query_index]
            order = np.argsort(-scores, kind="mergesort")
            ranked = [units[int(index)] for index in order]
            if len(ranked) != len(units):
                raise ValueError("Ranking did not retain every candidate unit")
            for rank, index in enumerate(order, start=1):
                unit = units[int(index)]
                ranking_rows.append({"question_id": row["question_id"], "paper_id": row["paper_id"], "view": view, "rank": rank, "unit_id": unit["unit_id"], "score": float(scores[int(index)]), "unit_tokens": int(unit["token_count"])})
            evaluated = runner.evaluate_ranking(row, ranked, view)
            for name in ("ces@5", "hit@5", "recall@5", "mrr"):
                per_question[view][name].append(float(evaluated[name]))
            for budget in BUDGETS:
                recall, ces, _, _ = budget_scores(runner, row, ranked, view, budget)
                per_question[view][f"budget_ces@{budget}"].append(ces)
                per_question[view][f"budget_recall@{budget}"].append(recall)
                needed_depth, exhausted = rank_depth_for_budget(ranked, budget)
                if budget == MAX_BUDGET:
                    depth_accumulator[view]["needed_depths"].append(needed_depth)
                    depth_accumulator[view]["exhausted_before_2048"] += int(exhausted)
            top256_tokens = sum(int(unit["token_count"]) for unit in ranked[:256])
            if len(ranked) > 256 and top256_tokens < MAX_BUDGET:
                depth_accumulator[view]["top256_needs_expansion"] += 1
        aggregate_new = {name: aggregate(per_question[view][name]) for name in ("ces@5", "hit@5", "recall@5", "mrr")}
        for metric, value in aggregate_new.items():
            old = float(original_metrics[view][metric])
            topk_rows.append({"view": view, "metric": metric, "old_value": old, "new_value": value, "absolute_difference": abs(value - old), "status": "PASS" if abs(value - old) <= REGRESSION_TOLERANCE else "BLOCKER"})
        del vectors, by_paper_vectors
        torch.cuda.empty_cache()
    pq.write_table(pa.Table.from_pylist(ranking_rows), output_dir / "retrieval_rankings.parquet", compression="zstd")
    write_csv(output_dir / "topk_regression.csv", topk_rows, ["view", "metric", "old_value", "new_value", "absolute_difference", "status"])
    if any(row["status"] != "PASS" for row in topk_rows):
        (output_dir / "summary.json").write_text(json.dumps({"status": "BLOCKER", "reason": "top_k_or_mrr_regression_mismatch", "topk_regression": topk_rows, "budget_metrics_computed": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_checksums(output_dir)
        raise RuntimeError("BLOCKER: recovered embeddings changed a frozen top-k or MRR regression metric")
    depth_rows = []
    for view in VIEWS:
        needed_depths = list(depth_accumulator[view]["needed_depths"])
        if len(needed_depths) != 1292:
            raise ValueError("Missing max-budget depth records")
        depth_rows.append({"view": view, "formal_question_count": 1292, "initial_top256_needs_expansion": int(depth_accumulator[view]["top256_needs_expansion"]), "top512_additional_needed": sum(256 < depth <= 512 for depth in needed_depths), "beyond_top512_needed": sum(depth > 512 for depth in needed_depths), "max_actual_required_rank_depth": max(needed_depths), "p95_required_rank_depth": float(np.percentile(needed_depths, 95)), "candidate_exhausted_before_2048_tokens": int(depth_accumulator[view]["exhausted_before_2048"]), "budget_prefix_incomplete@512": 0, "budget_prefix_incomplete@1024": 0, "budget_prefix_incomplete@2048": 0})
    write_csv(output_dir / "ranking_depth_summary.csv", depth_rows, ["view", "formal_question_count", "initial_top256_needs_expansion", "top512_additional_needed", "beyond_top512_needed", "max_actual_required_rank_depth", "p95_required_rank_depth", "candidate_exhausted_before_2048_tokens", "budget_prefix_incomplete@512", "budget_prefix_incomplete@1024", "budget_prefix_incomplete@2048"])
    bootstrap_rows, intervals = bootstrap_budget(per_question)
    write_csv(output_dir / "budget_bootstrap.csv", bootstrap_rows, ["row_type", "budget", "metric", "view_a", "view_b", "estimate", "ci_low", "ci_high", "raw_p", "holm_p", "bootstrap_replicates", "statistical_role"])
    metric_rows = []
    for view in VIEWS:
        row: dict[str, Any] = {"view": view, "formal_question_count": 1292}
        for budget in BUDGETS:
            for kind, label in (("budget_ces", "BudgetCES"), ("budget_recall", "BudgetRecall")):
                metric = f"{kind}@{budget}"
                row[f"{label}@{budget}"] = aggregate(per_question[view][metric])
                row[f"{label}@{budget}_ci_low"] = intervals[(view, metric)][0]
                row[f"{label}@{budget}_ci_high"] = intervals[(view, metric)][1]
        metric_rows.append(row)
    metric_fields = ["view", "formal_question_count"]
    for budget in BUDGETS:
        for label in ("BudgetCES", "BudgetRecall"):
            metric_fields.extend([f"{label}@{budget}", f"{label}@{budget}_ci_low", f"{label}@{budget}_ci_high"])
    write_csv(output_dir / "budget_metrics.csv", metric_rows, metric_fields)
    assets: dict[str, Any] = {}
    for filename in ["query_embeddings.npy", *[f"{view}_corpus_embeddings.npy" for view in VIEWS]]:
        array = np.load(output_dir / filename, mmap_mode="r")
        assets[filename] = {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": sha256_file(output_dir / filename)}
    for filename in ["query_ids.parquet", *[f"{view}_unit_ids.parquet" for view in VIEWS], "retrieval_rankings.parquet"]:
        assets[filename] = {"sha256": sha256_file(output_dir / filename)}
    assets["retrieval_rankings.parquet"]["row_count"] = len(ranking_rows)
    config = {
        "experiment_id": "corpus_v1_rag_budget_final_v2",
        "purpose": "authorized recovery of missing embeddings and rankings for frozen budget evaluation; no experimental-condition change",
        "supersedes": {"original_budget_at_1024": "superseded_due_to_top10_truncation", "pre_authorization_v2_blocker": True},
        "frozen_inputs": {"formal_question_count": 1292, "qasper_version": "v0.3 test", "retrieval_results_sha256": sha256_file(retrieval_path), "original_config_sha256": sha256_file(config_path), "qasper_archive_sha256": sha256_file(archive_path), "gold_mapping": {"total": 5648, "mapped": 5420, "mapping_rate": 0.9596317280453258, "excluded": 228}},
        "model": {"name": MODEL, "revision": MODEL_REVISION, "dtype": "float16", "pooling": "CLS", "normalization": "L2", "similarity": "normalized_dot_product", "query_prefix": runner.QUERY_PREFIX, "max_tokens": 512, "batch_size": 128},
        "views": {"B1": "unchanged frozen Fixed 512/64", "B2": "unchanged frozen Atomic", "B3": "unchanged frozen one-hop SEP"},
        "embedding_storage": {"format": "npy", "dtype": "float32", "index": "deterministic per-paper contiguous row ranges plus persisted unit ordering"},
        "persisted_assets": assets,
        "ranking": {"scope": "paper_scoped", "ranking_storage": "all candidates per query in Parquet", "max_budget": MAX_BUDGET, "prefix_rule": "stop before next whole unit would exceed budget"},
        "statistics": {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "holm_family": "9 primary BudgetCES pairwise tests across three budgets"},
        "runner_sha256": sha256_file(runner_path),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "RAG_FINAL_READY", "top_k_regression_passed": True, "top_k_regression": topk_rows, "ranking_depth": depth_rows, "budget_metrics": metric_rows, "mrr_definition": "first rank in the persisted complete per-paper ranking that recovers any gold evidence; RR=0 only after the candidate set is exhausted without a hit", "mrr_ranking_depth": "full per-paper candidate ranking, persisted in retrieval_rankings.parquet", "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "device": str(device), "device_name": torch.cuda.get_device_name(0), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}, "runtime_seconds": time.time() - started, "interpretation_boundary": "This recovers missing ranking depth under the frozen RAG definition; it does not introduce a new corpus construction condition or establish a universally optimal view."}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_checksums(output_dir)
    print(json.dumps({"status": summary["status"], "budget_metrics": metric_rows, "ranking_depth": depth_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
