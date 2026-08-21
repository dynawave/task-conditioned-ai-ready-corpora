#!/usr/bin/env python3
"""Record the frozen-asset gate for the final RAG budget evaluation.

No model, embedding, index, corpus view, or retrieval operation is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

BUDGETS = (512, 1024, 2048)
VIEWS = ("B1", "B2", "B3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )


def top_k_regression(root: Path, retrieval_path: Path) -> dict[str, object]:
    rows = pq.read_table(
        retrieval_path, columns=["view", "ces_at_5", "hit_at_5", "recall_at_5"]
    ).to_pylist()
    with (root / "metrics.csv").open(encoding="utf-8") as handle:
        original = {row["view"]: row for row in csv.DictReader(handle)}
    metrics: dict[str, dict[str, float]] = {}
    for view in VIEWS:
        selected = [row for row in rows if row["view"] == view]
        current = {
            "CES@5": sum(row["ces_at_5"] for row in selected) / len(selected),
            "Hit@5": sum(row["hit_at_5"] for row in selected) / len(selected),
            "Recall@5": sum(row["recall_at_5"] for row in selected) / len(selected),
        }
        if any(abs(current[key] - float(original[view][key.lower()])) >= 1e-12 for key in current):
            raise ValueError(f"Top-k regression mismatch for {view}")
        metrics[view] = current
    return {
        "source": "frozen retrieval_results.parquet per-question fields, independently re-aggregated and compared with rag/metrics.csv",
        "match_tolerance": 1e-12,
        "passed": True,
        "metrics": metrics,
    }


def main() -> None:
    root = Path("aicorpus-derived/experiments/corpus_v1/rag")
    output_dir = root / "budget_final_v2"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_rows = list(csv.DictReader((root / "budget_sensitivity.csv").open(encoding="utf-8")))
    prior = {(int(row["budget"]), row["view"]): row for row in prior_rows}
    retrieval_path = root / "retrieval_results.parquet"
    regression = top_k_regression(root, retrieval_path)
    config = {
        "experiment_id": "corpus_v1_rag_budget_final_v2",
        "status": "blocked_before_ranking_extension",
        "frozen_inputs": {
            "retrieval_results": retrieval_path.as_posix(),
            "retrieval_results_sha256": sha256_file(retrieval_path),
            "formal_question_count": 1292,
            "views": list(VIEWS),
            "budgets": list(BUDGETS),
        },
        "required_reuse_assets": ["corpus embeddings", "query embeddings", "index or equivalent exact-score asset"],
        "asset_audit": {
            "corpus_embeddings": "not_persisted",
            "query_embeddings": "not_persisted",
            "index": "not_persisted",
            "full_ranking": "not_persisted; only top-10 is available",
        },
        "prohibited_actions_not_performed": ["embedding", "training", "corpus view rebuild", "parameter tuning", "retrieval extension"],
        "budget_rule": "ranked prefix; stop before the first whole unit that would exceed B",
        "bootstrap_protocol": {"replicates": 10000, "holm": True},
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    depth_rows = []
    metrics_rows = []
    for budget in BUDGETS:
        for view in VIEWS:
            audited = prior[(budget, view)]
            incomplete = int(audited["ranking_incomplete_questions"])
            depth_rows.append(
                {
                    "budget": budget,
                    "view": view,
                    "formal_question_count": 1292,
                    "stored_rank_depth_max": 10,
                    "max_actual_required_rank_depth": "",
                    "top50_resolved_queries": "",
                    "top100_resolved_queries": "",
                    "top200_resolved_queries": "",
                    "budget_prefix_incomplete": incomplete,
                    "status": "BLOCKED_MISSING_EMBEDDING_AND_FULL_RANKING_ASSETS" if incomplete else "PREFIX_CLOSED_IN_STORED_TOP10",
                }
            )
            metrics_rows.append(
                {
                    "budget": budget,
                    "view": view,
                    "budget_ces": "",
                    "budget_recall": "",
                    "status": "NOT_COMPUTED_BLOCKED_BY_MISSING_REUSABLE_EMBEDDINGS",
                    "reason": "The required ranking extension cannot be performed without either persisted embeddings/index or prohibited re-embedding.",
                }
            )
    write_csv(
        output_dir / "ranking_depth_summary.csv", depth_rows,
        ["budget", "view", "formal_question_count", "stored_rank_depth_max", "max_actual_required_rank_depth", "top50_resolved_queries", "top100_resolved_queries", "top200_resolved_queries", "budget_prefix_incomplete", "status"],
    )
    write_csv(
        output_dir / "budget_metrics.csv", metrics_rows,
        ["budget", "view", "budget_ces", "budget_recall", "status", "reason"],
    )
    bootstrap_rows = []
    for budget in BUDGETS:
        for metric in ("BudgetCES", "BudgetRecall"):
            for left, right in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
                bootstrap_rows.append(
                    {
                        "budget": budget,
                        "metric": metric,
                        "view_a": left,
                        "view_b": right,
                        "point_difference": "",
                        "ci_low": "",
                        "ci_high": "",
                        "raw_p": "",
                        "holm_p": "",
                        "bootstrap_replicates": 10000,
                        "status": "NOT_RUN_BLOCKED_BY_MISSING_REUSABLE_EMBEDDINGS",
                    }
                )
    write_csv(
        output_dir / "budget_bootstrap.csv", bootstrap_rows,
        ["budget", "metric", "view_a", "view_b", "point_difference", "ci_low", "ci_high", "raw_p", "holm_p", "bootstrap_replicates", "status"],
    )
    summary = {
        "status": "RAG_FINAL_NOT_READY",
        "reason": "The formal run did not persist corpus embeddings, query embeddings, an index, or full rankings. Ranking extension would require re-embedding, which is prohibited by the frozen-task instruction.",
        "original_top_k_results": "remain_valid; no top-k computation was rerun",
        "original_budget_at_1024": "not_final because its ranking may be truncated before the budget prefix closes",
        "ranking_extension_performed": False,
        "budget_metrics_computed": False,
        "bootstrap_computed": False,
        "mrr": {
            "original_definition": "first rank with any recovered gold evidence over the full in-memory ranking during the original run; stored only as an aggregate",
            "final_budget_v2_action": "not recomputed because no full ranking or embeddings were persisted",
        },
        "ranking_completeness": depth_rows,
        "top_k_regression": regression,
        "required_to_unblock": "Authorize a new embedding pass, or provide the original persisted corpus/query embeddings (plus unit ordering) sufficient to extend rankings to a closed 2048-token prefix.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_checksums(output_dir)
    print(json.dumps({"status": summary["status"], "output_dir": output_dir.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
