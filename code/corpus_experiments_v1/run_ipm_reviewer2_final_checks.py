#!/usr/bin/env python3
"""Low-cost Reviewer 2 evidence-chain checks over frozen IPM assets.

No model is trained, no retrieval is rerun, and no frozen artifact is changed.
The script writes one temporary JSON payload plus the requested Markdown files;
the CSV files are materialized separately through the workspace spreadsheet
runtime so that they remain machine-readable spreadsheet artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ipm_reviewer2_final_checks"
RAG = ROOT / "aicorpus-derived/experiments/corpus_v1/rag"
RAG_V2 = RAG / "budget_final_v2"
SOURCE_UTILITY = ROOT / "aicorpus-derived/experiments/corpus_v1/source_suitability_utility_v1"
IPM_ROBUSTNESS = ROOT / "results/ipm_final_robustness"
BOOTSTRAP_REPLICATES = 10_000
RQ1_BOOTSTRAP_SEED = 20260818
RQ2_BOOTSTRAP_SEED = 20260819
VIEWS = ("B1", "B2", "B3")
RQ1_METRICS = ("CES@5", "BudgetCES@512", "BudgetCES@1024", "BudgetCES@2048")
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def rows_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [json_value(row) for row in frame.replace({np.nan: None}).to_dict("records")]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def membership_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def s2k_provenance_and_characteristics() -> tuple[str, pd.DataFrame, dict[str, Any]]:
    gate_path = ROOT / "code/14_source_selection/output/paper_gate0_audit.jsonl"
    feature_path = ROOT / "code/14_source_selection/output/paper_features.jsonl"
    manifest_path = ROOT / "aicorpus-derived/e2_prep_v25/e2_prep_v25.sqlite"
    pilot_script = ROOT / "code/23_evidence_conditioned_sft_pilot/run_text_instruction_pilot.py"
    d2k_script = ROOT / "code/29_schema_aligned_corpus_v2/run_schema_aligned_v2.py"
    nested_script = ROOT / "code/38_nested_pool_e2_prep/prepare_nested_e2.py"
    pilot_summary_path = ROOT / "aicorpus-derived/evidence_conditioned_sft_pilot/summary.json"
    d2k_summary_path = ROOT / "aicorpus-derived/e3_d2k_preparation/summary.json"

    eligible_rows = [json.loads(line) for line in gate_path.open(encoding="utf-8") if line.strip()]
    eligible = [str(row["paper_id"]) for row in eligible_rows if row.get("technical_eligible")]
    require(len(eligible) == 30_883 and len(set(eligible)) == 30_883, "Gate-0 eligible pool is not 30,883 unique papers")

    import random

    expected = list(eligible)
    random.Random(20260727).shuffle(expected)
    expected_s2k = expected[:2000]
    expected_s500 = expected[:500]

    connection = sqlite3.connect(f"file:{manifest_path.resolve().as_posix()}?mode=ro", uri=True)
    manifest_s2k = [
        str(row[0])
        for row in connection.execute(
            "select paper_id from source_pool_members where pool_id='S2K' order by ordinal"
        )
    ]
    manifest_s500 = [
        str(row[0])
        for row in connection.execute(
            "select paper_id from source_pool_members where pool_id='S500' order by ordinal"
        )
    ]
    connection.close()
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    d2k_summary = json.loads(d2k_summary_path.read_text(encoding="utf-8"))
    require(manifest_s2k == expected_s2k, "S2K manifest does not equal shuffled Gate-0 prefix")
    require(manifest_s500 == expected_s500 == manifest_s2k[:500], "S500 nested prefix invariant failed")
    require(membership_sha(manifest_s2k) == "4381084194802c40c2a55fa77a32fb0fcc9e6b44dbd6a9e8fbfd38dc8fd108f4", "S2K SHA mismatch")
    require(membership_sha(manifest_s500) == "1819eb0c5ae5341c38af83c75fa05b87f78898247b38a975ba5b3c70a055c6f4", "S500 SHA mismatch")
    require(pilot_summary["sample"]["paper_ids"] == expected_s500, "Pilot S500 list mismatch")
    require(d2k_summary["scope"]["paper_id_sha256"] == membership_sha(manifest_s2k), "D2K summary SHA mismatch")

    features = pd.read_json(feature_path, lines=True)
    gate = pd.DataFrame(eligible_rows)[["paper_id", "technical_eligible"]]
    features = features.merge(gate, on="paper_id", how="inner", validate="one_to_one")
    features = features[features.technical_eligible].copy()
    require(len(features) == 30_883 and features.paper_id.nunique() == 30_883, "Feature/Gate-0 eligible merge failed")
    features["group"] = np.where(features.paper_id.astype(str).isin(set(manifest_s2k)), "S2K", "eligible_remainder")
    require((features.group == "S2K").sum() == 2000 and (features.group == "eligible_remainder").sum() == 28_883, "S2K/remainder cardinality failed")

    denominator = features.paragraph_count.replace(0, np.nan).astype(float)
    features["figure_density_per_100_paragraphs"] = features.figure_count * 100.0 / denominator
    features["table_density_per_100_paragraphs"] = features.table_count * 100.0 / denominator
    features["formula_density_per_100_paragraphs"] = features.formula_count * 100.0 / denominator
    continuous = [
        ("page_count", "pages"),
        ("paragraph_count", "paragraphs"),
        ("section_count", "sections"),
        ("figure_density_per_100_paragraphs", "objects_per_100_paragraphs"),
        ("table_density_per_100_paragraphs", "objects_per_100_paragraphs"),
        ("formula_density_per_100_paragraphs", "objects_per_100_paragraphs"),
        ("recognized_heading_count", "recognized_headings"),
    ]
    binary = [
        ("has_abstract", "proportion"),
        ("has_introduction_background", "proportion"),
        ("has_methods_materials", "proportion"),
        ("has_results", "proportion"),
        ("has_discussion", "proportion"),
        ("has_conclusion", "proportion"),
    ]
    comparison_rows: list[dict[str, Any]] = []
    for variable, unit in continuous:
        left = pd.to_numeric(features.loc[features.group.eq("S2K"), variable], errors="coerce").dropna()
        right = pd.to_numeric(features.loc[features.group.eq("eligible_remainder"), variable], errors="coerce").dropna()
        pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
        smd = (float(left.mean()) - float(right.mean())) / pooled if pooled > 0 else 0.0
        comparison_rows.append(
            {
                "variable": variable,
                "variable_type": "continuous",
                "unit_or_definition": unit,
                "availability_status": "AVAILABLE_PERSISTED_PRE_TASK",
                "s2k_n_nonmissing": len(left),
                "s2k_missing_n": 2000 - len(left),
                "s2k_mean_or_proportion": float(left.mean()),
                "s2k_median": float(left.median()),
                "remainder_n_nonmissing": len(right),
                "remainder_missing_n": 28_883 - len(right),
                "remainder_mean_or_proportion": float(right.mean()),
                "remainder_median": float(right.median()),
                "smd": smd,
                "absolute_proportion_difference": None,
            }
        )
    for variable, unit in binary:
        left = pd.to_numeric(features.loc[features.group.eq("S2K"), variable], errors="coerce").dropna()
        right = pd.to_numeric(features.loc[features.group.eq("eligible_remainder"), variable], errors="coerce").dropna()
        difference = float(left.mean() - right.mean())
        comparison_rows.append(
            {
                "variable": variable,
                "variable_type": "binary",
                "unit_or_definition": unit,
                "availability_status": "AVAILABLE_PERSISTED_PRE_TASK",
                "s2k_n_nonmissing": len(left),
                "s2k_missing_n": 2000 - len(left),
                "s2k_mean_or_proportion": float(left.mean()),
                "s2k_median": None,
                "remainder_n_nonmissing": len(right),
                "remainder_missing_n": 28_883 - len(right),
                "remainder_mean_or_proportion": float(right.mean()),
                "remainder_median": None,
                "smd": None,
                "absolute_proportion_difference": abs(difference),
            }
        )
    unavailable = [
        ("source_tokens", "frozen tokenizer token count"),
        ("average_paragraph_length", "frozen tokenizer tokens per paragraph"),
        ("reference_like_fraction", "reference-like paragraphs / paragraphs"),
        ("suspected_word_split_indicator", "any suspected word-split paragraph"),
        ("publication_year", "year"),
        ("document_type", "document type"),
        ("language", "language"),
    ]
    for variable, definition in unavailable:
        comparison_rows.append(
            {
                "variable": variable,
                "variable_type": "not_available",
                "unit_or_definition": definition,
                "availability_status": "NOT_PERSISTED_FOR_FULL_30883_POOL_NO_IMPUTATION",
                "s2k_n_nonmissing": 0,
                "s2k_missing_n": 2000,
                "s2k_mean_or_proportion": None,
                "s2k_median": None,
                "remainder_n_nonmissing": 0,
                "remainder_missing_n": 28_883,
                "remainder_mean_or_proportion": None,
                "remainder_median": None,
                "smd": None,
                "absolute_proportion_difference": None,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    available = comparison[comparison.availability_status.eq("AVAILABLE_PERSISTED_PRE_TASK")]
    max_abs_smd = float(available.smd.dropna().abs().max())
    max_abs_prop = float(available.absolute_proportion_difference.dropna().max())
    imbalance = max_abs_smd >= 0.10 or max_abs_prop >= 0.10

    markdown = f"""# S2K sampling provenance

## Recovered selection rule

The authoritative implementation is recoverable. `paper_gate0_audit.jsonl` is read in its persisted order, the 30,883 rows with `technical_eligible=true` are collected, `random.Random(20260727).shuffle(eligible)` is applied, and the first 2,000 paper IDs are retained. The frozen SQLite S2K ordinal list equals that random prefix exactly (SHA-256 `{membership_sha(manifest_s2k)}`). S500 is the first 500 entries of the same shuffled order (SHA-256 `{membership_sha(manifest_s500)}`).

- Random: yes, a seeded shuffle followed by prefix selection.
- Stratified: no for S500/S2K.
- Topic clusters: not used for S2K selection.
- Length or structural variables: not used for S2K selection.
- Round-robin: not used for S2K selection.
- Seed: `20260727`.
- Timing: S2K membership was fixed before the later T1a/T2/T3 yield modelling and source-suitability analyses. The script then generated task candidates from the already selected IDs.
- Nested-family status: S500 is a prefix of S2K. S2K was later preserved as the historical base of the S500/S2K/S5K/S10K/S20K family; the stratified topic/paragraph/structure deficit-fill rule applies only to later S5K/S10K/S20K extensions and must not be attributed to S2K.

## Direct evidence

- `{pilot_script.relative_to(ROOT).as_posix()}`: original S500 shuffle and prefix.
- `{d2k_script.relative_to(ROOT).as_posix()}`: D2K/S2K reproduction with the same shuffle and nested-prefix gate.
- `{nested_script.relative_to(ROOT).as_posix()}`: labels S500/S2K as `historical_frozen_random_prefix` and later extensions as stratified deficit fill.
- `{manifest_path.relative_to(ROOT).as_posix()}::source_pool_members`: frozen ordinal membership.

## Pre-task characteristic comparison

The comparison is descriptive only. Continuous SMD uses the unweighted pooled-SD definition `difference in means / sqrt((SD_S2K^2 + SD_remainder^2)/2)`. Object densities use persisted object counts per 100 persisted paragraphs because frozen-token denominators were not persisted for all 30,883 papers. No post-task yield or eligibility label was used. Variables unavailable for the full pool are explicitly marked and were not imputed.

Observed maximum absolute SMD was {max_abs_smd:.4f}; maximum absolute binary proportion difference was {max_abs_prop:.4f}. Conclusion: {'some observed imbalance exists' if imbalance else 'no major observed imbalance on measured pre-task characteristics'}. This is not a claim that S2K probabilistically represents every unmeasured property of the full eligible pool.

## Manuscript wording gate

The named v86 main manuscript and Supplement were not present in the workspace or attachment store, so their exact wording is `NOT RECOVERABLE` in this run. Any v86 claim that S2K itself used topic/structure stratification or round-robin is contradicted by the recovered code and should not be retained.
"""
    metadata = {
        "eligible_n": len(eligible),
        "s2k_n": len(manifest_s2k),
        "remainder_n": len(eligible) - len(manifest_s2k),
        "seed": 20260727,
        "selection_rule": "random.Random(seed).shuffle(persisted_gate0_eligible_order); prefix_2000",
        "s2k_sha256": membership_sha(manifest_s2k),
        "s500_sha256": membership_sha(manifest_s500),
        "max_abs_smd": max_abs_smd,
        "max_abs_proportion_difference": max_abs_prop,
        "imbalance_conclusion": "some observed imbalance exists" if imbalance else "no major observed imbalance on measured pre-task characteristics",
    }
    return markdown, comparison, metadata


def merge_interval_length(spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    spans = sorted(spans)
    left, right = spans[0]
    total = 0
    for next_left, next_right in spans[1:]:
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
        for atomic_id, triple in unit["segments"].items():
            start, end, total = [int(value) for value in triple]
            intervals[atomic_id].append((start, end))
            totals[atomic_id] = total
    return {
        atomic_id
        for atomic_id, spans in intervals.items()
        if totals[atomic_id] > 0 and merge_interval_length(spans) >= totals[atomic_id]
    }


def gold_scores(gold_sets: list[dict[str, Any]], recovered: set[str]) -> tuple[float, float]:
    best_recall = 0.0
    complete = 0.0
    for annotation in gold_sets:
        elements = annotation["evidence_elements"]
        hits = 0
        for alternatives in elements:
            hit = any(bool(alternative) and set(alternative) <= recovered for alternative in alternatives)
            hits += int(hit)
        recall = hits / len(elements) if elements else 0.0
        best_recall = max(best_recall, recall)
        complete = max(complete, float(bool(elements) and recall == 1.0))
    return best_recall, complete


def load_rq1_question_metrics() -> pd.DataFrame:
    query = pd.read_parquet(RAG_V2 / "query_ids.parquet")
    ranking = pd.read_parquet(RAG_V2 / "retrieval_rankings.parquet")
    require(len(query) == 1292 and query.question_id.nunique() == 1292, "RQ1 query cardinality failed")
    require(ranking.question_id.nunique() == 1292 and set(ranking.view) == set(VIEWS), "RQ1 ranking coverage failed")
    query_by_id = query.set_index("question_id")
    unit_by_view: dict[str, dict[str, dict[str, Any]]] = {}
    for view in VIEWS:
        units = pd.read_parquet(RAG_V2 / f"{view}_unit_ids.parquet")
        unit_by_view[view] = {
            str(row.unit_id): {
                "unit_id": str(row.unit_id),
                "token_count": int(row.unit_tokens),
                "atomic_ids": json.loads(row.atomic_ids_json),
                "segments": json.loads(row.segments_json),
            }
            for row in units.itertuples(index=False)
        }
    rows: list[dict[str, Any]] = []
    for (question_id, view), group in ranking.groupby(["question_id", "view"], sort=False):
        group = group.sort_values("rank", kind="mergesort")
        ranked = [unit_by_view[str(view)][str(unit_id)] for unit_id in group.unit_id]
        gold_sets = json.loads(query_by_id.loc[question_id, "gold_sets_json"])
        _, ces5 = gold_scores(gold_sets, recovered_ids(ranked[:5], str(view)))
        values: dict[str, float] = {"CES@5": ces5}
        for budget in (512, 1024, 2048):
            selected: list[dict[str, Any]] = []
            used = 0
            for unit in ranked:
                token_count = int(unit["token_count"])
                if used + token_count > budget:
                    break
                selected.append(unit)
                used += token_count
            _, ces = gold_scores(gold_sets, recovered_ids(selected, str(view)))
            values[f"BudgetCES@{budget}"] = ces
        rows.append(
            {
                "question_id": str(question_id),
                "paper_id": str(query_by_id.loc[question_id, "paper_id"]),
                "view": str(view),
                **values,
            }
        )
    output = pd.DataFrame(rows)
    require(len(output) == 1292 * 3 and not output.duplicated(["question_id", "view"]).any(), "RQ1 question metric matrix failed")

    expected = {
        "CES@5": {"B1": 0.6888544891640866, "B2": 0.5820433436532507, "B3": 0.6222910216718266},
        "BudgetCES@512": {"B1": 0.21749226006191952, "B2": 0.5023219814241486, "B3": 0.3893188854489164},
        "BudgetCES@1024": {"B1": 0.39473684210526316, "B2": 0.6501547987616099, "B3": 0.5340557275541795},
        "BudgetCES@2048": {"B1": 0.6075851393188855, "B2": 0.8003095975232198, "B3": 0.684984520123839},
    }
    for metric, view_values in expected.items():
        current = output.groupby("view")[metric].mean().to_dict()
        for view, value in view_values.items():
            require(abs(float(current[view]) - value) <= 1e-12, f"RQ1 frozen regression failed: {metric}/{view}")
    return output


def question_level_intervals() -> dict[tuple[str, str, str, str], tuple[float, float]]:
    intervals: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    old = pd.read_csv(RAG / "bootstrap.csv")
    for row in old.itertuples(index=False):
        metric = "CES@5" if row.metric == "ces@5" else "BudgetCES@1024"
        kind = "view_estimate" if row.row_type == "view_estimate" else "pairwise_difference"
        intervals[(kind, metric, str(row.view_a), "" if pd.isna(row.view_b) else str(row.view_b))] = (float(row.ci_low), float(row.ci_high))
    budget = pd.read_csv(RAG_V2 / "budget_bootstrap.csv")
    budget = budget[budget.metric.eq("BudgetCES")]
    for row in budget.itertuples(index=False):
        metric = f"BudgetCES@{int(row.budget)}"
        intervals[(str(row.row_type), metric, str(row.view_a), "" if pd.isna(row.view_b) else str(row.view_b))] = (float(row.ci_low), float(row.ci_high))
    return intervals


def normalized_interval(
    intervals: dict[tuple[str, str, str, str], tuple[float, float]],
    row_type: str,
    metric: str,
    left: str,
    right: str = "",
) -> tuple[float | None, float | None]:
    key = (row_type, metric, left, right)
    if key in intervals:
        return intervals[key]
    reverse = (row_type, metric, right, left)
    if reverse in intervals:
        low, high = intervals[reverse]
        return -high, -low
    return None, None


def cluster_bootstrap(
    metrics: pd.DataFrame,
    subset_label: str,
    include_question_intervals: bool,
    seed: int,
) -> pd.DataFrame:
    papers = sorted(metrics.paper_id.astype(str).unique())
    question_ids = sorted(metrics.question_id.astype(str).unique())
    paper_index = {paper: index for index, paper in enumerate(papers)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(papers), size=(BOOTSTRAP_REPLICATES, len(papers)), dtype=np.int32)
    counts_by_paper = (
        metrics[["question_id", "paper_id"]].drop_duplicates().groupby("paper_id").size().reindex(papers).to_numpy(float)
    )
    denominators = counts_by_paper[draws].sum(axis=1)
    sampled_means: dict[tuple[str, str], np.ndarray] = {}
    values_by_view_metric: dict[tuple[str, str], np.ndarray] = {}
    for view in VIEWS:
        frame = metrics[metrics.view.eq(view)].set_index("question_id").loc[question_ids]
        for metric in RQ1_METRICS:
            values_by_view_metric[(view, metric)] = frame[metric].to_numpy(float)
            sums = frame.assign(_paper=frame.paper_id.astype(str)).groupby("_paper")[metric].sum().reindex(papers).to_numpy(float)
            sampled_means[(view, metric)] = sums[draws].sum(axis=1) / denominators
    old_intervals = question_level_intervals() if include_question_intervals else {}
    rows: list[dict[str, Any]] = []
    for metric in RQ1_METRICS:
        for view in VIEWS:
            sampled = sampled_means[(view, metric)]
            point = float(values_by_view_metric[(view, metric)].mean())
            low, high = [float(value) for value in np.quantile(sampled, [0.025, 0.975])]
            q_low, q_high = normalized_interval(old_intervals, "view_estimate", metric, view) if include_question_intervals else (None, None)
            q_width = q_high - q_low if q_low is not None else None
            c_width = high - low
            rows.append(
                {
                    "subset": subset_label,
                    "row_type": "view_estimate",
                    "metric": metric,
                    "view_a": view,
                    "view_b": None,
                    "n_questions": len(question_ids),
                    "n_papers": len(papers),
                    "estimate": point,
                    "paper_cluster_ci_low": low,
                    "paper_cluster_ci_high": high,
                    "paper_cluster_ci_width": c_width,
                    "question_ci_low": q_low,
                    "question_ci_high": q_high,
                    "question_ci_width": q_width,
                    "ci_width_ratio_cluster_over_question": c_width / q_width if q_width else None,
                    "effect_sign": "positive" if point > 0 else "zero",
                    "conclusion_changed": False if include_question_intervals else None,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": seed,
                }
            )
        for left, right in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
            point_values = values_by_view_metric[(left, metric)] - values_by_view_metric[(right, metric)]
            sampled = sampled_means[(left, metric)] - sampled_means[(right, metric)]
            point = float(point_values.mean())
            low, high = [float(value) for value in np.quantile(sampled, [0.025, 0.975])]
            q_low, q_high = normalized_interval(old_intervals, "pairwise_difference", metric, left, right) if include_question_intervals else (None, None)
            q_width = q_high - q_low if q_low is not None else None
            c_width = high - low
            old_status = (q_low > 0, q_high < 0) if q_low is not None else (None, None)
            new_status = (low > 0, high < 0)
            rows.append(
                {
                    "subset": subset_label,
                    "row_type": "pairwise_difference",
                    "metric": metric,
                    "view_a": left,
                    "view_b": right,
                    "n_questions": len(question_ids),
                    "n_papers": len(papers),
                    "estimate": point,
                    "paper_cluster_ci_low": low,
                    "paper_cluster_ci_high": high,
                    "paper_cluster_ci_width": c_width,
                    "question_ci_low": q_low,
                    "question_ci_high": q_high,
                    "question_ci_width": q_width,
                    "ci_width_ratio_cluster_over_question": c_width / q_width if q_width else None,
                    "effect_sign": "positive" if point > 0 else "negative" if point < 0 else "zero",
                    "conclusion_changed": old_status != new_status if q_low is not None else None,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def mapping_audit(question_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows = pd.read_csv(RAG / "public_gold_mapping.csv")
    unmatched = pd.read_csv(RAG / "unmatched_gold.csv")
    require(
        len(all_rows) == 5648
        and int(all_rows.status.eq("mapped").sum()) == 5420
        and int(all_rows.status.eq("unmatched").sum()) == 228
        and len(unmatched) == 228,
        "Gold mapping row totals failed",
    )
    key_columns = ["question_id", "annotation_index", "evidence_index", "evidence_sha256"]
    embedded_unmatched = all_rows[all_rows.status.eq("unmatched")][key_columns].sort_values(key_columns).reset_index(drop=True)
    separate_unmatched = unmatched[key_columns].sort_values(key_columns).reset_index(drop=True)
    require(embedded_unmatched.equals(separate_unmatched), "unmatched_gold.csv is not the exact unmatched subset")
    require(not all_rows.duplicated(["question_id", "annotation_index", "evidence_index"]).any(), "Duplicate gold evidence mapping rows")
    formal = question_metrics[["question_id", "paper_id"]].drop_duplicates()
    count = all_rows.groupby("question_id").agg(
        original_gold_evidence_count=("status", "size"),
        mapped_gold_evidence_count=("status", lambda values: int((values == "mapped").sum())),
        unmapped_gold_evidence_count=("status", lambda values: int((values != "mapped").sum())),
    ).reset_index()
    audit = formal.merge(count, on="question_id", how="left", validate="one_to_one")
    audit[["original_gold_evidence_count", "mapped_gold_evidence_count", "unmapped_gold_evidence_count"]] = audit[
        ["original_gold_evidence_count", "mapped_gold_evidence_count", "unmapped_gold_evidence_count"]
    ].fillna(0).astype(int)
    audit["complete_mapping_question"] = (
        audit.original_gold_evidence_count.gt(0)
        & audit.mapped_gold_evidence_count.eq(audit.original_gold_evidence_count)
    )
    audit = audit.sort_values(["paper_id", "question_id"], kind="mergesort").reset_index(drop=True)
    require(len(audit) == 1292, "Mapping audit does not cover 1,292 formal questions")
    complete = audit[audit.complete_mapping_question]
    incomplete = audit[~audit.complete_mapping_question]
    metadata = {
        "global_original_gold_evidence_n": len(all_rows),
        "global_mapped_gold_evidence_n": int(all_rows.status.eq("mapped").sum()),
        "global_unmapped_gold_evidence_n": int(all_rows.status.eq("unmatched").sum()),
        "global_mapping_rate": float(all_rows.status.eq("mapped").sum() / len(all_rows)),
        "formal_1292_original_gold_evidence_n": int(audit.original_gold_evidence_count.sum()),
        "formal_1292_mapped_gold_evidence_n": int(audit.mapped_gold_evidence_count.sum()),
        "formal_1292_unmapped_gold_evidence_n": int(audit.unmapped_gold_evidence_count.sum()),
        "formal_1292_mapping_rate": float(audit.mapped_gold_evidence_count.sum() / audit.original_gold_evidence_count.sum()),
        "complete_questions_n": len(complete),
        "complete_questions_fraction": len(complete) / len(audit),
        "incomplete_questions_n": len(incomplete),
        "complete_gold_count_mean": float(complete.original_gold_evidence_count.mean()),
        "complete_gold_count_median": float(complete.original_gold_evidence_count.median()),
        "incomplete_gold_count_mean": float(incomplete.original_gold_evidence_count.mean()),
        "incomplete_gold_count_median": float(incomplete.original_gold_evidence_count.median()),
    }
    return audit, metadata


def mapping_sensitivity(question_metrics: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete_ids = set(audit.loc[audit.complete_mapping_question, "question_id"].astype(str))
    complete_metrics = question_metrics[question_metrics.question_id.astype(str).isin(complete_ids)].copy()
    full = cluster_bootstrap(question_metrics, "FULL_1292", False, RQ1_BOOTSTRAP_SEED + 100)
    complete = cluster_bootstrap(complete_metrics, "COMPLETE_MAPPING_ONLY", False, RQ1_BOOTSTRAP_SEED + 101)
    combined = pd.concat([full, complete], ignore_index=True)
    full_points = full.set_index(["row_type", "metric", "view_a", "view_b"], drop=False).estimate
    deltas = []
    for row in combined.itertuples(index=False):
        key = (row.row_type, row.metric, row.view_a, row.view_b)
        full_point = float(full_points.loc[key])
        deltas.append((full_point, float(row.estimate) - full_point))
    combined["full_1292_estimate"] = [item[0] for item in deltas]
    combined["estimate_change_from_full"] = [item[1] for item in deltas]
    view_rows = complete[complete.row_type.eq("view_estimate")]
    best = view_rows.loc[view_rows.groupby("metric").estimate.idxmax(), ["metric", "view_a"]]
    best_map = dict(zip(best.metric, best.view_a))
    metadata = {
        "best_complete_mapping": best_map,
        "topk_best_preserved": best_map["CES@5"] == "B1",
        "budget_best_preserved": all(best_map[f"BudgetCES@{budget}"] == "B2" for budget in (512, 1024, 2048)),
        "max_absolute_view_estimate_change": float(combined[combined.row_type.eq("view_estimate")].estimate_change_from_full.abs().max()),
    }
    return combined, metadata


def percentile_score(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", ascending=True)
    return (ranks - 1.0) / (len(values) - 1.0)


def stable_order(scores: np.ndarray, paper_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((paper_ids.astype(str), -scores.astype(float)))


def selected_paper_indices(scores: np.ndarray, paper_ids: np.ndarray, fraction: float) -> np.ndarray:
    return stable_order(scores, paper_ids)[: math.ceil(len(scores) * fraction)]


def rq2_length_only() -> tuple[pd.DataFrame, dict[str, Any]]:
    oof = pd.read_parquet(SOURCE_UTILITY / "oof_predictions.parquet")
    require(len(oof) == 1898 * 4 and not oof.duplicated(["task", "paper_id"]).any(), "RQ2 OOF cardinality failed")
    main = oof[oof.task.isin(TASKS)].copy()
    frames = {
        task: main[main.task.eq(task)].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        for task in TASKS
    }
    reference = frames[TASKS[0]]
    for task, frame in frames.items():
        require(len(frame) == 1898, f"RQ2 N failed: {task}")
        require(frame.paper_id.astype(str).equals(reference.paper_id.astype(str)), f"RQ2 membership drift: {task}")
        require(np.array_equal(frame.source_tokens.to_numpy(np.int64), reference.source_tokens.to_numpy(np.int64)), f"RQ2 token drift: {task}")

    count = main.pivot(index="paper_id", columns="task", values="oof_predicted_count").sort_index()
    common = pd.concat([percentile_score(count[task]) for task in TASKS], axis=1).mean(axis=1)
    frozen_curves = pd.read_csv(IPM_ROBUSTNESS / "rq2_budget_curves.csv")
    rows: list[dict[str, Any]] = []
    bootstrap_results: dict[str, dict[str, float]] = {}
    sanity_expected = {"T1a_ACRONYM": 0.27408282276518786, "T2_NUMERIC": 0.33273922782340615, "T3_CLAIM": 0.3469928993070408}
    for task in TASKS:
        frame = frames[task]
        ids = frame.paper_id.astype(str).to_numpy()
        costs = frame.source_tokens.to_numpy(np.int64)
        yields = frame.actual_mv_count.to_numpy(float)
        total = float(yields.sum())
        specific_scores = count.loc[ids, task].to_numpy(float)
        common_scores = common.loc[ids].to_numpy(float)
        length_scores = costs.astype(float)
        for fraction in BUDGETS:
            deterministic = {
                "length_only": (selected_paper_indices(length_scores, ids, fraction), "source_tokens_desc_paper_id_asc"),
                "common": (selected_paper_indices(common_scores, ids, fraction), "mean_average_tie_percentiles_of_three_oof_predicted_count_scores"),
                "task_specific": (selected_paper_indices(specific_scores, ids, fraction), "oof_predicted_count"),
            }
            length_capture = float(yields[deterministic["length_only"][0]].sum() / total)
            for strategy, (selected, score_name) in deterministic.items():
                captured = float(yields[selected].sum())
                capture_fraction = captured / total
                rows.append(
                    {
                        "task": task,
                        "budget_type": "paper_count",
                        "budget_fraction": fraction,
                        "strategy": strategy,
                        "ranking_score": score_name,
                        "selected_papers": len(selected),
                        "captured_instances": captured,
                        "total_instances": total,
                        "capture_fraction": capture_fraction,
                        "lift": capture_fraction / fraction,
                        "specific_minus_length": capture_fraction - length_capture if strategy == "task_specific" else None,
                        "common_minus_length": capture_fraction - length_capture if strategy == "common" else None,
                        "specific_minus_length_ci_low": None,
                        "specific_minus_length_ci_high": None,
                        "common_minus_length_ci_low": None,
                        "common_minus_length_ci_high": None,
                        "bootstrap_replicates": None,
                    }
                )
            random = frozen_curves[
                frozen_curves.task.eq(task)
                & frozen_curves.budget_type.eq("paper_count")
                & np.isclose(frozen_curves.budget_fraction, fraction)
                & frozen_curves.strategy.eq("random")
            ]
            require(len(random) == 1 and int(random.iloc[0].random_repeats) == 10_000, f"Frozen random curve missing: {task}/{fraction}")
            random = random.iloc[0]
            rows.append(
                {
                    "task": task,
                    "budget_type": "paper_count",
                    "budget_fraction": fraction,
                    "strategy": "random",
                    "ranking_score": "uniform_random_paper_order_frozen_10000",
                    "selected_papers": float(random.selected_papers),
                    "captured_instances": float(random.captured_instances),
                    "total_instances": float(random.total_instances),
                    "capture_fraction": float(random.capture_fraction),
                    "lift": float(random.lift),
                    "specific_minus_length": None,
                    "common_minus_length": None,
                    "specific_minus_length_ci_low": None,
                    "specific_minus_length_ci_high": None,
                    "common_minus_length_ci_low": None,
                    "common_minus_length_ci_high": None,
                    "bootstrap_replicates": 10_000,
                }
            )

        specific_20 = selected_paper_indices(specific_scores, ids, 0.20)
        specific_capture = float(yields[specific_20].sum() / total)
        require(abs(specific_capture - sanity_expected[task]) <= 1e-12, f"RQ2 frozen 20% sanity failed: {task}")
        rng = np.random.default_rng(int(sha256_text(f"RQ2_LENGTH_BOOTSTRAP|{RQ2_BOOTSTRAP_SEED}|{task}")[:16], 16))
        spec_diff = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        common_diff = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for index in range(BOOTSTRAP_REPLICATES):
            sample = rng.integers(0, len(frame), size=len(frame), dtype=np.int32)
            sample_ids = np.asarray([f"{position:04d}|{ids[source]}" for position, source in enumerate(sample)], dtype=str)
            y = yields[sample]
            sample_total = float(y.sum())
            selected_length = selected_paper_indices(length_scores[sample], sample_ids, 0.20)
            selected_specific = selected_paper_indices(specific_scores[sample], sample_ids, 0.20)
            selected_common = selected_paper_indices(common_scores[sample], sample_ids, 0.20)
            length_value = float(y[selected_length].sum() / sample_total)
            spec_diff[index] = float(y[selected_specific].sum() / sample_total) - length_value
            common_diff[index] = float(y[selected_common].sum() / sample_total) - length_value
        bootstrap_results[task] = {
            "specific_low": float(np.quantile(spec_diff, 0.025)),
            "specific_high": float(np.quantile(spec_diff, 0.975)),
            "common_low": float(np.quantile(common_diff, 0.025)),
            "common_high": float(np.quantile(common_diff, 0.975)),
        }

    output = pd.DataFrame(rows).sort_values(["task", "budget_fraction", "strategy"], kind="mergesort").reset_index(drop=True)
    for task, result in bootstrap_results.items():
        mask = output.task.eq(task) & np.isclose(output.budget_fraction, 0.20)
        output.loc[mask & output.strategy.eq("task_specific"), ["specific_minus_length_ci_low", "specific_minus_length_ci_high", "bootstrap_replicates"]] = [result["specific_low"], result["specific_high"], BOOTSTRAP_REPLICATES]
        output.loc[mask & output.strategy.eq("common"), ["common_minus_length_ci_low", "common_minus_length_ci_high", "bootstrap_replicates"]] = [result["common_low"], result["common_high"], BOOTSTRAP_REPLICATES]
    at20 = output[np.isclose(output.budget_fraction, 0.20)].pivot(index="task", columns="strategy", values="capture_fraction")
    metadata = {
        "effective_n": 1898,
        "length_capture_20": {task: float(at20.loc[task, "length_only"]) for task in TASKS},
        "specific_capture_20": {task: float(at20.loc[task, "task_specific"]) for task in TASKS},
        "common_capture_20": {task: float(at20.loc[task, "common"]) for task in TASKS},
        "bootstrap": bootstrap_results,
        "specific_ci_excludes_zero": {task: bool(result["specific_low"] > 0 or result["specific_high"] < 0) for task, result in bootstrap_results.items()},
    }
    return output, metadata


def t2_numeric_audit() -> tuple[pd.DataFrame, str, dict[str, Any]]:
    validation_dir = ROOT / "code/corpus_experiments_v1/sft/validation"
    sys.path.insert(0, str(validation_dir))
    from common import exact_or_normalized_in, is_strict_numeric_answer, normalize_text, numeric_signature  # type: ignore

    qasper_path = ROOT / "aicorpus-data/public_benchmarks/qasper-test-and-evaluator-v0.3.tgz"
    scitat_path = ROOT / "aicorpus-data/external_gold/SciTaT-main/dataset/scitat_test.json"
    funnel: list[dict[str, Any]] = []
    with tarfile.open(qasper_path, "r:*") as archive:
        member = next(name for name in archive.getnames() if name.endswith("qasper-test-v0.3.json"))
        qasper = json.load(archive.extractfile(member))
    papers_n = len(qasper)
    qas_n = sum(len(paper["qas"]) for paper in qasper.values())
    annotations_n = sum(len(qa["answers"]) for paper in qasper.values() for qa in paper["qas"])
    spans: list[dict[str, Any]] = []
    for paper_id, paper in qasper.items():
        for qa in paper["qas"]:
            for annotation in qa["answers"]:
                answer = annotation["answer"]
                evidence = [normalize_text(value) for value in answer.get("evidence") or []]
                for span in answer.get("extractive_spans") or []:
                    spans.append({"span": normalize_text(span), "evidence": evidence})
    numeric = [row for row in spans if is_strict_numeric_answer(row["span"])]
    with_evidence = [row for row in numeric if bool(row["evidence"])]
    matched = [row for row in with_evidence if any(exact_or_normalized_in(row["span"], item)[1] for item in row["evidence"])]
    qasper_steps = [
        ("official_test_papers", papers_n, papers_n, "upstream_context_not_candidate_unit"),
        ("official_test_questions", qas_n, qas_n, "upstream_context_not_candidate_unit"),
        ("human_answer_annotations", annotations_n, annotations_n, "upstream_context_not_candidate_unit"),
        ("human_extractive_spans", len(spans), len(spans), "candidate_unit_start"),
        ("strict_numeric_range_percent_number_unit_or_statistical_schema", len(spans), len(numeric), "eligibility_filter"),
        ("nonempty_human_annotated_evidence", len(numeric), len(with_evidence), "eligibility_filter"),
        ("normalized_answer_present_in_human_gold_evidence", len(with_evidence), len(matched), "eligibility_filter"),
    ]
    for order, (rule, input_n, remaining_n, role) in enumerate(qasper_steps, start=1):
        funnel.append({"dataset": "QASPER", "stage_order": order, "rule": rule, "input_n": input_n, "excluded_n": input_n - remaining_n, "remaining_n": remaining_n, "stage_role": role})
    require(len(matched) == 126, "QASPER numeric N=126 reproduction failed")

    scitat = json.loads(scitat_path.read_text(encoding="utf-8"))
    allowed = {"Table Look up", "Span Look up", "Paragraph Look up"}
    lookup = [item for item in scitat if item.get("question_type") in allowed]
    single_numeric = []
    for item in lookup:
        answers = item.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        numeric_answers = [normalize_text(answer) for answer in answers if is_strict_numeric_answer(answer)]
        if len(numeric_answers) == 1:
            single_numeric.append((item, numeric_answers[0]))

    def flatten_scitat_source(row: dict[str, Any]) -> str:
        parts: list[str] = []
        paragraph = row.get("paragraph")
        if isinstance(paragraph, dict):
            parts.append(str(paragraph.get("text", "")))
        elif paragraph:
            parts.append(str(paragraph))
        for table in row.get("tables") or []:
            parts.append(str(table.get("caption", "")))
            for table_row in table.get("table") or []:
                parts.append(" | ".join(str(cell) for cell in table_row))
        return "\n".join(parts)

    scitat_matched = [
        (item, answer)
        for item, answer in single_numeric
        if exact_or_normalized_in(answer, flatten_scitat_source(item))[1]
    ]
    scitat_steps = [
        ("official_test_items", len(scitat), len(scitat), "candidate_unit_start"),
        ("lookup_only_question_type_no_calculation_or_multistep", len(scitat), len(lookup), "eligibility_filter"),
        ("exactly_one_strict_numeric_final_answer", len(lookup), len(single_numeric), "eligibility_filter"),
        ("normalized_answer_present_in_supplied_table_or_text", len(single_numeric), len(scitat_matched), "eligibility_filter"),
    ]
    for order, (rule, input_n, remaining_n, role) in enumerate(scitat_steps, start=1):
        funnel.append({"dataset": "SciTaT", "stage_order": order, "rule": rule, "input_n": input_n, "excluded_n": input_n - remaining_n, "remaining_n": remaining_n, "stage_role": role})
    require(len(scitat_matched) == 5, "SciTaT numeric N=5 reproduction failed")

    matrix = [
        ("strict_numeric_schema", "numeric_recovery", "DIRECT_OVERLAP", "numeric_signature re-applies the same strict numeric predicate"),
        ("strict_numeric_schema", "exact_span_match", "NO_OVERLAP", "exact character containment is not required by numeric syntax"),
        ("strict_numeric_schema", "unit_preservation", "PARTIAL_OVERLAP", "numeric schema permits number-only, range, percent, unit, and statistical forms; unit is not mandatory"),
        ("strict_numeric_schema", "human_annotated_evidence_support", "NO_OVERLAP", "numeric syntax does not establish evidence support"),
        ("nonempty_human_annotated_evidence", "numeric_recovery", "NO_OVERLAP", "evidence presence does not determine numeric parsing"),
        ("nonempty_human_annotated_evidence", "exact_span_match", "PARTIAL_OVERLAP", "an evidence container is required but exact containment is not"),
        ("nonempty_human_annotated_evidence", "unit_preservation", "PARTIAL_OVERLAP", "unit/value checking is evaluated only inside required evidence"),
        ("nonempty_human_annotated_evidence", "human_annotated_evidence_support", "DIRECT_OVERLAP", "the metric uses the same required human evidence field"),
        ("normalized_answer_present_in_gold_evidence", "numeric_recovery", "NO_OVERLAP", "normalized containment does not determine numeric parsing"),
        ("normalized_answer_present_in_gold_evidence", "exact_span_match", "PARTIAL_OVERLAP", "eligibility accepts normalized containment; exact containment is a stricter diagnostic"),
        ("normalized_answer_present_in_gold_evidence", "unit_preservation", "DIRECT_OVERLAP", "reported unit_preserved is numeric signature plus the same normalized evidence match"),
        ("normalized_answer_present_in_gold_evidence", "human_annotated_evidence_support", "DIRECT_OVERLAP", "reported evidence support is the same normalized evidence-match gate"),
    ]
    matrix_lines = ["| eligibility_rule | evaluation_metric | overlap | reason |", "|---|---|---|---|"]
    matrix_lines.extend(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in matrix)
    markdown = """# T2 numeric eligibility and circularity audit

## QASPER

The frozen code iterates human extractive spans, retains spans satisfying the strict numeric/range/percent/number+unit/statistical predicate, requires nonempty human evidence, and requires normalized answer containment in at least one human-annotated evidence string. The final N=126 is reproduced exactly. Exact character containment is not an eligibility gate, but normalized containment is. An explicit unit is not required: number-only, range, percent, and statistical forms can qualify.

## SciTaT

The frozen code starts from official test items, retains lookup-only question types, requires exactly one strict numeric final answer, and requires normalized answer containment in the supplied table/text. The final N=5 is reproduced exactly. The small N is principally the result of a deliberately narrow, non-calculation lookup-only and recoverable-answer gate; it is not evidence that SciTaT itself contains only five numeric examples.

## Eligibility-by-metric overlap matrix

""" + "\n".join(matrix_lines) + """

## Interpretation

QASPER's four 1.0 values are best described as a **deterministic recoverability/consistency audit**, not external validator performance or independent accuracy. Numeric recovery, evidence support, and the reported unit-preservation computation directly reuse eligibility predicates; exact-span recovery is a stricter diagnostic than the normalized-containment gate but remains conditioned on that gate. SciTaT is likewise a narrow secondary recoverability diagnostic. The v86 files were unavailable, so their exact wording is `NOT RECOVERABLE`; any wording calling these four values independent external accuracy should be downgraded.
"""
    metadata = {
        "qasper_final_n": len(matched),
        "scitat_final_n": len(scitat_matched),
        "interpretation": "deterministic recoverability/consistency audit",
        "independent_validator_performance": False,
    }
    return pd.DataFrame(funnel), markdown, metadata


def build_summary(
    s2k: dict[str, Any],
    rq1_cluster: pd.DataFrame,
    mapping: dict[str, Any],
    mapping_sensitivity_meta: dict[str, Any],
    rq2: dict[str, Any],
    t2: dict[str, Any],
) -> str:
    rq1_views = rq1_cluster[rq1_cluster.row_type.eq("view_estimate")]
    rq1_best = {
        metric: str(group.loc[group.estimate.idxmax(), "view_a"])
        for metric, group in rq1_views.groupby("metric")
    }
    ratios = rq1_cluster[rq1_cluster.question_ci_width.notna()].ci_width_ratio_cluster_over_question
    changed = bool(rq1_cluster.conclusion_changed.fillna(False).any())
    task_labels = {"T1a_ACRONYM": "T1a", "T2_NUMERIC": "T2", "T3_CLAIM": "T3"}
    length_lines = []
    for task in TASKS:
        boot = rq2["bootstrap"][task]
        length_lines.append(
            f"- {task_labels[task]}: Length-only={rq2['length_capture_20'][task]:.6f}, "
            f"Task-specific={rq2['specific_capture_20'][task]:.6f}; Task-specific minus Length-only "
            f"95% CI [{boot['specific_low']:+.6f}, {boot['specific_high']:+.6f}]."
        )
    return f"""# Reviewer 2 final checks summary

## A. S2K sampling

1. The real algorithm was a seeded shuffle of the persisted 30,883 Gate-0 eligible paper IDs followed by the first-2,000 prefix.
2. Seed: `{s2k['seed']}`.
3. Membership was fixed before later T1a/T2/T3 yield modelling and source-suitability analysis.
4. Exact v86 wording is `NOT RECOVERABLE` because the named Word files were unavailable. Any description of S2K itself as stratified, topic-clustered, structural, or round-robin is inaccurate.
5. {s2k['imbalance_conclusion']}; maximum observed |SMD|={s2k['max_abs_smd']:.4f}, while several requested variables were not persisted for the full pool and were not imputed.

## B. RQ1 cluster robustness

6. CES@5 best view remains {rq1_best['CES@5']}.
7. BudgetCES best view remains {rq1_best['BudgetCES@512']}/{rq1_best['BudgetCES@1024']}/{rq1_best['BudgetCES@2048']} at 512/1024/2048.
8. Cluster/question CI-width ratios range from {ratios.min():.3f} to {ratios.max():.3f} (median {ratios.median():.3f}).
9. The fixed-Top-k versus fixed-token-budget reversal remains: B1 is best at CES@5 and B2 at all three token budgets. Any inferential conclusion changed: {str(changed).upper()}.

## C. Mapping sensitivity

10. Complete mapping among formal questions: {mapping['complete_questions_n']}/{1292} ({mapping['complete_questions_fraction']:.2%}). The global mapping audit is {mapping['global_mapped_gold_evidence_n']}/{mapping['global_original_gold_evidence_n']} ({mapping['global_mapping_rate']:.2%}); within the formal 1,292-question analysis it is {mapping['formal_1292_mapped_gold_evidence_n']}/{mapping['formal_1292_original_gold_evidence_n']} ({mapping['formal_1292_mapping_rate']:.2%}).
11. Complete-only best views: {mapping_sensitivity_meta['best_complete_mapping']}.
12. Core ranking changed: {'YES' if not (mapping_sensitivity_meta['topk_best_preserved'] and mapping_sensitivity_meta['budget_best_preserved']) else 'NO'}.
13. No material reversal is observed; maximum absolute view-estimate shift is {mapping_sensitivity_meta['max_absolute_view_estimate_change']:.4f}. This does not prove missing mappings are ignorable, but the primary ordering is not driven by incomplete questions.

## D. RQ2 length-only baseline

14-16. At the 20% paper budget:

{chr(10).join(length_lines)}

17. The RQ2 gains cannot be reduced to "selecting longer papers" where the paired CI excludes zero; where it crosses zero, superiority over length alone is not established.
18. Structural-feature language should remain observational/predictive and should not imply causal superiority. No new main-text number is automatically required; the full baseline belongs in the Supplement.

## E. T2 numeric audit

19. QASPER N=126 is the strict-numeric, nonempty-human-evidence, normalized-evidence-contained subset of human extractive spans.
20. Numeric recovery, evidence support, and the reported unit-preservation computation directly overlap eligibility; exact span is stricter than the normalized-match gate but conditioned on it.
21. Preferred label: `{t2['interpretation']}`.
22. Exact v86 wording is `NOT RECOVERABLE`; independent-validator-performance wording would be inaccurate.
23. Downgrade any external-accuracy claim to recoverability/consistency; no change to the frozen subset or values is needed.

## F. Submission judgment

24. Direct conflict with the frozen core result: NO. Two wording risks remain: attributing later stratification to S2K, and presenting T2 recoverability as independent accuracy.
25. Main text: the true S2K random-prefix provenance and the qualified Top-k/token-budget robustness statement.
26. Supplement: full cluster bootstrap, complete-mapping sensitivity, length-only curves, and eligibility funnels/matrix.
27. Wording-only: S2K stratification/nested-family chronology, T2 validation label, and noncausal RQ2 structural-feature interpretation.
28. Further mandatory low-cost experiments: NO; the requested evidence-chain checks are complete. Exact v86 wording still needs a manual wording pass once the two Word files are supplied.
29. From the experiment/statistics side, the paper can be frozen and moved to English drafting after that wording pass.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    s2k_md, s2k_comparison, s2k_meta = s2k_provenance_and_characteristics()
    question_metrics = load_rq1_question_metrics()
    rq1_cluster = cluster_bootstrap(question_metrics, "FULL_1292", True, RQ1_BOOTSTRAP_SEED)
    audit, mapping_meta = mapping_audit(question_metrics)
    mapping_sensitivity_frame, mapping_sensitivity_meta = mapping_sensitivity(question_metrics, audit)
    rq2_frame, rq2_meta = rq2_length_only()
    t2_funnel, t2_md, t2_meta = t2_numeric_audit()
    summary = build_summary(s2k_meta, rq1_cluster, mapping_meta, mapping_sensitivity_meta, rq2_meta, t2_meta)

    (output_dir / "s2k_sampling_provenance.md").write_text(s2k_md, encoding="utf-8")
    (output_dir / "t2_numeric_circularity_audit.md").write_text(t2_md, encoding="utf-8")
    (output_dir / "reviewer2_final_checks_summary.md").write_text(summary, encoding="utf-8")
    payload = {
        "csv": {
            "s2k_characteristic_comparison.csv": rows_for_json(s2k_comparison),
            "rq1_paper_cluster_bootstrap.csv": rows_for_json(rq1_cluster),
            "rq1_mapping_audit.csv": rows_for_json(audit),
            "rq1_complete_mapping_sensitivity.csv": rows_for_json(mapping_sensitivity_frame),
            "rq2_length_only_baseline.csv": rows_for_json(rq2_frame),
            "t2_numeric_eligibility_funnel.csv": rows_for_json(t2_funnel),
        },
        "metadata": {
            "s2k": s2k_meta,
            "mapping": mapping_meta,
            "mapping_sensitivity": mapping_sensitivity_meta,
            "rq2": rq2_meta,
            "t2": t2_meta,
            "rq1": {
                "questions": int(question_metrics.question_id.nunique()),
                "papers": int(question_metrics.paper_id.nunique()),
                "best_by_metric": {
                    metric: str(group.loc[group.estimate.idxmax(), "view_a"])
                    for metric, group in rq1_cluster[rq1_cluster.row_type.eq("view_estimate")].groupby("metric")
                },
                "any_conclusion_changed": bool(rq1_cluster.conclusion_changed.fillna(False).any()),
            },
            "v86_word_files_available": False,
            "new_llm_training": False,
            "new_retrieval": False,
        },
    }
    (output_dir / "_analysis_payload.json").write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
