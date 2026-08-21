#!/usr/bin/env python3
"""Submission-stage RQ2/RQ3 robustness analyses over frozen corpus_v1 assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats


EXPERIMENT_ID = "IPM_FINAL_ROBUSTNESS_V1"
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
TASK_LABELS = {"T1a_ACRONYM": "T1a", "T2_NUMERIC": "T2", "T3_CLAIM": "T3"}
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
DELTAS = (0.01, 0.02, 0.03, 0.05)
RANDOM_REPEATS = 10_000
BOOTSTRAP_REPLICATES = 10_000
RANDOM_SEED = 20260817
SEEDS = (42, 314159, 271828)
PHI_SCALES = {
    "T1a_ACRONYM": (1000, 2000),
    "T2_NUMERIC": (3000, 3678),
    "T3_CLAIM": (2000, 2286),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=convert) + "\n",
        encoding="utf-8",
    )


def manifest_entries(directory: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        entries[name.strip().replace("\\", "/")] = digest
    return entries


def verify_manifest_asset(directory: Path, relative: str) -> str:
    key = relative.replace("\\", "/")
    entries = manifest_entries(directory)
    if key not in entries:
        raise RuntimeError(f"Frozen checksum manifest has no entry for {directory / relative}")
    actual = sha256_file(directory / relative)
    if actual != entries[key]:
        raise RuntimeError(f"Frozen asset SHA-256 mismatch: {directory / relative}")
    return actual


def stable_order(scores: np.ndarray, paper_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((paper_ids.astype(str), -scores.astype(float)))


def selected_indices(
    scores: np.ndarray,
    costs: np.ndarray,
    paper_ids: np.ndarray,
    budget_type: str,
    fraction: float,
) -> np.ndarray:
    order = stable_order(scores, paper_ids)
    if budget_type == "paper_count":
        return order[: math.ceil(len(order) * fraction)]
    token_budget = math.floor(float(costs.sum()) * fraction)
    cumulative = np.cumsum(costs[order], dtype=np.int64)
    return order[cumulative <= token_budget]


def random_indices(
    rng: np.random.Generator,
    costs: np.ndarray,
    budget_type: str,
    fraction: float,
) -> np.ndarray:
    if budget_type == "paper_count":
        return rng.choice(len(costs), size=math.ceil(len(costs) * fraction), replace=False)
    order = rng.permutation(len(costs))
    token_budget = math.floor(float(costs.sum()) * fraction)
    cumulative = np.cumsum(costs[order], dtype=np.int64)
    return order[cumulative <= token_budget]


def capture(indices: np.ndarray, yields: np.ndarray) -> float:
    total = float(yields.sum())
    return float(yields[indices].sum() / total) if total > 0 else 0.0


def percentile_score(values: pd.Series) -> pd.Series:
    """Average-tie percentile rank mapped exactly to [0, 1]."""
    ranks = values.rank(method="average", ascending=True)
    return (ranks - 1.0) / (len(values) - 1.0)


def load_and_verify_inputs(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    base = root / "aicorpus-derived/experiments/corpus_v1"
    source = base / "source_suitability_utility_v1"
    scaling = base / "sft_scaling_v1"
    c2 = scaling / "full_grid_v1"
    c3 = scaling / "scale_extension_v1"
    phi = scaling / "second_model_phi4mini_v1"
    paths = {"source": source, "c2": c2, "c3": c3, "phi": phi}
    hashes: dict[str, str] = {}
    for label, directory, files in (
        ("source", source, ("oof_predictions.parquet", "selection_metrics.csv", "primary_bootstrap.csv", "cross_task_ranking.csv", "experiment_config.json")),
        ("c2", c2, ("practical_equivalence.csv", "n_star_summary.csv", "summary.json")),
        ("c3", c3, ("practical_equivalence.csv", "n_star_summary.csv", "plateau_summary.csv", "summary.json")),
        ("phi", phi, ("bootstrap.csv", "scale_metrics.csv", "summary.json")),
    ):
        for filename in files:
            hashes[f"{label}/{filename}"] = verify_manifest_asset(directory, filename)
    return paths, hashes


def prepare_oof(source_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof = pd.read_parquet(source_dir / "oof_predictions.parquet")
    if len(oof) != 1898 * 4 or oof.duplicated(["task", "paper_id"]).any():
        raise RuntimeError("OOF cardinality/uniqueness gate failed")
    if not bool(oof.model_converged.all()) or not np.isfinite(
        oof[["oof_predicted_count", "oof_predicted_rate_per_10k"]].to_numpy(float)
    ).all():
        raise RuntimeError("OOF convergence/finite-value gate failed")
    frames = {
        task: oof[oof.task.eq(task)].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        for task in TASKS
    }
    reference = frames[TASKS[0]]
    for task, frame in frames.items():
        if len(frame) != 1898:
            raise RuntimeError(f"Expected N=1,898 for {task}")
        if not frame.paper_id.astype(str).equals(reference.paper_id.astype(str)):
            raise RuntimeError(f"Paper membership drift for {task}")
        if not frame.source_id.astype(str).equals(reference.source_id.astype(str)):
            raise RuntimeError(f"Source membership drift for {task}")
        if not np.array_equal(frame.source_tokens.to_numpy(np.int64), reference.source_tokens.to_numpy(np.int64)):
            raise RuntimeError(f"Source-token drift for {task}")
    main = pd.concat([frames[task] for task in TASKS], ignore_index=True)
    diagnostics = {
        "effective_n": 1898,
        "rows_main_tasks": len(main),
        "paper_sets_identical": True,
        "source_ids_identical": True,
        "source_tokens_identical": True,
        "all_oof_models_converged": True,
        "folds": sorted(int(value) for value in oof.fold.unique()),
    }
    return main, diagnostics


def score_tables(oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    count = oof.pivot(index="paper_id", columns="task", values="oof_predicted_count").sort_index()
    rate = oof.pivot(index="paper_id", columns="task", values="oof_predicted_rate_per_10k").sort_index()
    common_count = pd.concat([percentile_score(count[task]) for task in TASKS], axis=1).mean(axis=1)
    common_rate = pd.concat([percentile_score(rate[task]) for task in TASKS], axis=1).mean(axis=1)
    return {"count": count, "rate": rate, "common_count": common_count, "common_rate": common_rate}


def frozen_20_sanity(oof: pd.DataFrame, scores: dict[str, pd.DataFrame], source_dir: Path) -> dict[str, Any]:
    frozen = pd.read_csv(source_dir / "selection_metrics.csv")
    checks: list[dict[str, Any]] = []
    for task in TASKS:
        frame = oof[oof.task.eq(task)].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        ids = frame.paper_id.astype(str).to_numpy()
        costs = frame.source_tokens.to_numpy(np.int64)
        yields = frame.actual_mv_count.to_numpy(float)
        for budget_type, old_budget in (("paper_count", "paper"), ("source_token", "token")):
            score = scores["count" if budget_type == "paper_count" else "rate"].loc[ids, task].to_numpy(float)
            selected = selected_indices(score, costs, ids, budget_type, 0.20)
            current = {
                "capture": capture(selected, yields),
                "selected_papers": len(selected),
                "captured_instances": float(yields[selected].sum()),
                "selected_tokens": float(costs[selected].sum()),
            }
            target = frozen[
                frozen.task.eq(task)
                & frozen.strategy.eq("TASK_SPECIFIC")
                & frozen.budget_type.eq(old_budget)
                & np.isclose(frozen.budget_fraction, 0.20)
            ].iloc[0]
            exact = (
                abs(current["capture"] - float(target.yield_capture)) <= 1e-12
                and current["selected_papers"] == int(target.n_selected)
                and current["captured_instances"] == float(target.actual_mv_yield)
                and current["selected_tokens"] == float(target.source_tokens_selected)
            )
            if not exact:
                raise RuntimeError(f"Frozen 20% task-specific regression failed: {task}/{budget_type}")

            # Reproduce the historical 1,000-repeat random row exactly before increasing to 10,000.
            rng = np.random.default_rng(
                RANDOM_SEED + sum(ord(c) for c in task) + 200 + (0 if old_budget == "paper" else 10000)
            )
            random_rows = []
            for _ in range(1000):
                idx = random_indices(rng, costs, budget_type, 0.20)
                random_rows.append(
                    (
                        len(idx),
                        float(costs[idx].sum()),
                        float(yields[idx].sum()),
                        capture(idx, yields),
                    )
                )
            random_mean = np.asarray(random_rows, dtype=float).mean(axis=0)
            old_random = frozen[
                frozen.task.eq(task)
                & frozen.strategy.eq("RANDOM")
                & frozen.budget_type.eq(old_budget)
                & np.isclose(frozen.budget_fraction, 0.20)
            ].iloc[0]
            random_exact = (
                abs(random_mean[0] - float(old_random.n_selected)) <= 1e-12
                and abs(random_mean[1] - float(old_random.source_tokens_selected)) <= 1e-8
                and abs(random_mean[2] - float(old_random.actual_mv_yield)) <= 1e-12
                and abs(random_mean[3] - float(old_random.yield_capture)) <= 1e-12
            )
            if not random_exact:
                raise RuntimeError(f"Frozen 20% random regression failed: {task}/{budget_type}")
            checks.append(
                {
                    "task": task,
                    "budget_type": budget_type,
                    "capture": current["capture"],
                    "random_capture_1000": float(random_mean[3]),
                    "task_specific_exact": exact,
                    "random_1000_exact": random_exact,
                }
            )
    return {"passed": True, "checks": checks}


def qwen_delta_inputs(paths: dict[str, Path]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    c2 = pd.read_csv(paths["c2"] / "practical_equivalence.csv")
    c3 = pd.read_csv(paths["c3"] / "practical_equivalence.csv")
    tables = {
        "T1a_ACRONYM": c2[c2.task.eq("T1a_ACRONYM")].rename(
            columns={"point_N2000_minus_N": "reference_minus_scale"}
        ),
        "T2_NUMERIC": c3[c3.task.eq("T2_NUMERIC")].rename(
            columns={"mean_NEM_difference_reference_minus_scale": "reference_minus_scale"}
        ),
        "T3_CLAIM": c3[c3.task.eq("T3_CLAIM")].rename(
            columns={"mean_NEM_difference_reference_minus_scale": "reference_minus_scale"}
        ),
    }
    c2_summary = json.loads((paths["c2"] / "summary.json").read_text(encoding="utf-8"))
    c3_summary = json.loads((paths["c3"] / "summary.json").read_text(encoding="utf-8"))
    plateau = {
        "T1a_ACRONYM": {
            "largest_minus_previous": float(
                next(item for item in c2_summary["plateau"] if item["task"] == "T1a_ACRONYM")["N2000_minus_N1000"]
            ),
            "capacity_limited": False,
        }
    }
    for item in c3_summary["plateau"]:
        plateau[item["task"]] = {
            "largest_minus_previous": float(item["largest_minus_previous"]),
            "capacity_limited": bool(item["capacity_limited"]),
        }
    return tables, plateau


def qwen_delta_sanity(paths: dict[str, Path]) -> dict[str, Any]:
    tables, _ = qwen_delta_inputs(paths)
    expected = {
        "T1a_ACRONYM": (2000, 1000, "IDENTIFIED"),
        "T2_NUMERIC": (3678, 3000, "IDENTIFIED"),
        "T3_CLAIM": (2286, None, "NOT_REACHED"),
    }
    rows = []
    for task, table in tables.items():
        table = table.sort_values("scale")
        eligible = table[table.one_sided_ucb95 <= 0.02 + 1e-15]
        n_star = int(eligible.scale.iloc[0]) if len(eligible) else None
        status = "IDENTIFIED" if n_star is not None else "NOT_REACHED"
        reference = int(table.reference_scale.iloc[0])
        if (reference, n_star, status) != expected[task]:
            raise RuntimeError(f"Frozen delta=0.02 n* regression failed for {task}")
        rows.append({"task": task, "reference_n": reference, "n_star": n_star, "status": status})
    return {"passed": True, "rows": rows}


def bootstrap_specific_common_20(
    task: str,
    budget_type: str,
    frame: pd.DataFrame,
    specific_scores: np.ndarray,
    common_scores: np.ndarray,
) -> dict[str, float]:
    ids_all = frame.paper_id.astype(str).to_numpy()
    costs_all = frame.source_tokens.to_numpy(np.int64)
    yields_all = frame.actual_mv_count.to_numpy(float)
    rng = np.random.default_rng(
        int(sha256_text(f"IPM_FINAL_RQ2_BOOTSTRAP_V1|{task}|{budget_type}")[:16], 16)
    )
    specific_values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    common_values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(frame), size=len(frame), dtype=np.int32)
        yields = yields_all[sample]
        costs = costs_all[sample]
        ids = np.asarray([f"{position:04d}|{ids_all[source]}" for position, source in enumerate(sample)], dtype=str)
        specific = selected_indices(specific_scores[sample], costs, ids, budget_type, 0.20)
        common = selected_indices(common_scores[sample], costs, ids, budget_type, 0.20)
        total = float(yields.sum())
        specific_values[index] = float(yields[specific].sum() / total)
        common_values[index] = float(yields[common].sum() / total)
        differences[index] = specific_values[index] - common_values[index]
    return {
        "specific_ci_low": float(np.quantile(specific_values, 0.025)),
        "specific_ci_high": float(np.quantile(specific_values, 0.975)),
        "common_ci_low": float(np.quantile(common_values, 0.025)),
        "common_ci_high": float(np.quantile(common_values, 0.975)),
        "gain_ci_low": float(np.quantile(differences, 0.025)),
        "gain_ci_high": float(np.quantile(differences, 0.975)),
    }


def rq2_analysis(oof: pd.DataFrame, scores: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bootstrap_20: dict[tuple[str, str], dict[str, float]] = {}
    for task in TASKS:
        frame = oof[oof.task.eq(task)].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        ids = frame.paper_id.astype(str).to_numpy()
        costs = frame.source_tokens.to_numpy(np.int64)
        yields = frame.actual_mv_count.to_numpy(float)
        for budget_type in ("paper_count", "source_token"):
            score_kind = "count" if budget_type == "paper_count" else "rate"
            specific_scores = scores[score_kind].loc[ids, task].to_numpy(float)
            common_scores = scores[f"common_{score_kind}"].loc[ids].to_numpy(float)
            bootstrap_20[(task, budget_type)] = bootstrap_specific_common_20(
                task, budget_type, frame, specific_scores, common_scores
            )
            for fraction in BUDGETS:
                deterministic: dict[str, tuple[np.ndarray, str]] = {
                    "task_specific": (
                        selected_indices(specific_scores, costs, ids, budget_type, fraction),
                        f"oof_predicted_{'count' if score_kind == 'count' else 'rate_per_10k'}",
                    ),
                    "common": (
                        selected_indices(common_scores, costs, ids, budget_type, fraction),
                        f"mean_average_tie_percentiles_of_three_oof_predicted_{score_kind}_scores",
                    ),
                }
                common_capture = capture(deterministic["common"][0], yields)
                for strategy, (selected, score_description) in deterministic.items():
                    cap = capture(selected, yields)
                    boot = bootstrap_20[(task, budget_type)] if math.isclose(fraction, 0.20) else None
                    if boot is None:
                        ci_low = ci_high = np.nan
                    elif strategy == "task_specific":
                        ci_low, ci_high = boot["specific_ci_low"], boot["specific_ci_high"]
                    else:
                        ci_low, ci_high = boot["common_ci_low"], boot["common_ci_high"]
                    rows.append(
                        {
                            "task": task,
                            "budget_type": budget_type,
                            "budget_fraction": fraction,
                            "strategy": strategy,
                            "ranking_score": score_description,
                            "selected_papers": float(len(selected)),
                            "actual_token_fraction": float(costs[selected].sum() / costs.sum()),
                            "captured_instances": float(yields[selected].sum()),
                            "total_instances": float(yields.sum()),
                            "capture_fraction": cap,
                            "lift": cap / fraction,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "specific_gain_over_common": cap - common_capture if strategy == "task_specific" else np.nan,
                            "relative_gain": (cap - common_capture) / common_capture if strategy == "task_specific" and common_capture > 0 else np.nan,
                            "gain_ci_low": boot["gain_ci_low"] if strategy == "task_specific" and boot else np.nan,
                            "gain_ci_high": boot["gain_ci_high"] if strategy == "task_specific" and boot else np.nan,
                            "random_repeats": np.nan,
                        }
                    )

                rng = np.random.default_rng(
                    RANDOM_SEED
                    + sum(ord(c) for c in task)
                    + int(fraction * 1000)
                    + (0 if budget_type == "paper_count" else 10000)
                )
                random_values = np.empty((RANDOM_REPEATS, 4), dtype=float)
                for index in range(RANDOM_REPEATS):
                    selected = random_indices(rng, costs, budget_type, fraction)
                    random_values[index] = (
                        len(selected),
                        float(costs[selected].sum() / costs.sum()),
                        float(yields[selected].sum()),
                        capture(selected, yields),
                    )
                mean = random_values.mean(axis=0)
                rows.append(
                    {
                        "task": task,
                        "budget_type": budget_type,
                        "budget_fraction": fraction,
                        "strategy": "random",
                        "ranking_score": "uniform_random_paper_order",
                        "selected_papers": mean[0],
                        "actual_token_fraction": mean[1],
                        "captured_instances": mean[2],
                        "total_instances": float(yields.sum()),
                        "capture_fraction": mean[3],
                        "lift": mean[3] / fraction,
                        "ci_low": float(np.quantile(random_values[:, 3], 0.025)),
                        "ci_high": float(np.quantile(random_values[:, 3], 0.975)),
                        "specific_gain_over_common": np.nan,
                        "relative_gain": np.nan,
                        "gain_ci_low": np.nan,
                        "gain_ci_high": np.nan,
                        "random_repeats": RANDOM_REPEATS,
                    }
                )
    curves = pd.DataFrame(rows).sort_values(
        ["task", "budget_type", "budget_fraction", "strategy"], kind="mergesort"
    ).reset_index(drop=True)
    common = curves[np.isclose(curves.budget_fraction, 0.20)].copy().reset_index(drop=True)
    return common, curves


def ranking_overlap(oof: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    rate = oof.pivot(index="paper_id", columns="task", values="oof_predicted_rate_per_10k").sort_index()
    frozen = pd.read_csv(source_dir / "cross_task_ranking.csv")
    rows = []
    for left_index, left in enumerate(TASKS):
        for right in TASKS[left_index + 1 :]:
            spearman = float(stats.spearmanr(rate[left], rate[right]).statistic)
            old = frozen[frozen.task_a.eq(left) & frozen.task_b.eq(right)].iloc[0]
            if abs(spearman - float(old.spearman)) > 1e-12:
                raise RuntimeError(f"Frozen Spearman regression failed: {left}/{right}")
            for fraction in (0.05, 0.10, 0.20, 0.30):
                n = math.ceil(len(rate) * fraction)
                left_ids = set(rate.sort_values([left], ascending=False, kind="mergesort").index[:n])
                right_ids = set(rate.sort_values([right], ascending=False, kind="mergesort").index[:n])
                jaccard = len(left_ids & right_ids) / len(left_ids | right_ids)
                if fraction in (0.10, 0.20, 0.30):
                    expected = float(old[f"top{int(fraction * 100)}_jaccard"])
                    if abs(jaccard - expected) > 1e-12:
                        raise RuntimeError(f"Frozen Jaccard regression failed: {left}/{right}/{fraction}")
                rows.append(
                    {
                        "task_pair": f"{left}__{right}",
                        "task_a": left,
                        "task_b": right,
                        "score": "oof_predicted_rate_per_10k",
                        "spearman": spearman,
                        "top_fraction": fraction,
                        "jaccard": jaccard,
                    }
                )
    return pd.DataFrame(rows)


def phi_bootstrap(paths: dict[str, Path]) -> dict[str, dict[str, float]]:
    phi = paths["phi"]
    output: dict[str, dict[str, float]] = {}
    for task in TASKS:
        small, large = PHI_SCALES[task]
        arrays: dict[tuple[int, int], np.ndarray] = {}
        ids: list[str] | None = None
        for scale in (small, large):
            for seed in SEEDS:
                short = TASK_LABELS[task].lower()
                directory = phi / "runs" / f"{short}_n{scale}_seed{seed}"
                relative = (directory / "predictions_primary.parquet").relative_to(phi).as_posix()
                verify_manifest_asset(phi, relative)
                frame = pd.read_parquet(directory / "predictions_primary.parquet").sort_values("instance_id").reset_index(drop=True)
                current_ids = frame.instance_id.astype(str).tolist()
                if ids is None:
                    ids = current_ids
                elif current_ids != ids:
                    raise RuntimeError(f"Phi paired Eval ID drift: {task}")
                arrays[(scale, seed)] = frame.correct_NEM.to_numpy(float)
        size = len(ids or [])
        rng = np.random.default_rng(int(sha256_text(f"C5_PHI4MINI_BOOTSTRAP_V1|{task}")[:16], 16))
        values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for index in range(BOOTSTRAP_REPLICATES):
            drawn_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
            components = []
            for seed_index in drawn_seeds:
                seed = SEEDS[int(seed_index)]
                examples = rng.integers(0, size, size=size)
                components.append(
                    float((arrays[(large, seed)][examples] - arrays[(small, seed)][examples]).mean())
                )
            values[index] = float(np.mean(components))
        point = float(
            np.mean([arrays[(large, seed)].mean() - arrays[(small, seed)].mean() for seed in SEEDS])
        )
        frozen = pd.read_csv(phi / "bootstrap.csv")
        old = frozen[frozen.task.eq(task)].iloc[0]
        if (
            abs(point - float(old.point_difference)) > 1e-12
            or abs(float(np.quantile(values, 0.025)) - float(old.ci95_low)) > 1e-12
            or abs(float(np.quantile(values, 0.975)) - float(old.ci95_high)) > 1e-12
        ):
            raise RuntimeError(f"Phi bootstrap regression failed: {task}")
        output[task] = {
            "point": point,
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "one_sided_ucb95": float(np.quantile(values, 0.95)),
        }
    return output


def rq3_delta_analysis(paths: dict[str, Path]) -> pd.DataFrame:
    qwen_tables, qwen_plateau = qwen_delta_inputs(paths)
    phi = phi_bootstrap(paths)
    rows: list[dict[str, Any]] = []
    for task, table in qwen_tables.items():
        table = table.sort_values("scale")
        reference = int(table.reference_scale.iloc[0])
        for delta in DELTAS:
            eligible = table[table.one_sided_ucb95 <= delta + 1e-15]
            if len(eligible):
                selected = eligible.iloc[0]
                n_star: float | None = int(selected.scale)
                status = "IDENTIFIED"
                ucb: float | None = float(selected.one_sided_ucb95)
                point: float | None = float(selected.reference_minus_scale)
                ci_low: float | None = float(selected.ci95_low)
                ci_high: float | None = float(selected.ci95_high)
            else:
                n_star = ucb = point = ci_low = ci_high = None
                status = "NOT_REACHED"
            plateau_difference = qwen_plateau[task]["largest_minus_previous"]
            plateau_status = (
                "REFERENCE_NOT_PLATEAUED" if plateau_difference > delta + 1e-15 else "REFERENCE_PLATEAU_COMPATIBLE"
            )
            if plateau_status == "REFERENCE_NOT_PLATEAUED" and qwen_plateau[task]["capacity_limited"]:
                plateau_conclusion = "PLATEAU_NOT_IDENTIFIED_WITHIN_AVAILABLE_POOL"
            else:
                plateau_conclusion = ""
            rows.append(
                {
                    "model": "Qwen2.5-3B-Instruct",
                    "task": task,
                    "delta": delta,
                    "reference_n": reference,
                    "n_star": n_star,
                    "status": status,
                    "reference_minus_n_star": point,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "one_sided_ucb95": ucb,
                    "available_candidate_scales": ";".join(str(int(value)) for value in table.scale),
                    "largest_minus_previous": plateau_difference,
                    "plateau_status": plateau_status,
                    "plateau_conclusion": plateau_conclusion,
                    "evidence_scope": "full_frozen_scale_grid",
                }
            )
    for task in TASKS:
        small, reference = PHI_SCALES[task]
        result = phi[task]
        for delta in DELTAS:
            identified = result["one_sided_ucb95"] <= delta + 1e-15
            plateau_status = (
                "REFERENCE_NOT_PLATEAUED" if result["point"] > delta + 1e-15 else "REFERENCE_PLATEAU_COMPATIBLE"
            )
            rows.append(
                {
                    "model": "Phi-4-mini-instruct",
                    "task": task,
                    "delta": delta,
                    "reference_n": reference,
                    "n_star": small if identified else None,
                    "status": "IDENTIFIED" if identified else "NOT_REACHED",
                    "reference_minus_n_star": result["point"] if identified else None,
                    "ci95_low": result["ci95_low"] if identified else None,
                    "ci95_high": result["ci95_high"] if identified else None,
                    "one_sided_ucb95": result["one_sided_ucb95"] if identified else None,
                    "available_candidate_scales": str(small),
                    "largest_minus_previous": result["point"],
                    "plateau_status": plateau_status,
                    "plateau_conclusion": "",
                    "evidence_scope": "limited_two_point_replication",
                }
            )
    output = pd.DataFrame(rows).sort_values(["model", "task", "delta"], kind="mergesort").reset_index(drop=True)
    qwen_02 = output[output.model.eq("Qwen2.5-3B-Instruct") & np.isclose(output.delta, 0.02)]
    expected = {
        "T1a_ACRONYM": (1000.0, "IDENTIFIED"),
        "T2_NUMERIC": (3000.0, "IDENTIFIED"),
        "T3_CLAIM": (np.nan, "NOT_REACHED"),
    }
    for task, (n_star, status) in expected.items():
        row = qwen_02[qwen_02.task.eq(task)].iloc[0]
        if row.status != status or ((pd.isna(n_star) and not pd.isna(row.n_star)) or (not pd.isna(n_star) and row.n_star != n_star)):
            raise RuntimeError(f"Final delta=0.02 regression failed: {task}")
    return output


def scientific_summary(
    diagnostics: dict[str, Any],
    sanity_20: dict[str, Any],
    curves: pd.DataFrame,
    overlap: pd.DataFrame,
    delta: pd.DataFrame,
    input_paths: dict[str, Path],
) -> str:
    lines = [
        "# IPM final robustness",
        "",
        "## Data and reproduction",
        "",
        f"- RQ2 inputs: `{(input_paths['source'] / 'oof_predictions.parquet').as_posix()}`, `selection_metrics.csv`, and `cross_task_ranking.csv`.",
        f"- RQ3 inputs: `{(input_paths['c2'] / 'practical_equivalence.csv').as_posix()}`, `{(input_paths['c3'] / 'practical_equivalence.csv').as_posix()}`, and Phi `bootstrap.csv` plus the 18 frozen primary-prediction files.",
        f"- Effective source-paper N remained {diagnostics['effective_n']:,}; task memberships and source-token values were identical.",
        "- The frozen 20% paper-count and source-token task-specific results, including the historical 1,000-repeat random baselines, were reproduced exactly.",
        "- No data-version or membership inconsistency was found.",
        "",
        "## Task-specific versus Common",
        "",
        "Common is the mean of the three average-tie OOF percentile ranks, using count scores for paper budgets and rate-per-10k scores for token budgets; each is shared across tasks.",
        "",
    ]
    primary = curves[np.isclose(curves.budget_fraction, 0.20) & curves.strategy.eq("task_specific")]
    for _, row in primary.iterrows():
        if row.gain_ci_low > 0:
            judgment = "clearly better"
        elif row.gain_ci_high < 0:
            judgment = "clearly worse"
        else:
            judgment = "not clearly different"
        lines.append(
            f"- {TASK_LABELS[row.task]}, {row.budget_type}: gain={row.specific_gain_over_common:+.4f} "
            f"(95% CI [{row.gain_ci_low:+.4f}, {row.gain_ci_high:+.4f}]); {judgment}."
        )
    lines.extend(["", "## Budget pattern", ""])
    for task in TASKS:
        statements = []
        for budget_type in ("paper_count", "source_token"):
            group = curves[curves.task.eq(task) & curves.budget_type.eq(budget_type) & curves.strategy.eq("task_specific")]
            gain = group.set_index("budget_fraction").specific_gain_over_common
            best_fraction = float(gain.idxmax())
            signs = "positive at all six budgets" if bool((gain > 0).all()) else "not positive at all six budgets"
            statements.append(f"{budget_type}: peak gain at {best_fraction:.0%}, {signs}")
        lines.append(f"- {TASK_LABELS[task]} — " + "; ".join(statements) + ".")
    lines.append("- Lift generally declined as budgets grew, so prioritization gains were concentrated at lower-to-mid budgets; 20% is a representative interior point, not an optimized universal budget.")
    lines.extend(["", "## Ranking overlap", ""])
    for pair in overlap.task_pair.unique():
        group = overlap[overlap.task_pair.eq(pair)]
        row = group.iloc[0]
        jaccard = group.set_index("top_fraction").jaccard
        lines.append(
            f"- {TASK_LABELS[row.task_a]}–{TASK_LABELS[row.task_b]}: Spearman={row.spearman:.3f}; "
            f"Top-5/10/20/30% Jaccard={jaccard.loc[0.05]:.3f}/{jaccard.loc[0.10]:.3f}/{jaccard.loc[0.20]:.3f}/{jaccard.loc[0.30]:.3f}."
        )
    lines.extend(["", "## Delta sensitivity", ""])
    for model in ("Qwen2.5-3B-Instruct", "Phi-4-mini-instruct"):
        lines.append(f"- {model}:")
        for task in TASKS:
            group = delta[delta.model.eq(model) & delta.task.eq(task)].sort_values("delta")
            values = [f"δ={row.delta:.2f}: {'N'+str(int(row.n_star)) if pd.notna(row.n_star) else 'NOT REACHED'}" for _, row in group.iterrows()]
            suffix = " (two-point evidence only)" if model.startswith("Phi") else ""
            lines.append(f"  - {TASK_LABELS[task]}: " + "; ".join(values) + suffix + ".")
    t3 = delta[delta.model.eq("Qwen2.5-3B-Instruct") & delta.task.eq("T3_CLAIM")]
    conflict = bool((t3.status != "NOT_REACHED").any())
    lines.extend(
        [
            "",
            "## Submission judgment",
            "",
            "- Overall, the analyses support task-dependent source prioritization, but not universal superiority of task-specific ranking: at 20%, T3 is clearly better under both budgets and T2 under paper count, whereas T1a and T2 token are not clearly different from Common.",
            "- The 20% Specific–Common comparison and qualitative delta result merit main-text mention. Full curves, overlap diagnostics, and two-point Phi sensitivity belong in the Supplement.",
            "- The Qwen T1a/T2 plateau versus T3 not-yet-plateaued distinction remains directionally valid across the requested margins, while exact n* is margin-dependent; this is qualitative robustness rather than n* invariance.",
            f"- Core v84 contradiction detected: {'yes' if conflict else 'no'}. No manuscript file was modified.",
            "- Wording should avoid implying that task-specific ranking always beats a fair Common baseline. Abstract-level numeric additions are not warranted; retain only the qualified qualitative claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--plot-python", type=Path)
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[3]
    output_dir = root / "results/ipm_final_robustness"
    paths, input_hashes = load_and_verify_inputs(root)
    oof, diagnostics = prepare_oof(paths["source"])
    scores = score_tables(oof)
    sanity_20 = frozen_20_sanity(oof, scores, paths["source"])
    delta_sanity = qwen_delta_sanity(paths)
    sanity = {
        "status": "PASS",
        "effective_n": diagnostics["effective_n"],
        "frozen_20_percent": sanity_20,
        "delta_0_02": delta_sanity,
    }
    if args.sanity_only:
        print(
            json.dumps(
                sanity,
                ensure_ascii=False,
                indent=2,
                default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
            )
        )
        return
    if output_dir.exists():
        incomplete = not (output_dir / "analysis_config.json").exists() and not (output_dir / "checksums.sha256").exists()
        allowed = {
            "rq2_common_vs_specific.csv",
            "rq2_budget_curves.csv",
            "rq2_ranking_overlap.csv",
            "rq3_delta_sensitivity.csv",
            "fig_rq2_paper_budget.svg",
            "fig_rq2_paper_budget.pdf",
            "fig_rq2_token_budget.svg",
            "fig_rq2_token_budget.pdf",
        }
        present = {path.name for path in output_dir.iterdir() if path.is_file()}
        if not args.resume_incomplete or not incomplete or not present <= allowed:
            raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    common, curves = rq2_analysis(oof, scores)
    overlap = ranking_overlap(oof, paths["source"])
    delta = rq3_delta_analysis(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    common.to_csv(output_dir / "rq2_common_vs_specific.csv", index=False, encoding="utf-8", lineterminator="\n")
    curves.to_csv(output_dir / "rq2_budget_curves.csv", index=False, encoding="utf-8", lineterminator="\n")
    overlap.to_csv(output_dir / "rq2_ranking_overlap.csv", index=False, encoding="utf-8", lineterminator="\n")
    delta.to_csv(output_dir / "rq3_delta_sensitivity.csv", index=False, encoding="utf-8", lineterminator="\n")
    if args.plot_python is None or not args.plot_python.is_file():
        raise RuntimeError("A Python runtime with matplotlib is required via --plot-python")
    plot_script = root / "code/corpus_experiments_v1/sft/plot_ipm_final_robustness.py"
    plot_environment = os.environ.copy()
    for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        plot_environment.pop(variable, None)
    plot_run = subprocess.run(
        [str(args.plot_python), str(plot_script), "--input", str(output_dir / "rq2_budget_curves.csv"), "--output", str(output_dir)],
        check=False,
        capture_output=True,
        text=True,
        cwd=root,
        env=plot_environment,
    )
    if plot_run.returncode != 0:
        raise RuntimeError(f"Figure rendering failed: {plot_run.stderr.strip()}")
    plot_metadata = json.loads(plot_run.stdout.strip())
    (output_dir / "robustness_summary.md").write_text(
        scientific_summary(diagnostics, sanity_20, curves, overlap, delta, paths), encoding="utf-8"
    )
    config = {
        "experiment_id": EXPERIMENT_ID,
        "status": "SUCCESS",
        "frozen_input_hashes": input_hashes,
        "sanity_gate": sanity,
        "rq2": {
            "tasks": list(TASKS),
            "budgets": list(BUDGETS),
            "paper_budget_rounding": "ceil(fraction * 1898)",
            "token_budget": "floor(fraction * total source tokens); whole documents; stop before first exceedance",
            "task_specific_paper_score": "oof_predicted_count",
            "task_specific_token_score": "oof_predicted_rate_per_10k",
            "common_definition": "within each frozen budget score type, mean of T1a/T2/T3 average-tie OOF percentile ranks mapped to [0,1]; one cross-task ranking per budget type",
            "tie_break": "paper_id ascending after descending score",
            "random_repeats": RANDOM_REPEATS,
            "random_seed": RANDOM_SEED,
            "specific_common_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap": "paper resampling with selection recomputed; paired specific-minus-common at 20%",
            "lift": "capture_fraction / nominal_budget_fraction",
        },
        "rq3": {
            "deltas": list(DELTAS),
            "criterion": "smallest candidate with frozen/recomputed one-sided UCB95(reference NEM - candidate NEM) <= delta",
            "qwen": "reuse frozen C2/C3 10,000-replicate hierarchical bootstrap outputs",
            "phi": "recompute original C5 10,000-replicate bootstrap distribution from frozen primary predictions to recover one-sided UCB95; no missing scales added",
        },
        "runtime_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "figure_renderer": plot_metadata["renderer"],
            "reportlab": plot_metadata["reportlab"],
        },
        "manuscript_modified": False,
        "llm_training_run": False,
    }
    write_json(output_dir / "analysis_config.json", config)
    write_checksums(output_dir)
    print(json.dumps({"status": "SUCCESS", "output": output_dir.as_posix(), "runtime_seconds": time.time() - started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
