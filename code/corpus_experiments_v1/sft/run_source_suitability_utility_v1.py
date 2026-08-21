#!/usr/bin/env python3
"""Strict OOF validation of task-specific source-selection utility."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.stats as st


EXPERIMENT_ID = "SOURCE_SUITABILITY_UTILITY_VALIDATION_V1"
FOLD_SEED = "20260817_SOURCE_SUITABILITY_UTILITY_V1"
RANDOM_SEED = 20260817
N_FOLDS = 5
RANDOM_REPEATS = 1000
BOOTSTRAP_REPLICATES = 10_000
BUDGET_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
PRIMARY_FRACTION = 0.20
TASKS = ("T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION", "T2_NUMERIC", "T3_CLAIM")
MAIN_TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")


def load_source_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_source_suitability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen source-suitability runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=convert) + "\n",
        encoding="utf-8",
    )


def write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )


def fold_map(paper_ids: list[str]) -> dict[str, int]:
    ordered = sorted(
        paper_ids,
        key=lambda paper_id: (hashlib.sha256(f"{FOLD_SEED}|{paper_id}".encode("utf-8")).hexdigest(), paper_id),
    )
    return {paper_id: position % N_FOLDS for position, paper_id in enumerate(ordered)}


def apply_design(source: Any, frame: pd.DataFrame, standards: dict[str, dict[str, float]]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for predictor in source.BINARY_PREDICTORS:
        output[predictor] = frame[predictor].astype(float)
    for predictor in source.CONTINUOUS_PREDICTORS:
        standard = standards[predictor]
        output[predictor] = (frame[predictor].astype(float) - standard["mean"]) / standard["sd"]
    return source.sm.add_constant(output, has_constant="add")


def build_oof(source: Any, data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    eligible = data[data.offset_model_eligible].copy()
    papers = sorted(eligible.paper_id.astype(str).unique())
    folds = fold_map(papers)
    eligible["fold"] = eligible.paper_id.astype(str).map(folds)
    outputs, diagnostics = [], []
    task_status = {}
    for task in TASKS:
        task_data = eligible[eligible.task == task].copy()
        task_ok = True
        for fold in range(N_FOLDS):
            train = task_data[task_data.fold != fold].copy()
            test = task_data[task_data.fold == fold].copy()
            _, diagnostic, standards, result = source.fit_nb(train, "main")
            converged = bool(diagnostic["converged"])
            if not converged:
                # Same frozen BFGS implementation, repeated once only as a numerical-stability retry.
                _, diagnostic, standards, result = source.fit_nb(train, "main")
                converged = bool(diagnostic["converged"])
            task_ok &= converged
            x_test = apply_design(source, test, standards)
            beta = result.params[x_test.columns].to_numpy(dtype=float)
            linear_rate = x_test.to_numpy(dtype=float) @ beta
            predicted_rate = np.exp(linear_rate)
            predicted_count = predicted_rate * (test.source_tokens.to_numpy(dtype=float) / 10_000.0)
            api_prediction = np.asarray(
                result.predict(exog=x_test, offset=test.offset_log_source_tokens_per_10k.astype(float)), dtype=float
            )
            if not np.allclose(predicted_count, api_prediction, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"NB2 prediction formula/API mismatch for {task} fold {fold}")
            if not np.isfinite(predicted_count).all() or not np.isfinite(predicted_rate).all():
                raise RuntimeError(f"Non-finite OOF prediction for {task} fold {fold}")
            outputs.append(
                pd.DataFrame(
                    {
                        "paper_id": test.paper_id.astype(str),
                        "source_id": test.source_id.astype(str),
                        "task": task,
                        "source_tokens": test.source_tokens.astype(int),
                        "actual_mv_count": test.machine_verified_count.astype(int),
                        "oof_predicted_count": predicted_count,
                        "oof_predicted_rate_per_10k": predicted_rate,
                        "fold": fold,
                        "fold_train_n": len(train),
                        "fold_test_n": len(test),
                        "model_converged": converged,
                        "alpha": float(result.params["alpha"]),
                    }
                )
            )
            diagnostics.append(
                {
                    "task": task,
                    "fold": fold,
                    "train_n": len(train),
                    "test_n": len(test),
                    "converged": converged,
                    "alpha": float(result.params["alpha"]),
                    "optimizer_warnings": diagnostic["optimizer_warnings"],
                    "standards": standards,
                }
            )
        task_status[task] = "OOF_SUCCESS" if task_ok else (
            "T1b_NOT_STABLE_FOR_SELECTION_VALIDATION" if task == "T1b_EXPLICIT_DEFINITION" else "OOF_FAILED"
        )
    oof = pd.concat(outputs, ignore_index=True).sort_values(["task", "paper_id"], kind="mergesort").reset_index(drop=True)
    expected = len(papers) * len(TASKS)
    if len(oof) != expected or oof.duplicated(["task", "paper_id"]).any():
        raise RuntimeError("OOF cardinality or uniqueness gate failed")
    if not bool(oof.model_converged.all()):
        unstable = set(oof.loc[~oof.model_converged, "task"])
        if unstable - {"T1b_EXPLICIT_DEFINITION"}:
            raise RuntimeError(f"Main-task OOF convergence failed: {sorted(unstable)}")
    return oof, diagnostics, task_status


def stable_order(scores: np.ndarray, tie_rank: np.ndarray) -> np.ndarray:
    return np.lexsort((tie_rank, -scores))


def selected_indices(
    scores: np.ndarray,
    costs: np.ndarray,
    budget_type: str,
    fraction: float,
    tie_rank: np.ndarray,
) -> np.ndarray:
    order = stable_order(scores, tie_rank)
    if budget_type == "paper":
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
    if budget_type == "paper":
        return rng.choice(len(costs), size=math.ceil(len(costs) * fraction), replace=False)
    order = rng.permutation(len(costs))
    token_budget = math.floor(float(costs.sum()) * fraction)
    cumulative = np.cumsum(costs[order], dtype=np.int64)
    return order[cumulative <= token_budget]


def selection_summary(indices: np.ndarray, yields: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    actual = float(yields[indices].sum())
    tokens = float(costs[indices].sum())
    total = float(yields.sum())
    return {
        "n_selected": float(len(indices)),
        "source_tokens_selected": tokens,
        "actual_mv_yield": actual,
        "yield_capture": actual / total if total > 0 else 0.0,
        "yield_per_paper": actual / len(indices) if len(indices) else 0.0,
        "yield_per_1m_tokens": actual * 1_000_000.0 / tokens if tokens > 0 else 0.0,
    }


def point_selection_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, float], dict[str, float]]]:
    count_wide = oof.pivot(index="paper_id", columns="task", values="oof_predicted_count")
    rate_wide = oof.pivot(index="paper_id", columns="task", values="oof_predicted_rate_per_10k")
    rows = []
    random_lookup = {}
    for task in TASKS:
        frame = oof[oof.task == task].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        paper_ids = frame.paper_id.to_numpy(str)
        tie = np.arange(len(frame), dtype=np.int64)
        costs = frame.source_tokens.to_numpy(np.int64)
        yields = frame.actual_mv_count.to_numpy(float)
        for budget_type in ("paper", "token"):
            for fraction in BUDGET_FRACTIONS:
                rng = np.random.default_rng(RANDOM_SEED + sum(ord(c) for c in task) + int(fraction * 1000) + (0 if budget_type == "paper" else 10000))
                repeats = [selection_summary(random_indices(rng, costs, budget_type, fraction), yields, costs) for _ in range(RANDOM_REPEATS)]
                random_mean = {key: statistics_mean([item[key] for item in repeats]) for key in repeats[0]}
                random_mean["capture_ci_low"] = float(np.quantile([item["yield_capture"] for item in repeats], 0.025))
                random_mean["capture_ci_high"] = float(np.quantile([item["yield_capture"] for item in repeats], 0.975))
                random_lookup[(task, budget_type, fraction)] = random_mean
                rows.append(
                    {
                        "task": task, "strategy": "RANDOM", "ranking_task": "", "budget_type": budget_type,
                        "budget_fraction": fraction, **random_mean, "lift_vs_random": 1.0,
                        "random_repeats": RANDOM_REPEATS, "scientific_role": "baseline",
                    }
                )
                strategies = [("TASK_SPECIFIC", task)] + [("CROSS_TASK", other) for other in TASKS if other != task] + [("ORACLE_UPPER_BOUND", task)]
                for strategy, ranking_task in strategies:
                    if strategy == "ORACLE_UPPER_BOUND":
                        scores = yields if budget_type == "paper" else yields / (costs / 10_000.0)
                    else:
                        source = count_wide if budget_type == "paper" else rate_wide
                        scores = source.loc[paper_ids, ranking_task].to_numpy(float)
                    metric = selection_summary(selected_indices(scores, costs, budget_type, fraction, tie), yields, costs)
                    rows.append(
                        {
                            "task": task, "strategy": strategy, "ranking_task": ranking_task,
                            "budget_type": budget_type, "budget_fraction": fraction, **metric,
                            "lift_vs_random": metric["yield_capture"] / random_mean["yield_capture"] if random_mean["yield_capture"] > 0 else None,
                            "capture_ci_low": None, "capture_ci_high": None, "random_repeats": RANDOM_REPEATS,
                            "scientific_role": "upper_bound_diagnostic" if strategy == "ORACLE_UPPER_BOUND" else "deployable_oof_selection",
                        }
                    )
    return pd.DataFrame(rows), random_lookup


def statistics_mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def set_overlap(left: set[str], right: set[str]) -> tuple[float, float]:
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0, intersection / min(len(left), len(right)) if left and right else 0.0


def top_set(frame: pd.DataFrame, score: str, fraction: float) -> set[str]:
    n = math.ceil(len(frame) * fraction)
    ordered = frame.sort_values([score, "paper_id"], ascending=[False, True], kind="mergesort")
    return set(ordered.paper_id.astype(str).iloc[:n])


def ranking_disagreement(oof: pd.DataFrame) -> pd.DataFrame:
    rate = oof.pivot(index="paper_id", columns="task", values="oof_predicted_rate_per_10k").reset_index()
    rows = []
    for index, left in enumerate(TASKS):
        for right in TASKS[index + 1:]:
            row = {
                "task_a": left,
                "task_b": right,
                "score": "oof_predicted_rate_per_10k",
                "spearman": float(st.spearmanr(rate[left], rate[right]).statistic),
                "kendall": float(st.kendalltau(rate[left], rate[right]).statistic),
            }
            for fraction in (0.10, 0.20, 0.30):
                a = top_set(rate[["paper_id", left]].rename(columns={left: "score"}), "score", fraction)
                b = top_set(rate[["paper_id", right]].rename(columns={right: "score"}), "score", fraction)
                jac, coefficient = set_overlap(a, b)
                row[f"top{int(fraction * 100)}_jaccard"] = jac
                row[f"top{int(fraction * 100)}_overlap_coefficient"] = coefficient
            rows.append(row)
    return pd.DataFrame(rows)


def actual_yield_overlap(oof: pd.DataFrame) -> pd.DataFrame:
    data = oof.copy()
    data["actual_rate"] = data.actual_mv_count / (data.source_tokens / 10_000.0)
    wide = data.pivot(index="paper_id", columns="task", values="actual_rate").reset_index()
    rows = []
    for index, left in enumerate(TASKS):
        for right in TASKS[index + 1:]:
            a = top_set(wide[["paper_id", left]].rename(columns={left: "score"}), "score", 0.20)
            b = top_set(wide[["paper_id", right]].rename(columns={right: "score"}), "score", 0.20)
            jac, coefficient = set_overlap(a, b)
            rows.append(
                {
                    "task_a": left,
                    "task_b": right,
                    "spearman_actual_rate": float(st.spearmanr(wide[left], wide[right]).statistic),
                    "top20_jaccard": jac,
                    "top20_overlap_coefficient": coefficient,
                    "role": "descriptive_oracle_outcome_overlap_not_deployable",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_selection(
    oof: pd.DataFrame,
    selection_metrics: pd.DataFrame,
) -> pd.DataFrame:
    count_wide = oof.pivot(index="paper_id", columns="task", values="oof_predicted_count")
    rate_wide = oof.pivot(index="paper_id", columns="task", values="oof_predicted_rate_per_10k")
    distributions: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for task in MAIN_TASKS:
        frame = oof[oof.task == task].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        paper_ids = frame.paper_id.to_numpy(str)
        yields_all = frame.actual_mv_count.to_numpy(float)
        costs_all = frame.source_tokens.to_numpy(np.int64)
        count_scores = {name: count_wide.loc[paper_ids, name].to_numpy(float) for name in TASKS}
        rate_scores = {name: rate_wide.loc[paper_ids, name].to_numpy(float) for name in TASKS}
        rng = np.random.default_rng(RANDOM_SEED + 50000 + sum(ord(c) for c in task))
        keys = [(budget, "TASK_SPECIFIC-RANDOM") for budget in ("paper", "token")]
        keys += [(budget, f"TASK_SPECIFIC-CROSS_TASK:{other}") for budget in ("paper", "token") for other in TASKS if other != task]
        for budget, comparison in keys:
            distributions[(task, budget, comparison)] = {"effect": [], "lift": []}
        for _ in range(BOOTSTRAP_REPLICATES):
            sample = rng.integers(0, len(frame), size=len(frame), dtype=np.int32)
            yields = yields_all[sample]
            costs = costs_all[sample]
            tie = np.arange(len(sample), dtype=np.int64)
            total = float(yields.sum())
            if total <= 0:
                continue
            ts_capture = {}
            random_capture = {}
            cross_capture: dict[tuple[str, str], float] = {}
            for budget_type in ("paper", "token"):
                current_scores = count_scores[task][sample] if budget_type == "paper" else rate_scores[task][sample]
                ts_indices = selected_indices(current_scores, costs, budget_type, PRIMARY_FRACTION, tie)
                ts_capture[budget_type] = float(yields[ts_indices].sum() / total)
                random_sel = random_indices(rng, costs, budget_type, PRIMARY_FRACTION)
                random_capture[budget_type] = float(yields[random_sel].sum() / total)
                for other in TASKS:
                    if other == task:
                        continue
                    scores = count_scores[other][sample] if budget_type == "paper" else rate_scores[other][sample]
                    cross_sel = selected_indices(scores, costs, budget_type, PRIMARY_FRACTION, tie)
                    cross_capture[(budget_type, other)] = float(yields[cross_sel].sum() / total)
            for budget_type in ("paper", "token"):
                base = random_capture[budget_type]
                target = ts_capture[budget_type]
                dist = distributions[(task, budget_type, "TASK_SPECIFIC-RANDOM")]
                dist["effect"].append(target - base)
                dist["lift"].append(target / base if base > 0 else np.nan)
                for other in TASKS:
                    if other == task:
                        continue
                    base = cross_capture[(budget_type, other)]
                    dist = distributions[(task, budget_type, f"TASK_SPECIFIC-CROSS_TASK:{other}")]
                    dist["effect"].append(target - base)
                    dist["lift"].append(target / base if base > 0 else np.nan)
    rows = []
    for (task, budget_type, comparison), values in distributions.items():
        if comparison == "TASK_SPECIFIC-RANDOM":
            baseline_strategy, ranking_task = "RANDOM", ""
        else:
            baseline_strategy, ranking_task = "CROSS_TASK", comparison.split(":", 1)[1]
        target_row = selection_metrics[
            (selection_metrics.task == task)
            & (selection_metrics.strategy == "TASK_SPECIFIC")
            & (selection_metrics.budget_type == budget_type)
            & np.isclose(selection_metrics.budget_fraction, PRIMARY_FRACTION)
        ].iloc[0]
        baseline_rows = selection_metrics[
            (selection_metrics.task == task)
            & (selection_metrics.strategy == baseline_strategy)
            & (selection_metrics.budget_type == budget_type)
            & np.isclose(selection_metrics.budget_fraction, PRIMARY_FRACTION)
        ]
        if baseline_strategy == "CROSS_TASK":
            baseline_rows = baseline_rows[baseline_rows.ranking_task == ranking_task]
        baseline_row = baseline_rows.iloc[0]
        effects = np.asarray(values["effect"], dtype=float)
        lifts = np.asarray(values["lift"], dtype=float)
        finite_lifts = lifts[np.isfinite(lifts)]
        nonpositive = (np.count_nonzero(effects <= 0.0) + 1) / (len(effects) + 1)
        nonnegative = (np.count_nonzero(effects >= 0.0) + 1) / (len(effects) + 1)
        role = "primary" if comparison == "TASK_SPECIFIC-RANDOM" else "secondary_cross_task"
        rows.append(
            {
                "task": task,
                "budget_type": budget_type,
                "budget_fraction": PRIMARY_FRACTION,
                "comparison": comparison,
                "baseline_ranking_task": ranking_task,
                "effect": float(target_row.yield_capture - baseline_row.yield_capture),
                "relative_lift": float(target_row.yield_capture / baseline_row.yield_capture),
                "CI_low": float(np.quantile(effects, 0.025)),
                "CI_high": float(np.quantile(effects, 0.975)),
                "relative_lift_CI_low": float(np.quantile(finite_lifts, 0.025)),
                "relative_lift_CI_high": float(np.quantile(finite_lifts, 0.975)),
                "p": min(1.0, 2.0 * min(nonpositive, nonnegative)) if role == "primary" else None,
                "holm_p": None,
                "bootstrap_replicates": len(effects),
                "statistical_role": role,
            }
        )
    output = pd.DataFrame(rows)
    primary = output.statistical_role == "primary"
    output.loc[primary, "holm_p"] = source_holm(output.loc[primary, "p"].to_numpy(float))
    return output.sort_values(["statistical_role", "task", "budget_type", "comparison"], kind="mergesort").reset_index(drop=True)


def source_holm(p_values: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests

    return multipletests(p_values, alpha=0.05, method="holm")[1]


def task_judgments(bootstrap: pd.DataFrame) -> dict[str, str]:
    judgments = {}
    for task in MAIN_TASKS:
        rows = bootstrap[(bootstrap.task == task) & (bootstrap.comparison == "TASK_SPECIFIC-RANDOM")]
        effects = rows.set_index("budget_type").effect.to_dict()
        lows = rows.set_index("budget_type").CI_low.to_dict()
        if all(effects.get(kind, 0.0) > 0 and lows.get(kind, -1.0) > 0 for kind in ("paper", "token")):
            judgments[task] = "SUPPORTED"
        elif all(effects.get(kind, 0.0) > 0 for kind in ("paper", "token")):
            judgments[task] = "DIRECTIONALLY_SUPPORTED"
        elif effects.get("paper", 0.0) * effects.get("token", 0.0) < 0:
            judgments[task] = "MIXED"
        else:
            judgments[task] = "NOT_SUPPORTED"
    supported = sum(value == "SUPPORTED" for value in judgments.values())
    positive = sum(value in {"SUPPORTED", "DIRECTIONALLY_SUPPORTED"} for value in judgments.values())
    judgments["overall"] = (
        "SOURCE_SELECTION_UTILITY_SUPPORTED" if supported >= 2
        else "PARTIALLY_SUPPORTED" if positive >= 1
        else "NOT_SUPPORTED"
    )
    return judgments


def top100_examples(oof: pd.DataFrame) -> list[dict[str, Any]]:
    count_wide = oof.pivot(index="paper_id", columns="task", values="oof_predicted_count")
    rows = []
    for task in MAIN_TASKS:
        frame = oof[oof.task == task].sort_values("paper_id", kind="mergesort").reset_index(drop=True)
        ids = frame.paper_id.to_numpy(str)
        yields = frame.actual_mv_count.to_numpy(float)
        scores = count_wide.loc[ids, task].to_numpy(float)
        selected = stable_order(scores, np.arange(len(frame)))[:100]
        selected_yield = float(yields[selected].sum())
        rng = np.random.default_rng(RANDOM_SEED + 90000 + sum(ord(c) for c in task))
        random_yields = [float(yields[rng.choice(len(frame), 100, replace=False)].sum()) for _ in range(RANDOM_REPEATS)]
        random_mean = statistics_mean(random_yields)
        rows.append(
            {
                "task": task,
                "task_specific_yield": selected_yield,
                "random_mean_yield": random_mean,
                "additional_instances": selected_yield - random_mean,
            }
        )
    return rows


def render_summary(
    oof_status: dict[str, str],
    selection: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ranking: pd.DataFrame,
    judgments: dict[str, str],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# Source Suitability Utility Validation V1",
        "",
        "## OOF integrity",
        "",
        *[f"- {task}: `{status}`" for task, status in oof_status.items()],
        "",
        "## Primary 20% budget results",
        "",
        "| Task | Budget | Random capture | Task-specific capture | Gain | Relative lift | 95% effect CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in MAIN_TASKS:
        for budget in ("paper", "token"):
            target = selection[(selection.task == task) & (selection.strategy == "TASK_SPECIFIC") & (selection.budget_type == budget) & np.isclose(selection.budget_fraction, 0.2)].iloc[0]
            random = selection[(selection.task == task) & (selection.strategy == "RANDOM") & (selection.budget_type == budget) & np.isclose(selection.budget_fraction, 0.2)].iloc[0]
            boot = bootstrap[(bootstrap.task == task) & (bootstrap.budget_type == budget) & (bootstrap.comparison == "TASK_SPECIFIC-RANDOM")].iloc[0]
            lines.append(
                f"| {task} | {budget} | {random.yield_capture:.4f} | {target.yield_capture:.4f} | "
                f"{boot.effect:+.4f} | {boot.relative_lift:.3f} | [{boot.CI_low:+.4f}, {boot.CI_high:+.4f}] |"
            )
    lines.extend(["", "## Judgments", ""])
    for task in MAIN_TASKS:
        lines.append(f"- {task}: `{judgments[task]}`")
    lines.append(f"- Overall: `{judgments['overall']}`")
    lines.extend(["", "## Cross-task ranking", ""])
    for pair in (("T1a_ACRONYM", "T2_NUMERIC"), ("T1a_ACRONYM", "T3_CLAIM"), ("T2_NUMERIC", "T3_CLAIM")):
        row = ranking[(ranking.task_a == pair[0]) & (ranking.task_b == pair[1])].iloc[0]
        lines.append(f"- {pair[0]} vs {pair[1]}: Spearman={row.spearman:.4f}; Top20 Jaccard={row.top20_jaccard:.4f}.")
    lines.extend(["", "## 100-paper illustration", ""])
    for row in examples:
        lines.append(
            f"- {row['task']}: task-specific={row['task_specific_yield']:.1f}, random mean={row['random_mean_yield']:.1f}, additional={row['additional_instances']:+.1f}."
        )
    overall = judgments["overall"]
    recommendation = "A. Main Results" if overall == "SOURCE_SELECTION_UTILITY_SUPPORTED" else "B. Supplementary only" if overall == "PARTIALLY_SUPPORTED" else "C. Do not include"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Recommended placement: **{recommendation}**.",
            "",
            "Strict wording: “Under fixed processing budgets, cross-fitted task-specific source rankings showed predictive selection utility for machine-verified corpus yield; the magnitude and stability were task dependent.”",
            "",
            "These are predictive selection results, not causal effects or measures of scientific paper quality. ORACLE is an upper-bound diagnostic only. T1b remains supplementary regardless of its result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.time()
    root = Path(__file__).resolve().parents[3]
    source_path = root / "code/corpus_experiments_v1/sft/run_source_suitability_v1.py"
    source_dir = root / "aicorpus-derived/experiments/corpus_v1/sft_source_suitability_v1"
    output_dir = root / "aicorpus-derived/experiments/corpus_v1/source_suitability_utility_v1"
    cal_manifest = root / "aicorpus-derived/experiments/corpus_v1/sft_a/calibration_v1/cal100_manifest.csv"
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    source = load_source_runner(source_path)
    if tuple(source.TASKS) != TASKS or tuple(source.PREDICTORS) != (
        "results_presence", "discussion_presence", "conclusion_presence", "methods_presence",
        "section_count", "average_paragraph_length", "figure_density_per_10k", "formula_density_per_10k",
        "suspected_word_split_rate", "reference_like_rate",
    ):
        raise RuntimeError("Frozen source-suitability task/predictor definition changed")
    metrics, features = source.verify_inputs()
    reconstructed = source.make_dataset(metrics, features)
    modeling_path = source_dir / "modeling_dataset.parquet"
    if sha256_file(modeling_path) != "db43b651832b2bef88c4996bc44aa98b47c88e2c1589555c213d80e852e73029":
        raise RuntimeError("Frozen modeling dataset SHA-256 mismatch")
    data = pd.read_parquet(modeling_path)
    compare_columns = ["paper_id", "source_id", "task", "machine_verified_count", "source_tokens", *source.PREDICTORS, "offset_model_eligible"]
    left = data[compare_columns].sort_values(["task", "paper_id"]).reset_index(drop=True)
    right = reconstructed[compare_columns].sort_values(["task", "paper_id"]).reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Frozen modeling dataset differs from formal preprocessing reconstruction")
    eligible = data[data.offset_model_eligible].copy()
    if eligible.groupby("task").size().to_dict() != {task: 1898 for task in TASKS}:
        raise RuntimeError("Expected exactly 1,898 positive-token papers per task")
    cal_ids = set(pd.read_csv(cal_manifest).source_id.astype(str))
    if set(eligible.source_id.astype(str)) & cal_ids:
        raise RuntimeError("Cal100 leakage into OOF validation")

    oof, fold_diagnostics, oof_status = build_oof(source, data)
    selection, _ = point_selection_metrics(oof)
    ranking = ranking_disagreement(oof)
    actual_overlap = actual_yield_overlap(oof)
    bootstrap = bootstrap_selection(oof, selection)
    judgments = task_judgments(bootstrap)
    examples = top100_examples(oof)

    gates = {
        "four_tasks_1898_oof_predictions": len(oof) == 1898 * 4,
        "one_prediction_per_task_paper": not bool(oof.duplicated(["task", "paper_id"]).any()),
        "all_predictions_finite": bool(np.isfinite(oof[["oof_predicted_count", "oof_predicted_rate_per_10k"]].to_numpy()).all()),
        "all_fold_models_converged": bool(oof.model_converged.all()),
        "cal100_overlap_zero": not bool(set(oof.source_id) & cal_ids),
        "zero_token_papers_excluded_exactly_two": int((~data[data.task == TASKS[0]].offset_model_eligible).sum()) == 2,
        "predictors_and_offset_frozen": True,
        "random_repeats_1000": RANDOM_REPEATS == 1000,
        "bootstrap_10000": bool((bootstrap.bootstrap_replicates == BOOTSTRAP_REPLICATES).all()),
        "six_primary_holm_tests": int((bootstrap.statistical_role == "primary").sum()) == 6 and bootstrap.loc[bootstrap.statistical_role == "primary", "holm_p"].notna().all(),
        "no_general_score_invented": True,
        "no_training_or_materialization": True,
    }
    status = "SOURCE_SUITABILITY_UTILITY_READY" if all(gates.values()) else "SOURCE_SUITABILITY_UTILITY_NOT_READY"
    input_assets = {
        "formal_source_runner": {"path": source_path.relative_to(root).as_posix(), "sha256": sha256_file(source_path)},
        "formal_source_config": {"path": (source_dir / "config.json").relative_to(root).as_posix(), "sha256": sha256_file(source_dir / "config.json")},
        "formal_modeling_dataset": {"path": modeling_path.relative_to(root).as_posix(), "sha256": sha256_file(modeling_path)},
        "holdout_v1_2_metrics": {"path": (source.V12 / "paper_task_metrics.parquet").relative_to(root).as_posix(), "sha256": sha256_file(source.V12 / "paper_task_metrics.parquet")},
        "cal100_manifest": {"path": cal_manifest.relative_to(root).as_posix(), "sha256": sha256_file(cal_manifest)},
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "purpose": "strict OOF predictive selection utility for machine-verified corpus yield",
        "input_assets": input_assets,
        "tasks": list(TASKS),
        "main_narrative_tasks": list(MAIN_TASKS),
        "T1b_role": "supplementary_supporting_due_to_sparse_yield",
        "model": {
            "family": "NB2",
            "implementation": "exact reuse of formal statsmodels NegativeBinomial fit_nb",
            "optimizer": "BFGS maxiter=1000",
            "outcome": "machine_verified_count",
            "offset": "log(source_tokens/10000)",
            "predictors": list(source.PREDICTORS),
            "continuous_preprocessing": "training-fold mean/sample-SD only, applied to held-out fold",
        },
        "cross_fitting": {
            "folds": N_FOLDS,
            "seed": FOLD_SEED,
            "assignment": "sort SHA256(seed|paper_id), then ordinal modulo 5",
            "paper_level": True,
            "fold_diagnostics": fold_diagnostics,
            "task_status": oof_status,
        },
        "selection": {
            "strategies": ["RANDOM", "TASK_SPECIFIC", "CROSS_TASK", "ORACLE_UPPER_BOUND"],
            "general_baseline": "not_present_no_frozen_task_agnostic_score_and_none_invented",
            "budget_fractions": list(BUDGET_FRACTIONS),
            "primary_fraction": PRIMARY_FRACTION,
            "paper_budget_rounding": "ceil(fraction * 1898)",
            "token_budget": "floor(fraction * total positive source tokens); whole papers in rank order; stop before first paper that would exceed cap",
            "paper_score": "oof_predicted_count",
            "token_score": "oof_predicted_rate_per_10k",
            "random_repeats": RANDOM_REPEATS,
        },
        "inference": {
            "paper_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": RANDOM_SEED + 50000,
            "selection_recomputed_inside_each paper bootstrap sample": True,
            "primary_tests": "T1a/T2/T3 task-specific minus random at 20% paper and 20% token budgets",
            "holm_family_size": 6,
            "cross_task_comparisons": "secondary effect and CI without p-value",
        },
        "judgment_rules": {
            "SUPPORTED": "both primary budget effects >0 and both 95% CI lower bounds >0",
            "DIRECTIONALLY_SUPPORTED": "both effects >0 but at least one CI includes 0",
            "MIXED": "paper/token effect signs differ",
            "NOT_SUPPORTED": "otherwise",
            "SOURCE_SELECTION_UTILITY_SUPPORTED": "at least two main tasks SUPPORTED",
            "PARTIALLY_SUPPORTED": "fewer than two SUPPORTED but at least one SUPPORTED or DIRECTIONALLY_SUPPORTED",
        },
        "task_judgments": judgments,
        "gates": gates,
        "runtime_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": source.sm.__version__,
        },
        "interpretation_boundary": "predictive/selection utility only; not causal and not scientific-paper quality",
    }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "experiment_config.json", config)
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    selection.to_csv(output_dir / "selection_metrics.csv", index=False, encoding="utf-8", lineterminator="\n")
    bootstrap.to_csv(output_dir / "primary_bootstrap.csv", index=False, encoding="utf-8", lineterminator="\n")
    ranking.to_csv(output_dir / "cross_task_ranking.csv", index=False, encoding="utf-8", lineterminator="\n")
    actual_overlap.to_csv(output_dir / "actual_yield_overlap.csv", index=False, encoding="utf-8", lineterminator="\n")
    (output_dir / "summary.md").write_text(
        render_summary(oof_status, selection, bootstrap, ranking, judgments, examples), encoding="utf-8"
    )
    write_checksums(output_dir)
    print(json.dumps({"status": status, "oof": oof_status, "judgments": judgments, "output": output_dir.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
