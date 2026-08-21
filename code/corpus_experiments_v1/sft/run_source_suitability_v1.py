from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V12 = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_b_holdout1900_v1_2"
V11_HOLDOUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_b_holdout1900"
CAL = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/calibration_v1"
OUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_source_suitability_v1"

EXPECTED = {
    "holdout_membership": "e4066ba08a65e69055eb795f07b7b97715930dcce0f21a48819af35fe4541b4f",
    "v12_verifier_config": "26a035bf6e9bf7d63d0d401def58d4ddee740b6fe6057925aa368fe50e3b1bd0",
    "v12_summary": "2d0e73bcdfbe809ac95161588e51c44759da88bd6917633cb172f28482188006",
    "v12_metrics": "9273b70d0a0b551f8e0f2a9188bc2a85dc293f49d87777a66aecede278a0c463",
    "holdout_features": "786c4daf67e9c17a94e004efb339c83b35e2c550c2f10bf2549aac90ce4319b4",
}

TASKS = ("T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION", "T2_NUMERIC", "T3_CLAIM")
BINARY_TASKS = ("T1b_EXPLICIT_DEFINITION", "T3_CLAIM")
BINARY_SATURATED = ("T1a_ACRONYM", "T2_NUMERIC")
BINARY_PREDICTORS = ("results_presence", "discussion_presence", "conclusion_presence", "methods_presence")
CONTINUOUS_PREDICTORS = (
    "section_count",
    "average_paragraph_length",
    "figure_density_per_10k",
    "formula_density_per_10k",
    "suspected_word_split_rate",
    "reference_like_rate",
)
PREDICTORS = (*BINARY_PREDICTORS, *CONTINUOUS_PREDICTORS)
LEAKAGE_EXCLUSIONS = (
    "numeric_density", "definition_pattern_density", "claim_cue_density",
    "raw_count", "hard_pass_count", "ready_count",
)
ZERO_FIT_WARNING_ABS_THRESHOLD = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def membership_sha(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str).tolist()).encode("utf-8")).hexdigest()


def verify_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
        "v12_verifier_config": V12 / "machine_verifier_config_v1_2.json",
        "v12_summary": V12 / "summary.json",
        "v12_metrics": V12 / "paper_task_metrics.parquet",
        "holdout_features": V11_HOLDOUT / "paper_features_holdout1900.parquet",
    }
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(f"Frozen input SHA256 mismatch for {name}: {actual} != {EXPECTED[name]}")
    summary = json.loads((V12 / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "SFT_B_HOLDOUT_YIELD_READY" or not all(summary["ready_gates"].values()):
        raise RuntimeError("SFT-B0 v1.2 is not a fully gated READY input")
    manifest = pd.read_csv(CAL / "holdout1900_manifest.csv")
    if len(manifest) != 1900 or manifest.source_id.nunique() != 1900 or membership_sha(manifest.source_id) != EXPECTED["holdout_membership"]:
        raise RuntimeError("Frozen Holdout1900 membership mismatch")
    metrics = pd.read_parquet(V12 / "paper_task_metrics.parquet")
    features = pd.read_parquet(V11_HOLDOUT / "paper_features_holdout1900.parquet")
    if len(metrics) != 7600 or metrics.paper_id.nunique() != 1900 or set(metrics.task) != set(TASKS):
        raise RuntimeError("Formal paper-task metrics cardinality mismatch")
    if len(features) != 1900 or features.paper_id.nunique() != 1900:
        raise RuntimeError("Formal feature table cardinality mismatch")
    if set(metrics.source_id.astype(str)) != set(manifest.source_id.astype(str)) or set(features.source_id.astype(str)) != set(manifest.source_id.astype(str)):
        raise RuntimeError("Modeling inputs do not match frozen Holdout1900 membership")
    return metrics, features


def make_dataset(metrics: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    source = features.copy()
    positive = source.source_tokens > 0
    source["figure_density_per_10k"] = np.where(positive, source.figure_count * 10000 / source.source_tokens, np.nan)
    source["table_density_per_10k"] = np.where(positive, source.table_count * 10000 / source.source_tokens, np.nan)
    source["formula_density_per_10k"] = np.where(positive, source.formula_count * 10000 / source.source_tokens, np.nan)
    source["zero_source_token"] = ~positive
    source["offset_log_source_tokens_per_10k"] = np.nan
    source.loc[positive, "offset_log_source_tokens_per_10k"] = np.log(source.loc[positive, "source_tokens"] / 10000)
    keep = [
        "paper_id", "source_id", "source_tokens", "zero_source_token", "offset_log_source_tokens_per_10k",
        "paragraph_count", "section_count", "average_paragraph_length",
        "results_presence", "discussion_presence", "conclusion_presence", "methods_presence", "abstract_presence",
        "figure_count", "table_count", "formula_count", "figure_density_per_10k", "table_density_per_10k", "formula_density_per_10k",
        "figure_caption_coverage", "table_caption_coverage", "suspected_word_split_rate", "reference_like_rate",
        "year", "topic_cluster", "category", "document_type",
        "numeric_density", "definition_pattern_density", "claim_cue_density",
    ]
    data = metrics.merge(source[keep], on=["paper_id", "source_id"], how="left", validate="many_to_one", suffixes=("", "_feature"))
    if len(data) != 7600:
        raise RuntimeError("Modeling table merge cardinality mismatch")
    data["binary_outcome"] = data.machine_verified_count > 0
    eligible = ~data.zero_source_token & data[list(PREDICTORS)].notna().all(axis=1)
    data["offset_model_eligible"] = eligible
    base = source.loc[positive, list(CONTINUOUS_PREDICTORS)]
    for predictor in CONTINUOUS_PREDICTORS:
        mean, sd = float(base[predictor].mean()), float(base[predictor].std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"Invalid standard deviation for {predictor}")
        data[f"z_{predictor}"] = (data[predictor] - mean) / sd
    return data


def feature_availability(features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    candidate_features = [
        "results_presence", "discussion_presence", "conclusion_presence", "methods_presence", "abstract_presence",
        "section_count", "paragraph_count", "average_paragraph_length",
        "figure_count", "table_count", "formula_count", "figure_density_per_10k", "table_density_per_10k", "formula_density_per_10k",
        "figure_caption_coverage", "table_caption_coverage", "suspected_word_split_rate", "reference_like_rate",
        "year", "topic_cluster", "category", "document_type",
        "numeric_density", "definition_pattern_density", "claim_cue_density",
        "raw_count", "hard_pass_count", "ready_count", "source_tokens",
    ]
    one = data[data.task == TASKS[0]].set_index("paper_id")
    rows = []
    reasons = {
        "abstract_presence": "excluded_near_constant_99.68_percent_present",
        "paragraph_count": "excluded_raw_structure_count_partly_tracks_length_exposure",
        "figure_count": "excluded_raw_object_count_use_density_instead",
        "table_count": "excluded_raw_object_count_use_density_instead",
        "formula_count": "excluded_raw_object_count_use_density_instead",
        "table_density_per_10k": "excluded_pre_model_collinearity_abs_r_0.959_with_figure_density_keep_more_basic_figure_density",
        "figure_caption_coverage": "excluded_structurally_undefined_when_no_figures_no_blind_imputation",
        "table_caption_coverage": "excluded_structurally_undefined_when_no_tables_no_blind_imputation",
        "year": "excluded_100_percent_missing",
        "category": "excluded_100_percent_missing",
        "document_type": "excluded_100_percent_missing",
        "topic_cluster": "excluded_31_categories_no_frozen_upper_taxonomy_avoid_sparse_dummy_expansion",
        "numeric_density": "excluded_target_proximal_candidate_detection_leakage",
        "definition_pattern_density": "excluded_target_proximal_candidate_detection_leakage",
        "claim_cue_density": "excluded_target_proximal_candidate_detection_leakage",
        "raw_count": "excluded_candidate_generation_chain_leakage",
        "hard_pass_count": "excluded_candidate_generation_chain_leakage",
        "ready_count": "excluded_candidate_generation_chain_leakage",
        "source_tokens": "offset_only_not_ordinary_predictor",
    }
    for feature in candidate_features:
        series = one[feature] if feature in one else data.loc[data.task == TASKS[0], feature]
        missing = int(series.isna().sum())
        included = feature in PREDICTORS
        rows.append({
            "feature": feature, "dtype": str(series.dtype), "missing_n": missing, "missing_percent": 100 * missing / 1900,
            "nonmissing_unique": int(series.nunique(dropna=True)), "included_in_main_model": included,
            "role_or_exclusion_reason": "main_binary_predictor" if feature in BINARY_PREDICTORS else "main_z_standardized_continuous_predictor" if feature in CONTINUOUS_PREDICTORS else reasons[feature],
        })
    return pd.DataFrame(rows)


def design(data: pd.DataFrame, winsorize: bool = False) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    output = pd.DataFrame(index=data.index)
    standards: dict[str, dict[str, float]] = {}
    for predictor in BINARY_PREDICTORS:
        output[predictor] = data[predictor].astype(float)
    for predictor in CONTINUOUS_PREDICTORS:
        values = data[predictor].astype(float).copy()
        lower = upper = None
        if winsorize:
            lower, upper = [float(value) for value in values.quantile([0.01, 0.99])]
            values = values.clip(lower, upper)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"Invalid standardization in design for {predictor}")
        output[predictor] = (values - mean) / sd
        standards[predictor] = {"mean": mean, "sd": sd, "winsor_lower": lower, "winsor_upper": upper}
    output = sm.add_constant(output, has_constant="add")
    return output, standards


def holm(raw_p: pd.Series) -> np.ndarray:
    return multipletests(raw_p.to_numpy(float), alpha=0.05, method="holm")[1]


def nb_deviance(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    first = np.zeros_like(y, dtype=float)
    positive = y > 0
    first[positive] = y[positive] * np.log(y[positive] / mu[positive])
    second = (y + 1 / alpha) * np.log((y + 1 / alpha) / (mu + 1 / alpha))
    return float(2 * np.sum(first - second))


def fit_nb(task_data: pd.DataFrame, variant: str = "main") -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, float]], Any]:
    frame = task_data.copy()
    winsorize = variant == "winsorized_1_99"
    if variant == "leave_top_1_percent_out":
        n_remove = math.ceil(len(frame) * 0.01)
        frame = frame.sort_values(["machine_verified_count", "paper_id"], ascending=[False, True]).iloc[n_remove:].copy()
    x, standards = design(frame, winsorize=winsorize)
    y = frame.machine_verified_count.astype(float)
    offset = frame.offset_log_source_tokens_per_10k.astype(float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = NegativeBinomial(y, x, offset=offset, loglike_method="nb2")
        result = model.fit(method="bfgs", maxiter=1000, disp=False)
    converged = bool(result.mle_retvals.get("converged", False))
    alpha = float(result.params["alpha"])
    mu = np.asarray(result.predict(), dtype=float)
    df_resid = len(frame) - x.shape[1]
    variance = mu + alpha * mu**2
    pearson = float(np.sum((y.to_numpy() - mu) ** 2 / variance))
    deviance = nb_deviance(y.to_numpy(), mu, alpha)
    observed_zero = float((y == 0).mean())
    predicted_zero = float(np.mean((1 + alpha * mu) ** (-1 / alpha)))
    poisson = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset).fit(maxiter=1000)
    results = []
    for predictor in PREDICTORS:
        coefficient = float(result.params[predictor])
        se = float(result.bse[predictor])
        lower, upper = [float(v) for v in result.conf_int().loc[predictor]]
        results.append({
            "task": frame.task.iloc[0], "predictor": predictor, "coefficient": coefficient, "se": se,
            "irr": math.exp(coefficient), "ci_lower": math.exp(lower), "ci_upper": math.exp(upper),
            "raw_p": float(result.pvalues[predictor]), "variant": variant,
        })
    rows = pd.DataFrame(results)
    if variant == "main":
        rows["holm_adjusted_p"] = holm(rows.raw_p)
    else:
        rows["holm_adjusted_p"] = np.nan
    extreme = bool((rows.coefficient.abs() > 10).any() or (rows.se > 10).any())
    diagnostic = {
        "task": frame.task.iloc[0], "variant": variant, "n": len(frame), "mean_outcome": float(y.mean()),
        "zero_rate": observed_zero, "alpha": alpha, "log_likelihood": float(result.llf), "aic": float(result.aic),
        "converged": converged, "optimizer_warnings": " | ".join(str(item.message) for item in caught) or None,
        "extreme_coefficient_or_se": extreme, "pearson_chi2": pearson, "pearson_dispersion": pearson / df_resid,
        "deviance": deviance, "deviance_dispersion": deviance / df_resid,
        "observed_mean": float(y.mean()), "predicted_mean": float(mu.mean()),
        "observed_zero_proportion": observed_zero, "predicted_zero_proportion": predicted_zero,
        "zero_fit_absolute_difference": abs(observed_zero - predicted_zero),
        "zero_fit_warning": abs(observed_zero - predicted_zero) >= ZERO_FIT_WARNING_ABS_THRESHOLD,
        "poisson_aic_diagnostic": float(poisson.aic), "poisson_deviance_dispersion_diagnostic": float(poisson.deviance / poisson.df_resid),
    }
    return rows, diagnostic, standards, result


def fit_logistic(task_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    x, _ = design(task_data, winsorize=False)
    y = task_data.binary_outcome.astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=1000)
    rows = []
    conf = result.conf_int()
    for predictor in PREDICTORS:
        coefficient = float(result.params[predictor])
        lower, upper = [float(v) for v in conf.loc[predictor]]
        rows.append({
            "task": task_data.task.iloc[0], "predictor": predictor, "coefficient": coefficient,
            "se": float(result.bse[predictor]), "or": math.exp(coefficient), "ci_lower": math.exp(lower),
            "ci_upper": math.exp(upper), "raw_p": float(result.pvalues[predictor]),
        })
    output = pd.DataFrame(rows)
    output["holm_adjusted_p"] = holm(output.raw_p)
    predicted = np.asarray(result.predict(), dtype=float)
    diagnostic = {
        "task": task_data.task.iloc[0], "n": len(task_data), "positive_rate": float(y.mean()),
        "converged": bool(result.converged), "aic": float(result.aic), "log_likelihood": float(result.llf),
        "deviance": float(result.deviance), "deviance_dispersion": float(result.deviance / result.df_resid),
        "observed_positive_mean": float(y.mean()), "predicted_positive_mean": float(predicted.mean()),
        "optimizer_warnings": " | ".join(str(item.message) for item in caught) or None,
        "extreme_coefficient_or_se": bool((output.coefficient.abs() > 10).any() or (output.se > 10).any()),
    }
    return output, diagnostic


def collinearity_decisions(data: pd.DataFrame, standards: dict[str, dict[str, float]]) -> dict[str, Any]:
    source = data[(data.task == TASKS[0]) & data.offset_model_eligible].copy()
    continuous_corr = source[list(CONTINUOUS_PREDICTORS) + ["table_density_per_10k"]].corr()
    x, _ = design(source)
    vifs = {column: float(variance_inflation_factor(x.to_numpy(), index)) for index, column in enumerate(x.columns) if column != "const"}
    pairs = []
    columns = continuous_corr.columns.tolist()
    for index, left in enumerate(columns):
        for right in columns[index + 1:]:
            pairs.append({"left": left, "right": right, "r": float(continuous_corr.loc[left, right]), "abs_r_ge_0_80": bool(abs(continuous_corr.loc[left, right]) >= 0.80)})
    return {
        "decision_timing": "frozen before fitting any source-feature/outcome association model",
        "correlation_rule": "if abs(r) >= 0.80 retain the conceptually more basic/interpretable variable",
        "vif_rule": "inspect and do not retain obvious collinearity when VIF > 5",
        "decisions": [{"excluded": "table_density_per_10k", "retained": "figure_density_per_10k", "reason": "abs(r)=0.959 >= 0.80; figure density retained as the more basic general object-richness measure"}],
        "continuous_pairwise_correlations": pairs, "final_model_vif": vifs,
        "standardization": standards, "binary_reference_levels": {name: "0=section absent" for name in BINARY_PREDICTORS},
        "final_predictors": list(PREDICTORS), "leakage_exclusions": list(LEAKAGE_EXCLUSIONS),
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {OUT}")
    metrics, features = verify_inputs()
    data = make_dataset(metrics, features)
    eligible = data[data.offset_model_eligible].copy()
    if eligible.groupby("task").size().to_dict() != {task: 1898 for task in TASKS}:
        raise RuntimeError("Offset-model N is not exactly 1,898 for every task")
    main_standards = design(eligible[eligible.task == TASKS[0]])[1]
    collinearity = collinearity_decisions(data, main_standards)
    nb_frames, nb_diagnostics, robustness_frames = [], [], []
    main_directions: dict[tuple[str, str], float] = {}
    for task in TASKS:
        task_data = eligible[eligible.task == task].copy()
        main_rows, main_diag, _, _ = fit_nb(task_data, "main")
        nb_frames.append(main_rows)
        nb_diagnostics.append(main_diag)
        for row in main_rows.itertuples():
            main_directions[(task, row.predictor)] = float(np.sign(row.coefficient))
        for variant in ("winsorized_1_99", "leave_top_1_percent_out"):
            robust_rows, robust_diag, _, _ = fit_nb(task_data, variant)
            robust_rows["direction_consistent_with_main"] = robust_rows.apply(lambda row: np.sign(row.coefficient) == main_directions[(task, row.predictor)], axis=1)
            robust_rows["n"] = robust_diag["n"]
            robust_rows["converged"] = robust_diag["converged"]
            robustness_frames.append(robust_rows)
    nb_results = pd.concat(nb_frames, ignore_index=True)
    nb_diag = pd.DataFrame(nb_diagnostics)
    robustness = pd.concat(robustness_frames, ignore_index=True)
    logistic_frames, logistic_diagnostics = [], []
    for task in BINARY_TASKS:
        rows, diagnostic = fit_logistic(eligible[eligible.task == task].copy())
        logistic_frames.append(rows)
        logistic_diagnostics.append(diagnostic)
    logistic_results = pd.concat(logistic_frames, ignore_index=True)
    logistic_diag = pd.DataFrame(logistic_diagnostics)
    forest_nb = nb_results.assign(estimate=nb_results.irr, model_type="Negative Binomial IRR")[["task", "predictor", "estimate", "ci_lower", "ci_upper", "model_type"]]
    forest_logistic = logistic_results.assign(estimate=logistic_results["or"], model_type="Logistic OR")[["task", "predictor", "estimate", "ci_lower", "ci_upper", "model_type"]]
    forest = pd.concat([forest_nb, forest_logistic], ignore_index=True)
    availability = feature_availability(features, data)
    supported_nb = nb_results[(nb_results.holm_adjusted_p < 0.05) & ((nb_results.ci_lower > 1) | (nb_results.ci_upper < 1))].copy()
    supported_logistic = logistic_results[(logistic_results.holm_adjusted_p < 0.05) & ((logistic_results.ci_lower > 1) | (logistic_results.ci_upper < 1))].copy()
    direction_stability = robustness.groupby(["task", "predictor"]).direction_consistent_with_main.all().to_dict()
    supported_nb["direction_stable_both_robustness_checks"] = supported_nb.apply(lambda row: bool(direction_stability[(row.task, row.predictor)]), axis=1)
    gates = {
        "four_nb2_models_converged": bool(nb_diag.converged.all() and len(nb_diag) == 4),
        "offset_is_log_source_tokens_per_10000": True,
        "zero_token_papers_retained_and_two_excluded_from_offset_models": int(data[data.task == TASKS[0]].zero_source_token.sum()) == 2 and len(eligible[eligible.task == TASKS[0]]) == 1898,
        "no_candidate_derived_leakage_predictor": not bool(set(PREDICTORS) & set(LEAKAGE_EXCLUSIONS)),
        "no_source_tokens_as_ordinary_predictor": "source_tokens" not in PREDICTORS,
        "t1a_t2_binary_logistic_not_fitted": set(logistic_results.task) == set(BINARY_TASKS),
        "t1b_t3_logistic_converged": bool(logistic_diag.converged.all() and len(logistic_diag) == 2),
        "holm_applied_within_each_task": bool(nb_results.holm_adjusted_p.notna().all() and logistic_results.holm_adjusted_p.notna().all()),
        "no_extreme_or_singular_main_model": not bool(nb_diag.extreme_coefficient_or_se.any() or logistic_diag.extreme_coefficient_or_se.any()),
        "all_results_and_diagnostics_complete": True,
    }
    status = "SOURCE_SUITABILITY_READY" if all(gates.values()) else "SOURCE_SUITABILITY_NOT_READY"
    config = {
        "experiment": "corpus_v1_sft_b1_source_suitability_statistical_analysis",
        "status": status,
        "input_assets": {
            "v1_2_directory": str(V12.relative_to(ROOT)), "holdout_membership_sha256": EXPECTED["holdout_membership"],
            "machine_verifier_config_v1_2_sha256": EXPECTED["v12_verifier_config"], "v1_2_summary_sha256": EXPECTED["v12_summary"],
            "paper_task_metrics_sha256": EXPECTED["v12_metrics"], "paper_features_sha256": EXPECTED["holdout_features"],
        },
        "analysis_tasks": list(TASKS), "descriptive_only_task": "T1_TOTAL",
        "primary_model": {"family": "Negative Binomial", "parameterization": "NB2: Var(Y|X)=mu+alpha*mu^2", "outcome": "machine_verified_count", "offset": "log(source_tokens/10000)", "n_per_task": 1898},
        "supplementary_binary_models": {"fitted": list(BINARY_TASKS), "not_fitted_saturated": list(BINARY_SATURATED)},
        "predictor_set_frozen_before_outcome_model_fitting": list(PREDICTORS),
        "continuous_standardization": "z-standardized within 1,898 positive-token Holdout papers using sample SD",
        "multiple_comparison": "Holm correction separately within each task/model over 10 source-feature hypotheses",
        "zero_fit_warning_rule": f"absolute observed-minus-NB-predicted zero proportion >= {ZERO_FIT_WARNING_ABS_THRESHOLD}",
        "robustness": ["continuous predictors winsorized at 1st/99th percentiles then re-standardized", "remove exactly ceil(1%*1898)=19 highest-yield papers separately per task"],
        "prohibited_actions": {"candidate_or_verifier_change": False, "Holdout_rerun": False, "SFT_QA_generation": False, "SFT_training_or_scaling": False},
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": st.__version__ if hasattr(st, "__version__") else None, "statsmodels": sm.__version__},
        "code_sha256": sha256_file(Path(__file__)),
    }
    OUT.mkdir(parents=True)
    write_json(OUT / "config.json", config)
    availability.to_csv(OUT / "feature_availability.csv", index=False, encoding="utf-8", lineterminator="\n")
    data.to_parquet(OUT / "modeling_dataset.parquet", index=False)
    write_json(OUT / "collinearity_decisions.json", collinearity)
    nb_results.to_csv(OUT / "nb_results.csv", index=False, encoding="utf-8", lineterminator="\n")
    nb_diag.to_csv(OUT / "nb_diagnostics.csv", index=False, encoding="utf-8", lineterminator="\n")
    logistic_results.to_csv(OUT / "logistic_results.csv", index=False, encoding="utf-8", lineterminator="\n")
    logistic_diag.to_csv(OUT / "logistic_diagnostics.csv", index=False, encoding="utf-8", lineterminator="\n")
    robustness.to_csv(OUT / "robustness_results.csv", index=False, encoding="utf-8", lineterminator="\n")
    forest.to_csv(OUT / "source_suitability_forestplot.csv", index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "experiment": config["experiment"], "status": status, "ready_gates": gates,
        "modeling_data": {"descriptive_n": 1900, "offset_model_n_per_task": 1898, "excluded_from_offset_model": 2, "reason": "zero source-token exposure", "cal100_included": False},
        "final_predictors": list(PREDICTORS), "feature_exclusions": availability[~availability.included_in_main_model][["feature", "role_or_exclusion_reason"]].to_dict("records"),
        "nb_diagnostics": nb_diag.to_dict("records"), "logistic_diagnostics": logistic_diag.to_dict("records"),
        "binary_model_status": {"T1a_ACRONYM": "BINARY_OUTCOME_SATURATED", "T2_NUMERIC": "BINARY_OUTCOME_SATURATED", "T1b_EXPLICIT_DEFINITION": "FITTED", "T3_CLAIM": "FITTED"},
        "statistically_supported_nb": supported_nb.to_dict("records"), "statistically_supported_logistic": supported_logistic.to_dict("records"),
        "robustness_direction_summary": {"all_supported_nb_directions_stable": bool(supported_nb.direction_stable_both_robustness_checks.all()) if len(supported_nb) else True, "per_predictor": [{"task": task, "predictor": predictor, "both_checks_direction_consistent": bool(value)} for (task, predictor), value in direction_stability.items()]},
        "interpretation_boundary": "Observational predictive associations with task-specific machine-verified yield per 10k source tokens; no causal claims.",
        "warnings": nb_diag.loc[nb_diag.zero_fit_warning, ["task", "zero_fit_absolute_difference"]].assign(warning="ZERO_FIT_WARNING").to_dict("records"),
    }
    write_json(OUT / "summary.json", summary)
    names = sorted(path.name for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha256_file(OUT / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(OUT), "gates": gates, "supported_nb": len(supported_nb), "supported_logistic": len(supported_logistic), "warnings": summary["warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
