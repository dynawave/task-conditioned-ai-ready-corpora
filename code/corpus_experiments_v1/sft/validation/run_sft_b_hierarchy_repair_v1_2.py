from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SFT_DIR = HERE.parent
ROOT = HERE.parents[3]

import sys
sys.path.insert(0, str(SFT_DIR))
sys.path.insert(0, str(HERE))

from common import sha256_file  # noqa: E402


CAL = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/calibration_v1"
V1 = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/machine_validation_v1"
V11 = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/machine_validation_v1_1"
HOLDOUT_V11 = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_b_holdout1900"
OUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_b_holdout1900_v1_2"
CANONICAL = ROOT / "aicorpus-derived/experiments/corpus_v1/pilot2k/run_02/canonical_documents.parquet"

EXPECTED = {
    "canonical": "2c72b3615d5299534d6336c9d4a4b4db0982853011718babeabc2aafa18e007b",
    "holdout_membership": "e4066ba08a65e69055eb795f07b7b97715930dcce0f21a48819af35fe4541b4f",
    "v11_config": "ae5a45cc055a3f151e77e97f0d59119956f39c1a8fdd4c03a8d3b7a8838c8cf3",
    "cal_labels_v1": "23abbae683e5e8fdb41634b97fcf21044cc6d80a878a02178b017c13273cdb4e",
    "holdout_labels_v11": "b2354915bc2fdca980b2662011df692a111ba23d8754092b5c7977428150862f",
    "holdout_features": "786c4daf67e9c17a94e004efb339c83b35e2c550c2f10bf2549aac90ce4319b4",
}
TASKS = ("T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION", "T2_NUMERIC", "T3_CLAIM")
SUMMARY_TASKS = (*TASKS[:2], "T1_TOTAL", *TASKS[2:])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def membership_sha(source_ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(source_ids.astype(str).tolist()).encode("utf-8")).hexdigest()


def report_task(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame.task_type == "T1_TERM",
        np.where(frame.task_subtype == "explicit_definition", "T1b_EXPLICIT_DEFINITION", "T1a_ACRONYM"),
        frame.task_type,
    )


def code_hashes() -> dict[str, str]:
    paths = {
        "rules.py": SFT_DIR / "rules.py",
        "quality_gates.py": SFT_DIR / "quality_gates.py",
        "detectors.py": SFT_DIR / "detectors.py",
        "validation/run_machine_validation.py": HERE / "run_machine_validation.py",
        "validation/run_sft_b_hierarchy_repair_v1_2.py": Path(__file__),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = {
        "canonical": CANONICAL,
        "v11_config": V11 / "machine_verifier_config_v1_1.json",
        "cal_labels_v1": V1 / "cal100_machine_verified.parquet",
        "holdout_labels_v11": HOLDOUT_V11 / "candidates_holdout1900.parquet",
        "holdout_features": HOLDOUT_V11 / "paper_features_holdout1900.parquet",
    }
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(f"Frozen input hash mismatch for {name}: {actual} != {EXPECTED[name]}")
    holdout_manifest = pd.read_csv(CAL / "holdout1900_manifest.csv")
    cal_manifest = pd.read_csv(CAL / "cal100_manifest.csv")
    if len(holdout_manifest) != 1900 or holdout_manifest.source_id.nunique() != 1900:
        raise RuntimeError("Holdout manifest is not 1,900 unique source IDs")
    if membership_sha(holdout_manifest.source_id) != EXPECTED["holdout_membership"]:
        raise RuntimeError("Frozen holdout membership SHA256 mismatch")
    if set(holdout_manifest.source_id.astype(str)) & set(cal_manifest.source_id.astype(str)):
        raise RuntimeError("Cal100/Holdout1900 membership overlap")
    verifier_config = json.loads((V11 / "machine_verifier_config_v1_1.json").read_text(encoding="utf-8"))
    if verifier_config["experiment"] != "corpus_v1_sft_a_machine_validation_v1_1" or not verifier_config["T3"]["candidate_decisions_unchanged_from_v1"]:
        raise RuntimeError("v1.1 frozen candidate-decision contract is not available")
    cal = pd.read_parquet(V1 / "cal100_machine_verified.parquet")
    holdout = pd.read_parquet(HOLDOUT_V11 / "candidates_holdout1900.parquet")
    holdout_features = pd.read_parquet(HOLDOUT_V11 / "paper_features_holdout1900.parquet")
    cal_features = pd.read_csv(CAL / "paper_features_cal100.csv")
    for name, frame in {"Cal100": cal, "Holdout1900": holdout}.items():
        if frame.candidate_id.duplicated().any() or not frame.raw_opportunity.astype(bool).all():
            raise RuntimeError(f"{name} candidate records fail frozen identity/raw checks")
    holdout_ids = set(holdout_manifest.source_id.astype(str))
    cal_ids = set(cal_manifest.source_id.astype(str))
    if not set(holdout.source_id.astype(str)).issubset(holdout_ids):
        raise RuntimeError("Holdout candidate source IDs are not a subset of the frozen manifest")
    if not set(cal.source_id.astype(str)).issubset(cal_ids):
        raise RuntimeError("Cal100 candidate source IDs are not a subset of the frozen manifest")
    if set(holdout_features.source_id.astype(str)) != holdout_ids or len(holdout_features) != 1900:
        raise RuntimeError("Holdout feature membership differs from the frozen 1,900-paper manifest")
    if set(cal_features.source_id.astype(str)) != cal_ids or len(cal_features) != 100:
        raise RuntimeError("Cal100 feature membership differs from the frozen 100-paper manifest")
    return cal, holdout, cal_features, holdout_features, verifier_config


def parse_scores(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    return json.loads(str(value))


def primitive_acceptance(row: pd.Series, threshold: float) -> bool:
    """Rebuild frozen v1.1 acceptance from stored verifier primitive outputs only."""
    if not bool(row["hard_gate_pass"]):
        return False
    scores = parse_scores(row["verifier_scores"])
    if row["task_type"] == "T1_TERM":
        if row["task_subtype"] == "explicit_definition":
            return bool(scores.get("exact_term")) and bool(scores.get("exact_definition")) and scores.get("explicit_cue") is not None
        counterfactual = scores.get("counterfactual", {})
        return (
            bool(scores.get("exact_term"))
            and bool(scores.get("exact_definition"))
            and bool(scores.get("independent_alignment"))
            and counterfactual.get("pass") is not False
        )
    if row["task_type"] == "T2_NUMERIC":
        required = (
            "exact_span", "normalization", "unit_consistent", "semantic_context",
            "roundtrip_exact", "roundtrip_normalized", "answer_uniqueness_check",
        )
        return all(scores.get(key) is True for key in required)
    if row["task_type"] == "T3_CLAIM":
        nli = scores.get("nli", {})
        deterministic = scores.get("deterministic", {})
        deterministic_ok = "condition_keys_checked" in deterministic and not deterministic.get("missing") and not deterministic.get("invalid")
        return float(nli.get("entailment", -1.0)) >= threshold and deterministic_ok
    raise RuntimeError(f"Unknown task type: {row['task_type']}")


def derive_v12(labels: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = labels.copy()
    output["report_task"] = report_task(output)
    output["machine_verified_v1_1"] = output.machine_verified.astype(bool)
    output["frozen_verifier_accept_v1_1"] = output.apply(primitive_acceptance, axis=1, threshold=threshold)
    mismatch = output.machine_verified_v1_1 != output.frozen_verifier_accept_v1_1
    if mismatch.any():
        raise RuntimeError(f"Stored v1.1 labels and reconstructed primitive acceptance differ for {int(mismatch.sum())} candidates")
    output["machine_verified_v1_2"] = output.materialization_ready.astype(bool) & output.frozen_verifier_accept_v1_1
    output["machine_reject_reason_v1_2"] = output.machine_reject_reason
    hierarchy_rejected = output.frozen_verifier_accept_v1_1 & ~output.materialization_ready.astype(bool)
    output.loc[hierarchy_rejected, "machine_reject_reason_v1_2"] = "materialization_ready_required"
    output.loc[output.machine_verified_v1_2, "machine_reject_reason_v1_2"] = None
    output["verification_level_v1_2"] = output.verification_level
    output.loc[hierarchy_rejected, "verification_level_v1_2"] = "REJECTED_HIERARCHY_MATERIALIZATION_NOT_READY"
    output["verification_scope_v1_2"] = output.verification_scope
    output.loc[output.report_task == "T1a_ACRONYM", "verification_scope_v1_2"] = "external_gold_calibrated_rule"
    output.loc[output.report_task == "T1b_EXPLICIT_DEFINITION", "verification_scope_v1_2"] = "deterministic_rule_verified_only"
    output["v1_2_transition"] = np.where(
        hierarchy_rejected, "v1_1_accepted_but_not_ready_rejected_in_v1_2",
        np.where(output.machine_verified_v1_2, "accepted_unchanged", "not_accepted_by_frozen_verifier"),
    )
    violations = output[output.machine_verified_v1_1 & ~output.materialization_ready.astype(bool)].copy()
    violations["materialization_ready_false_reason"] = np.where(
        violations.auto_flags.str.contains("materialization_requires_unambiguous_statistical_relation", regex=False),
        "materialization_requires_unambiguous_statistical_relation",
        np.where(
            violations.auto_flags.str.contains("materialization_deferred_long_claim", regex=False),
            "materialization_deferred_long_claim",
            "other_materialization_ready_false_reason",
        ),
    )
    return output, violations


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "source_tokens" not in output:
        output["source_tokens"] = output["paper_clean_token_count"]
    return output[["paper_id", "source_id", "source_tokens"]].drop_duplicates(["paper_id", "source_id"])


def paper_task_metrics(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_by_id = features.set_index("paper_id").to_dict("index")
    grouped = {(str(paper), task): group for (paper, task), group in labels.groupby(["paper_id", "report_task"], sort=False)}
    scopes = {
        "T1a_ACRONYM": "external_gold_calibrated_rule",
        "T1b_EXPLICIT_DEFINITION": "deterministic_rule_verified_only",
        "T2_NUMERIC": "qasper_numeric_calibrated_rule_with_scitat_diagnostic",
        "T3_CLAIM": "scifact_calibrated_nli_plus_deterministic_conditions",
    }
    rows: list[dict[str, Any]] = []
    for paper_id, source_id in features[["paper_id", "source_id"]].itertuples(index=False, name=None):
        paper_id, source_id = str(paper_id), str(source_id)
        tokens = feature_by_id[paper_id]["source_tokens"]
        valid_tokens = pd.notna(tokens) and float(tokens) > 0
        for task in TASKS:
            group = grouped.get((paper_id, task))
            raw = len(group) if group is not None else 0
            hard = int(group.hard_gate_pass.sum()) if group is not None else 0
            ready = int(group.materialization_ready.sum()) if group is not None else 0
            mv = int(group.machine_verified_v1_2.sum()) if group is not None else 0
            rows.append({
                "paper_id": paper_id, "source_id": source_id, "task": task,
                "raw_count": raw, "hard_pass_count": hard, "ready_count": ready,
                "machine_verified_count": mv, "source_tokens": int(tokens) if valid_tokens else None,
                "source_tokens_valid": bool(valid_tokens),
                "mv_per_10k_tokens": mv * 10000 / float(tokens) if valid_tokens else None,
                "has_raw": raw > 0, "has_hard_pass": hard > 0, "has_ready": ready > 0,
                "has_machine_verified": mv > 0, "verification_scope": scopes[task],
            })
    output = pd.DataFrame(rows)
    if len(output) != len(features) * len(TASKS):
        raise RuntimeError("paper_task_metrics cardinality mismatch")
    return output


def with_t1_total(metrics: pd.DataFrame) -> pd.DataFrame:
    t1 = metrics[metrics.task.isin(["T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION"])]
    total = t1.groupby(["paper_id", "source_id"], as_index=False).agg(
        raw_count=("raw_count", "sum"), hard_pass_count=("hard_pass_count", "sum"),
        ready_count=("ready_count", "sum"), machine_verified_count=("machine_verified_count", "sum"),
        source_tokens=("source_tokens", "first"), source_tokens_valid=("source_tokens_valid", "first"),
    )
    total["task"] = "T1_TOTAL"
    total["mv_per_10k_tokens"] = np.where(total.source_tokens_valid, total.machine_verified_count * 10000 / total.source_tokens, np.nan)
    for binary, count in (("has_raw", "raw_count"), ("has_hard_pass", "hard_pass_count"), ("has_ready", "ready_count"), ("has_machine_verified", "machine_verified_count")):
        total[binary] = total[count] > 0
    total["verification_scope"] = "mixed_scope_report_T1a_and_T1b_separately"
    return pd.concat([metrics, total[metrics.columns]], ignore_index=True)


def stat(values: pd.Series, include_max: bool = True) -> dict[str, float | None]:
    array = values.dropna().astype(float).to_numpy()
    if not len(array):
        return {key: None for key in ("mean", "sd", "variance", "median", "iqr", "p90", "p95", "max")}
    q1, median, q3, p90, p95 = np.percentile(array, [25, 50, 75, 90, 95])
    return {
        "mean": float(array.mean()), "sd": float(statistics.stdev(array)) if len(array) > 1 else 0.0,
        "variance": float(statistics.variance(array)) if len(array) > 1 else 0.0,
        "median": float(median), "iqr": float(q3 - q1), "p90": float(p90), "p95": float(p95),
        "max": float(array.max()) if include_max else None,
    }


def summaries(metrics: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = with_t1_total(metrics)
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for task in SUMMARY_TASKS:
        part = expanded[expanded.task == task]
        count = stat(part.machine_verified_count)
        rate = stat(part.mv_per_10k_tokens, include_max=False)
        prevalence = float(part.has_machine_verified.mean())
        zero = float((part.machine_verified_count == 0).mean())
        ratio = count["variance"] / count["mean"] if count["mean"] else None
        saturated = prevalence > 0.90 or prevalence < 0.10
        summary_rows.append({
            "split": split, "task": task,
            "raw": int(part.raw_count.sum()), "hard_pass": int(part.hard_pass_count.sum()), "ready": int(part.ready_count.sum()),
            "machine_verified": int(part.machine_verified_count.sum()), "papers_ge_1_machine_verified": int(part.has_machine_verified.sum()),
            "paper_coverage_percent": 100 * prevalence,
            "mv_count_mean": count["mean"], "mv_count_sd": count["sd"], "mv_count_median": count["median"], "mv_count_iqr": count["iqr"],
            "mv_count_p90": count["p90"], "mv_count_p95": count["p95"], "mv_count_max": count["max"], "mv_count_zero_proportion": zero,
            "mv_per_10k_mean": rate["mean"], "mv_per_10k_sd": rate["sd"], "mv_per_10k_median": rate["median"],
            "mv_per_10k_iqr": rate["iqr"], "mv_per_10k_p90": rate["p90"], "mv_per_10k_p95": rate["p95"],
        })
        diagnostic_rows.append({
            "split": split, "task": task, "mean_machine_verified_count": count["mean"],
            "variance_machine_verified_count": count["variance"], "variance_to_mean": ratio, "zero_proportion": zero,
            "has_machine_verified_prevalence": prevalence, "binary_outcome_saturated": saturated,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(diagnostic_rows)


def audit(split: str, labels: pd.DataFrame) -> list[dict[str, Any]]:
    before_ready_hard = int((labels.materialization_ready & ~labels.hard_gate_pass).sum())
    before_mv_ready = int((labels.machine_verified_v1_1 & ~labels.materialization_ready).sum())
    after_mv_ready = int((labels.machine_verified_v1_2 & ~labels.materialization_ready).sum())
    after_mv_hard = int((labels.machine_verified_v1_2 & ~labels.hard_gate_pass).sum())
    return [
        {"split": split, "invariant": "MATERIALIZATION_READY_implies_HARD_GATE_PASS", "before_violation_count": before_ready_hard, "after_violation_count": before_ready_hard, "passed_v1_2": before_ready_hard == 0},
        {"split": split, "invariant": "MACHINE_VERIFIED_implies_MATERIALIZATION_READY", "before_violation_count": before_mv_ready, "after_violation_count": after_mv_ready, "passed_v1_2": after_mv_ready == 0},
        {"split": split, "invariant": "MACHINE_VERIFIED_implies_HARD_GATE_PASS", "before_violation_count": int((labels.machine_verified_v1_1 & ~labels.hard_gate_pass).sum()), "after_violation_count": after_mv_hard, "passed_v1_2": after_mv_hard == 0},
        {"split": split, "invariant": "candidate_id_unique", "before_violation_count": int(labels.candidate_id.duplicated().sum()), "after_violation_count": int(labels.candidate_id.duplicated().sum()), "passed_v1_2": not labels.candidate_id.duplicated().any()},
    ]


def provenance_coverage(labels: pd.DataFrame) -> dict[str, Any]:
    mv = labels[labels.machine_verified_v1_2]
    provenance_ok = mv.apply(lambda row: bool(row.provenance and row.paragraph_id and row.bbox and pd.notna(row.page)), axis=1)
    t12 = mv[mv.report_task.isin(["T1a_ACRONYM", "T1b_EXPLICIT_DEFINITION", "T2_NUMERIC"])]
    t3 = mv[mv.report_task == "T3_CLAIM"]
    def span_ok(frame: pd.DataFrame) -> int:
        return int(frame.apply(lambda row: str(row.evidence_clean)[int(row.answer_start):int(row.answer_end)] == str(row.answer_span), axis=1).sum())
    return {
        "mv_provenance": {"n": int(len(mv)), "passed": int(provenance_ok.sum()), "coverage": float(provenance_ok.mean()) if len(mv) else None},
        "t1_t2_mv_answer_span_traceability": {"n": int(len(t12)), "passed": span_ok(t12), "coverage": float(span_ok(t12) / len(t12)) if len(t12) else None},
        "t3_mv_original_claim_evidence_span_integrity": {"n": int(len(t3)), "passed": span_ok(t3), "coverage": float(span_ok(t3) / len(t3)) if len(t3) else None},
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {OUT}")
    cal_v1, holdout_v11, cal_features_raw, holdout_features_raw, v11 = load_inputs()
    threshold = float(v11["T3"]["frozen_scifact_and_nli_config"]["selected_entailment_threshold"])
    cal_v12, cal_violations = derive_v12(cal_v1, threshold)
    holdout_v12, holdout_violations = derive_v12(holdout_v11, threshold)
    cal_features, holdout_features = make_features(cal_features_raw), make_features(holdout_features_raw)
    cal_metrics = paper_task_metrics(cal_v12, cal_features)
    holdout_metrics = paper_task_metrics(holdout_v12, holdout_features)
    cal_summary, cal_diagnostics = summaries(cal_metrics, "Cal100")
    holdout_summary, holdout_diagnostics = summaries(holdout_metrics, "Holdout1900")
    audit_frame = pd.DataFrame(audit("Cal100", cal_v12) + audit("Holdout1900", holdout_v12))
    details = pd.concat([cal_violations.assign(split="Cal100"), holdout_violations.assign(split="Holdout1900")], ignore_index=True)
    keep = ["split", "candidate_id", "report_task", "paper_id", "source_id", "task_type", "task_subtype", "stage", "auto_flags", "materialization_ready_false_reason", "machine_verified_v1_1", "frozen_verifier_accept_v1_1", "machine_verified_v1_2", "machine_reject_reason_v1_2"]
    details = details[keep]
    cal_integrity, holdout_integrity = provenance_coverage(cal_v12), provenance_coverage(holdout_v12)
    final_audit_ok = bool((audit_frame.after_violation_count == 0).all() and audit_frame.passed_v1_2.all())
    gates = {
        "cal100_hierarchy_violations_zero": final_audit_ok and int((cal_v12.machine_verified_v1_2 & ~cal_v12.materialization_ready).sum()) == 0,
        "holdout_hierarchy_violations_zero": final_audit_ok and int((holdout_v12.machine_verified_v1_2 & ~holdout_v12.materialization_ready).sum()) == 0,
        "holdout_1900_of_1900": len(holdout_features) == 1900,
        "holdout_membership_sha_match": membership_sha(pd.read_csv(CAL / "holdout1900_manifest.csv").source_id) == EXPECTED["holdout_membership"],
        "canonical_sha_match": sha256_file(CANONICAL) == EXPECTED["canonical"],
        "cal_candidate_ids_unique": not cal_v12.candidate_id.duplicated().any(),
        "holdout_candidate_ids_unique": not holdout_v12.candidate_id.duplicated().any(),
        "cal_mv_provenance_100_percent": cal_integrity["mv_provenance"]["coverage"] in {1.0, None},
        "holdout_mv_provenance_100_percent": holdout_integrity["mv_provenance"]["coverage"] in {1.0, None},
        "holdout_t1_t2_span_traceability_100_percent": holdout_integrity["t1_t2_mv_answer_span_traceability"]["coverage"] in {1.0, None},
        "holdout_t3_claim_span_integrity_100_percent": holdout_integrity["t3_mv_original_claim_evidence_span_integrity"]["coverage"] in {1.0, None},
    }
    status = "SFT_B_HOLDOUT_YIELD_READY" if all(gates.values()) else "SFT_B_HOLDOUT_YIELD_NOT_READY"
    v12_config = {
        "experiment": "corpus_v1_sft_b_hierarchy_repair_v1_2",
        "derived_from": {"machine_verifier_config_v1_1_path": str((V11 / "machine_verifier_config_v1_1.json").relative_to(ROOT)), "sha256": EXPECTED["v11_config"]},
        "change_type": "LOGICAL_HIERARCHY_REPAIR_ONLY",
        "machine_verified_v1_2": "materialization_ready_v1_1 AND frozen_verifier_accept_v1_1",
        "frozen_verifier_accept_v1_1": "deterministically reconstructed from persisted v1.1 verifier primitive scores/flags; exact equality to stored v1.1 labels is required",
        "unchanged": {
            "detectors": True, "hard_gates": True, "materialization_ready_definition": True,
            "external_validation": True, "nli_model_revision": v11["T3"]["NLI"]["model_revision"],
            "scifact_threshold": threshold, "counterfactual_checks": True,
        },
        "no_expensive_steps_rerun": {"PDF_parse": False, "candidate_detection": False, "NLI_inference": False, "external_benchmark": False},
    }
    OUT.mkdir(parents=True)
    write_json(OUT / "machine_verifier_config_v1_2.json", v12_config)
    run_config = {
        "experiment": v12_config["experiment"], "status": status,
        "machine_verifier_config_v1_2_sha256": sha256_file(OUT / "machine_verifier_config_v1_2.json"),
        "inputs": {**EXPECTED, "cal_features_sha256": sha256_file(CAL / "paper_features_cal100.csv")},
        "code_sha256": code_hashes(), "code_revision": {"git_commit": None, "git_commit_status": "workspace_not_a_git_repository"},
        "holdout_documents_processed": 1900, "cal100_documents_processed": 0,
        "repair_scope": "logical hierarchy only; no detector/gate/ready/verifier/NLI/threshold/counterfactual change",
    }
    write_json(OUT / "config.json", run_config)
    audit_frame.to_csv(OUT / "hierarchy_audit.csv", index=False, encoding="utf-8", lineterminator="\n")
    details.to_parquet(OUT / "hierarchy_violation_details.parquet", index=False)
    cal_v12.to_parquet(OUT / "cal100_candidate_labels_v1_2.parquet", index=False)
    holdout_v12.to_parquet(OUT / "holdout_candidate_labels_v1_2.parquet", index=False)
    cal_summary.to_csv(OUT / "cal100_task_summary.csv", index=False, encoding="utf-8", lineterminator="\n")
    holdout_summary.to_csv(OUT / "holdout_task_summary.csv", index=False, encoding="utf-8", lineterminator="\n")
    holdout_metrics.to_parquet(OUT / "paper_task_metrics.parquet", index=False)
    pd.concat([cal_diagnostics, holdout_diagnostics], ignore_index=True).to_csv(OUT / "distribution_diagnostics.csv", index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "experiment": run_config["experiment"], "status": status, "root_cause": {
            "classification": "machine_verified implementation omitted the frozen materialization_ready prerequisite",
            "not_a_quality_threshold_change": True,
            "cal100_pre_repair_mv_not_ready": int(len(cal_violations)), "holdout_pre_repair_mv_not_ready": int(len(holdout_violations)),
            "pre_repair_reason_counts": details.groupby(["split", "materialization_ready_false_reason"]).size().reset_index(name="count").to_dict("records"),
        },
        "hierarchy_audit": audit_frame.to_dict("records"), "cal100_task_summary": cal_summary.to_dict("records"),
        "holdout_task_summary": holdout_summary.to_dict("records"),
        "distribution_diagnostics": holdout_diagnostics.to_dict("records"),
        "zero_source_token_papers": int((~holdout_metrics[["paper_id", "source_tokens_valid"]].drop_duplicates().source_tokens_valid).sum()),
        "cal100_integrity": cal_integrity, "holdout_integrity": holdout_integrity, "ready_gates": gates,
        "limitations": ["Cal100 remains development/calibration only.", "No source-suitability model or SFT training was run.", "v1.2 changes only final acceptance hierarchy, not frozen quality rules or thresholds."],
    }
    write_json(OUT / "summary.json", summary)
    names = sorted(path.name for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha256_file(OUT / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(OUT), "gates": gates, "pre_repair": {"cal": len(cal_violations), "holdout": len(holdout_violations)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
