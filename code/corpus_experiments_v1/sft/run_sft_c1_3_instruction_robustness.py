from __future__ import annotations

"""SFT-C1.3 secondary instruction-template robustness evaluation.

The only executable model operations in this script are greedy evaluation of
existing Base/LoRA artifacts.  No training, data mutation, or re-materializing
is implemented here.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.sft import run_sft_scaling_endpoint_v1 as c1  # noqa: E402


BASE = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1"
MATERIALIZATION = BASE / "materialization"
C1 = BASE / "training_endpoint_v1"
C11 = BASE / "endpoint_diagnostic_v1"
C12 = BASE / "endpoint_adjudication_v1"
OUT = BASE / "instruction_robustness_v1"
PROTOCOL_SHA = "d4fa8dd565a0204285e2b70fc6914689a9042213ea4844cec4b44656aa146489"
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
STEMS = {"T1a_ACRONYM": "t1a", "T2_NUMERIC": "t2", "T3_CLAIM": "t3"}
T1_RE = re.compile(r"^What does (.+) stand for in the passage\?$")
T2_RE = re.compile(r"^According to the passage, what value is reported immediately after (.+)\?$")
NUMBER_RE = c1.NUMBER_RE


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(directory: Path) -> None:
    manifest = directory / "checksums.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"Missing manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"Frozen input mismatch: {directory.name}/{relative}")


def extract_variable(task: str, instruction: str) -> str:
    match = (T1_RE if task == "T1a_ACRONYM" else T2_RE).match(instruction)
    if not match:
        raise RuntimeError(f"Could not extract frozen template variable for {task}: {instruction!r}")
    return match.group(1)


def rendered_instruction(task: str, template_id: str, original: str) -> str:
    if template_id == "TEMPLATE_0_PRIMARY":
        return original
    if task == "T1a_ACRONYM":
        short = extract_variable(task, original)
        return ("Identify the full form of {SHORT_FORM} from the passage." if template_id == "TEMPLATE_1" else
                "According to the passage, expand the abbreviation {SHORT_FORM}.").replace("{SHORT_FORM}", short)
    if task == "T2_NUMERIC":
        anchor = extract_variable(task, original)
        return ("According to the passage, what value is reported for {ANCHOR}?" if template_id == "TEMPLATE_1" else
                "Extract the reported value for {ANCHOR} from the passage.").replace("{ANCHOR}", anchor)
    if template_id == "TEMPLATE_1":
        return "Extract the scientific finding expressed in the passage. Preserve all stated conditions, polarity, negation, and uncertainty."
    return "Return the scientific claim stated in the passage using its original wording, including any explicit conditions and uncertainty."


def template_config() -> dict[str, Any]:
    c1.verify_inputs()
    verify_manifest(MATERIALIZATION); verify_manifest(C1); verify_manifest(C11); verify_manifest(C12)
    if sha_file(C1 / "training_protocol_v1.json") != PROTOCOL_SHA:
        raise RuntimeError("C1 frozen protocol SHA mismatch")
    sources = {"T1a_ACRONYM": MATERIALIZATION / "t1a_eval.parquet", "T2_NUMERIC": MATERIALIZATION / "t2_eval.parquet", "T3_CLAIM": MATERIALIZATION / "t3_eval.parquet"}
    templates = {
        "T1a_ACRONYM": {
            "TEMPLATE_0_PRIMARY": "What does {SHORT_FORM} stand for in the passage? (frozen per-instance original instruction)",
            "TEMPLATE_1": "Identify the full form of {SHORT_FORM} from the passage.",
            "TEMPLATE_2": "According to the passage, expand the abbreviation {SHORT_FORM}.",
        },
        "T2_NUMERIC": {
            "TEMPLATE_0_PRIMARY": "According to the passage, what value is reported immediately after {ANCHOR}? (frozen per-instance original instruction)",
            "TEMPLATE_1": "According to the passage, what value is reported for {ANCHOR}?",
            "TEMPLATE_2": "Extract the reported value for {ANCHOR} from the passage.",
        },
        "T3_CLAIM": {
            "TEMPLATE_0_PRIMARY": "Extract the scientific finding stated in the passage. Preserve the wording, conditions, polarity, negation, and uncertainty expressed in the evidence. (frozen per-instance original instruction)",
            "TEMPLATE_1": "Extract the scientific finding expressed in the passage. Preserve all stated conditions, polarity, negation, and uncertainty.",
            "TEMPLATE_2": "Return the scientific claim stated in the passage using its original wording, including any explicit conditions and uncertainty.",
        },
    }
    safety: dict[str, Any] = {}
    for task, source in sources.items():
        frame = pd.read_parquet(source)
        rendered: dict[str, list[str]] = {key: [rendered_instruction(task, key, item) for item in frame.instruction.astype(str)] for key in templates[task]}
        if task in {"T1a_ACRONYM", "T2_NUMERIC"} and any(not extract_variable(task, x) for x in frame.instruction.astype(str)):
            raise RuntimeError(f"Empty frozen template variable: {task}")
        # All target strings must remain outside every rendered instruction;
        # for T2 this additionally verifies the frozen anchors did not include
        # their numeric gold target.
        # Safety applies to the two newly introduced templates.  TEMPLATE_0 is
        # the frozen original instruction, retained without alteration.
        gold_in_instruction = 0
        for index, gold in enumerate(frame.response.astype(str)):
            normalized_gold = c1.normalize(gold)
            for template_id in ("TEMPLATE_1", "TEMPLATE_2"):
                items = rendered[template_id]
                gold_in_instruction += int(bool(normalized_gold) and normalized_gold in c1.normalize(items[index]))
        safety[task] = {
            "eval_instances": int(len(frame)), "eval_instance_ids_sha256": sha_text("\n".join(frame.instance_id.astype(str))),
            "context_unchanged": True, "gold_response_unchanged": True, "candidate_unchanged": True,
            "template_variables_extracted_from_frozen_instruction": task in {"T1a_ACRONYM", "T2_NUMERIC"},
            "gold_answer_occurrences_in_rendered_instruction": gold_in_instruction,
            "no_gold_answer_in_template_or_rendered_instruction": gold_in_instruction == 0,
            "semantic_scope": "same extractive target; no answer-format example, summary request, explanation request, or added reasoning requirement",
        }
        if gold_in_instruction:
            raise RuntimeError(f"Template safety failure: rendered {task} instruction contains a gold response")
    return {
        "experiment": "corpus_v1_sft_c1_3_pre_grid_instruction_template_robustness",
        "status": "frozen_before_template_evaluation", "parent_c1_protocol_sha256": PROTOCOL_SHA,
        "role": {"TEMPLATE_0_PRIMARY": "primary metric template", "TEMPLATE_1": "secondary robustness evaluation", "TEMPLATE_2": "secondary robustness evaluation"},
        "templates": templates, "safety_checks": safety,
        "evaluation": {"chat_template": "frozen official tokenizer.apply_chat_template", "generation": "greedy do_sample=False, num_beams=1", "normalization": "Unicode NFKC; trim; collapse whitespace; case-sensitive", "max_new_tokens": c1.GENERATION_MAX_NEW_TOKENS},
        "models": {"Base": "frozen base", "N100": "existing seed42 endpoint adapter", "N2000": "existing seed42 endpoint adapter", "T3_N2000_original": "existing original endpoint adapter", "T3_N2000_recovery": "existing C1.1 technical-replicate adapter"},
        "prohibited": ["training", "training-data modification", "gold/context/candidate modification", "N250/N500/N1000", "new training seeds", "NEM normalization modification"],
    }


def freeze() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refuse to overwrite: {OUT}")
    config = template_config()
    OUT.mkdir(parents=True)
    write_json(OUT / "template_config_v1.json", config)
    print(canonical({"template_config_frozen": True, "sha256": sha_file(OUT / "template_config_v1.json")}))


def numeric_values(value: str) -> list[str]:
    return [c1.normalize(item) for item in NUMBER_RE.findall(c1.normalize(value))]


def per_example_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_containment"] = [bool(gold) and gold in prediction for gold, prediction in zip(result.normalized_gold, result.normalized_prediction)]
    result["correct_target_with_extra_text"] = (~result.correct_NEM.astype(bool)) & result.target_containment
    result["prediction_token_count"] = [len(c1.tokens(x)) for x in result.prediction]
    result["gold_token_count"] = [len(c1.tokens(x)) for x in result.gold]
    result["prediction_gold_token_ratio"] = result.prediction_token_count / result.gold_token_count
    if str(result.task.iloc[0]) == "T2_NUMERIC":
        values = result.gold.map(numeric_values)
        result["numeric_value_containment"] = [all(value in prediction for value in target) if target else False for target, prediction in zip(values, result.normalized_prediction)]
    return result


def conditional_accuracy(frame: pd.DataFrame, name: str) -> tuple[int, float | None]:
    required = frame[f"{name}_required"].fillna(False).astype(bool)
    return int(required.sum()), (float(frame.loc[required, f"{name}_preserved"].astype(float).mean()) if required.any() else None)


def metric_row(frame: pd.DataFrame, evaluation: dict[str, Any]) -> dict[str, Any]:
    task = str(frame.task.iloc[0])
    row: dict[str, Any] = {
        "task": task, "model": str(frame.model.iloc[0]), "template_id": str(frame.template_id.iloc[0]), "instances": int(len(frame)),
        "normalized_exact_match": float(evaluation["normalized_exact_match"]), "exact_match": float(evaluation["exact_match"]),
        "target_containment_rate": float(frame.target_containment.mean()),
        "correct_target_with_extra_text_count": int(frame.correct_target_with_extra_text.sum()),
        "correct_target_with_extra_text_rate": float(frame.correct_target_with_extra_text.mean()),
        "correct_target_with_extra_text_among_nem_failures": float(frame.loc[~frame.correct_NEM.astype(bool), "correct_target_with_extra_text"].mean()) if (~frame.correct_NEM.astype(bool)).any() else None,
        "median_output_gold_token_length_ratio": float(frame.prediction_gold_token_ratio.median()),
        "generation_hit_max_new_tokens_count": int(evaluation["generation_hit_max_new_tokens_count"]),
    }
    if task == "T1a_ACRONYM":
        row["long_form_containment_rate"] = row["target_containment_rate"]
        row["case_insensitive_normalized_em"] = float(evaluation["case_insensitive_normalized_em"])
    elif task == "T2_NUMERIC":
        row["numeric_value_containment_rate"] = float(frame.numeric_value_containment.mean())
        row["unit_preservation_accuracy"] = evaluation.get("unit_preservation_accuracy")
        row["unit_preservation_instances"] = evaluation.get("unit_preservation_instances")
        row["operator_preservation_accuracy"] = evaluation.get("operator_preservation_accuracy")
        row["operator_preservation_instances"] = evaluation.get("operator_preservation_instances")
    else:
        row["claim_containment_rate"] = row["target_containment_rate"]
        row["token_f1"] = float(evaluation["token_f1"])
        for name in ("negation", "uncertainty", "condition"):
            count, accuracy = conditional_accuracy(frame, name)
            row[f"{name}_preservation_instances"] = count
            row[f"{name}_preservation_accuracy"] = accuracy
    return row


def evaluate_model(model: Any, tokenizer: Any, task: str, base_frame: pd.DataFrame, model_name: str) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for template_id in ("TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"):
        frame = base_frame.copy()
        frame["original_instruction"] = frame.instruction.astype(str)
        frame["instruction"] = [rendered_instruction(task, template_id, item) for item in frame.original_instruction]
        prediction, evaluation = c1.evaluate(model, tokenizer, frame, task, f"c1_3_{model_name}_{template_id}")
        prediction["model"] = model_name
        prediction["template_id"] = template_id
        prediction["original_instruction"] = frame.original_instruction.tolist()
        prediction["evaluation_instruction"] = frame.instruction.tolist()
        prediction = per_example_diagnostics(prediction)
        parts.append(prediction)
        metrics.append(metric_row(prediction, evaluation))
    return parts, metrics


def prediction_source(task: str, model_name: str) -> tuple[Path, dict[str, Any]]:
    if model_name == "Base":
        return C1 / "base_predictions.parquet", read_json(C1 / "base_metrics.json")["metrics"][task]
    stem = STEMS[task]
    if task == "T3_CLAIM" and model_name == "N2000_recovery":
        return C11 / "t3_n2000_seed42_recovery/predictions.parquet", read_json(C11 / "t3_n2000_seed42_recovery/metrics.json")["evaluation"]
    suffix = "n2000" if model_name in {"N2000", "N2000_original"} else "n100"
    return C1 / f"runs/{stem}_{suffix}_seed42/predictions.parquet", read_json(C1 / f"runs/{stem}_{suffix}_seed42/metrics.json")["evaluation"]


def expected_combinations() -> set[tuple[str, str, str]]:
    values: set[tuple[str, str, str]] = set()
    for task in ("T1a_ACRONYM", "T2_NUMERIC"):
        for model in ("Base", "N100", "N2000"):
            for template in ("TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"):
                values.add((task, model, template))
    for model in ("Base", "N100", "N2000_original", "N2000_recovery"):
        for template in ("TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"):
            values.add(("T3_CLAIM", model, template))
    return values


def append_results(parts: list[pd.DataFrame], metrics: list[dict[str, Any]]) -> None:
    predictions_path, metrics_path = OUT / "per_example_template_predictions.parquet", OUT / "template_metrics.csv"
    existing_parts = [pd.read_parquet(predictions_path)] if predictions_path.is_file() else []
    existing_metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    existing_keys = {(str(row["task"]), str(row["model"]), str(row["template_id"])) for row in existing_metrics}
    new_keys = {(str(row["task"]), str(row["model"]), str(row["template_id"])) for row in metrics}
    if len(new_keys) != len(metrics) or existing_keys & new_keys:
        raise RuntimeError("Refuse duplicate/overlapping template result append")
    pd.concat([*existing_parts, *parts], ignore_index=True).to_parquet(predictions_path, index=False)
    pd.DataFrame([*existing_metrics, *metrics]).sort_values(["task", "model", "template_id"]).to_csv(metrics_path, index=False)


def materialize_primary() -> None:
    config_path = OUT / "template_config_v1.json"
    if not config_path.is_file():
        raise RuntimeError("Template config must be frozen before inference")
    if (OUT / "template_metrics.csv").exists() or (OUT / "per_example_template_predictions.parquet").exists():
        raise RuntimeError("Primary result artifacts already exist; refuse overwrite")
    verify_manifest(MATERIALIZATION); verify_manifest(C1); verify_manifest(C11); verify_manifest(C12)
    tokenizer = AutoTokenizer.from_pretrained(c1.MODEL, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    _trains, evals, _lengths = c1.load_sets(tokenizer)
    all_parts: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    primary_models = {"T1a_ACRONYM": ("Base", "N100", "N2000"), "T2_NUMERIC": ("Base", "N100", "N2000"), "T3_CLAIM": ("Base", "N100", "N2000_original", "N2000_recovery")}
    for task, models in primary_models.items():
        for model_name in models:
            source, evaluation = prediction_source(task, model_name)
            prediction = pd.read_parquet(source)
            prediction = prediction[prediction.task == task].copy()
            if set(prediction.instance_id.astype(str)) != set(evals[task].instance_id.astype(str)):
                raise RuntimeError(f"Frozen primary prediction ID mismatch: {task}/{model_name}")
            instruction_by_id = evals[task].set_index("instance_id").instruction.astype(str).to_dict()
            prediction["model"] = model_name
            prediction["template_id"] = "TEMPLATE_0_PRIMARY"
            prediction["original_instruction"] = prediction.instance_id.astype(str).map(instruction_by_id)
            prediction["evaluation_instruction"] = prediction.original_instruction
            prediction["prediction_source"] = "frozen_existing_primary_prediction"
            prediction = per_example_diagnostics(prediction)
            all_parts.append(prediction)
            all_metrics.append(metric_row(prediction, evaluation))
    append_results(all_parts, all_metrics)
    print(canonical({"primary_results_materialized": True, "rows": int(sum(len(x) for x in all_parts)), "metrics": len(all_metrics)}))


def evaluate_slot(task: str, model_name: str, template_id: str) -> None:
    if task not in TASKS or template_id not in {"TEMPLATE_1", "TEMPLATE_2"}:
        raise RuntimeError("Only one secondary TEMPLATE_1/TEMPLATE_2 evaluation slot is allowed")
    allowed = {"T1a_ACRONYM": {"Base", "N100", "N2000"}, "T2_NUMERIC": {"Base", "N100", "N2000"}, "T3_CLAIM": {"Base", "N100", "N2000_original", "N2000_recovery"}}
    if model_name not in allowed[task]:
        raise RuntimeError("Invalid task/model secondary evaluation combination")
    config_path = OUT / "template_config_v1.json"
    if not config_path.is_file():
        raise RuntimeError("Template config must be frozen before inference")
    existing_metrics = pd.read_csv(OUT / "template_metrics.csv") if (OUT / "template_metrics.csv").is_file() else pd.DataFrame()
    if not existing_metrics.empty and ((existing_metrics.task == task) & (existing_metrics.model == model_name) & (existing_metrics.template_id == template_id)).any():
        raise RuntimeError("Secondary result already exists; refuse overwrite")
    verify_manifest(MATERIALIZATION); verify_manifest(C1); verify_manifest(C11); verify_manifest(C12)
    tokenizer = AutoTokenizer.from_pretrained(c1.MODEL, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    _trains, evals, _lengths = c1.load_sets(tokenizer)
    if model_name == "Base":
        c1.configure_seed(c1.SEED); model = c1.load_base().eval()
    else:
        _source, _evaluation = prediction_source(task, model_name)
        adapter = (C11 / "t3_n2000_seed42_recovery/adapter") if model_name == "N2000_recovery" else (C1 / f"runs/{STEMS[task]}_{'n2000' if model_name in {'N2000', 'N2000_original'} else 'n100'}_seed42/adapter")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"Missing frozen adapter: {adapter}")
        c1.configure_seed(c1.SEED); model = PeftModel.from_pretrained(c1.load_base(), adapter).eval()
    frame = evals[task].copy()
    frame["original_instruction"] = frame.instruction.astype(str)
    frame["instruction"] = [rendered_instruction(task, template_id, item) for item in frame.original_instruction]
    prediction, evaluation = c1.evaluate(model, tokenizer, frame, task, f"c1_3_{model_name}_{template_id}")
    del model; torch.cuda.empty_cache()
    prediction["model"] = model_name
    prediction["template_id"] = template_id
    prediction["original_instruction"] = frame.original_instruction.tolist()
    prediction["evaluation_instruction"] = frame.instruction.tolist()
    prediction["prediction_source"] = "c1_3_secondary_inference"
    prediction = per_example_diagnostics(prediction)
    append_results([prediction], [metric_row(prediction, evaluation)])
    print(canonical({"secondary_template_evaluation_completed": [task, model_name, template_id], "instances": len(prediction)}))


def formal_protocol(config_sha: str) -> dict[str, Any]:
    return {
        "experiment": "corpus_v1_sft_full_scale_grid_v1_2_pre_execution",
        "inherits": {"C1_primary_protocol_sha256": PROTOCOL_SHA, "C1_2_protocol": "endpoint_adjudication_v1/training_protocol_v1_1.json", "C1_3_template_config_sha256": config_sha},
        "status": "frozen_not_executed",
        "formal_runs": {"tasks": list(TASKS), "scales": [100, 250, 500, 1000, 2000], "training_seeds": [42, 314159, 271828], "independent_fresh_base_lora_per_task_scale_seed": True, "endpoint_adapters_role": "development_endpoint_only"},
        "primary_evaluation": {"template": "TEMPLATE_0_PRIMARY", "metric": "Normalized Exact Match", "normalization": "Unicode NFKC; trim; collapse whitespace; case-sensitive"},
        "secondary_template_robustness_evaluation": {"templates": ["TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"], "required_task_scales": [100, 1000, 2000], "required_training_seeds": [42, 314159, 271828], "preselected_before_grid_results": True, "role": "secondary robustness only; does not replace primary TEMPLATE_0 NEM"},
        "per_run_provenance_requirements": {
            "training_membership_hash": True, "training_instance_ids": True,
            "epoch_1_order_sha256": True, "epoch_2_order_sha256": True, "epoch_3_order_sha256": True,
            "order_definition": "actual DataLoader sample order, or an equivalent deterministic ordered instance-ID hash, after all shuffling",
            "rng_record": ["python_seed", "numpy_seed", "torch_cpu_seed", "torch_cuda_seed"],
            "environment_record": ["python_version", "pytorch_version", "transformers_version", "peft_version", "cuda_version", "cudnn_version", "gpu_model"],
            "determinism_record": ["torch_deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark", "tf32"],
            "telemetry": ["loss_masking_unit_test", "finite_loss", "optimization_steps", "training_tokens", "runtime", "peak_vram", "fresh_base_adapter_reload", "saved_predictions"],
            "bitwise_determinism_required": False,
        },
        "statistics": {"utility": "mean NEM across three independent training seeds", "delta_nem": 0.02, "delta_nem_use": "N2000 practical-equivalence/non-inferiority only, not reproducibility", "n_star": "hierarchical bootstrap: outer training-seed resampling + inner paired Eval-example resampling"},
        "reference_plateau_rule": "After grid only, mark REFERENCE_NOT_PLATEAUED if mean_NEM(N2000)-mean_NEM(N1000)>0.02.",
    }


def summarize() -> None:
    config_path, metric_path, pred_path = OUT / "template_config_v1.json", OUT / "template_metrics.csv", OUT / "per_example_template_predictions.parquet"
    if not all(path.is_file() for path in (config_path, metric_path, pred_path)):
        raise RuntimeError("Frozen config and complete inference outputs are required")
    metrics = pd.read_csv(metric_path)
    observed = {(str(row.task), str(row.model), str(row.template_id)) for row in metrics.itertuples(index=False)}
    if observed != expected_combinations() or len(metrics) != len(expected_combinations()):
        raise RuntimeError("Incomplete or duplicate task/model/template result grid")
    primary_sources = {"T1a_ACRONYM": ("Base", "N100", "N2000"), "T2_NUMERIC": ("Base", "N100", "N2000"), "T3_CLAIM": ("Base", "N100", "N2000_original", "N2000_recovery")}
    primary_regression: list[dict[str, Any]] = []
    for task, models in primary_sources.items():
        for model in models:
            _source, frozen = prediction_source(task, model)
            actual = float(metrics[(metrics.task == task) & (metrics.model == model) & (metrics.template_id == "TEMPLATE_0_PRIMARY")].normalized_exact_match.iloc[0])
            expected = float(frozen["normalized_exact_match"])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"TEMPLATE_0 regression mismatch: {task}/{model}: {actual} != {expected}")
            primary_regression.append({"task": task, "model": model, "frozen_primary_nem": expected, "c1_3_primary_nem": actual, "tolerance": 1e-12, "passed": True})
    robustness: list[dict[str, Any]] = []
    for (task, model), frame in metrics.groupby(["task", "model"], sort=True):
        values = frame.set_index("template_id").normalized_exact_match
        robustness.append({"task": task, "model": model, "primary_nem": float(values["TEMPLATE_0_PRIMARY"]), "paraphrase_mean_nem": float(values[["TEMPLATE_1", "TEMPLATE_2"]].mean()), "worst_template_nem": float(values.min()), "template_sd_sample": float(values.std(ddof=1)), "template_0_target_containment": float(frame.set_index("template_id").loc["TEMPLATE_0_PRIMARY", "target_containment_rate"]), "paraphrase_target_containment_mean": float(frame.set_index("template_id").loc[["TEMPLATE_1", "TEMPLATE_2"], "target_containment_rate"].mean())})
    robustness_frame = pd.DataFrame(robustness).sort_values(["task", "model"])
    # The qualitative rule is the one supplied by the researcher: observe
    # whether both unseen templates preserve a clear learned-vs-Base gap;
    # no new numerical threshold is introduced.
    signals: dict[str, str] = {}
    for task in TASKS:
        data = robustness_frame[robustness_frame.task == task].set_index("model")
        base = float(data.loc["Base", "paraphrase_mean_nem"])
        nonbase = [float(data.loc[index, "paraphrase_mean_nem"]) for index in data.index if index != "Base"]
        signals[task] = "YES" if nonbase and all(value > base for value in nonbase) else "UNCERTAIN"
    config_sha = sha_file(config_path)
    formal = formal_protocol(config_sha)
    write_json(OUT / "formal_scale_protocol_v1.json", formal)
    summary = {
        "experiment": "corpus_v1_sft_c1_3_pre_grid_instruction_template_robustness",
        "status": "FULL_SCALE_GRID_READY", "template_config_sha256": config_sha,
        "template_role": "TEMPLATE_0_PRIMARY remains the formal primary metric; TEMPLATE_1/2 are secondary robustness evaluation.",
        "robustness_metrics": robustness, "template_generalization_signal": signals,
        "template_0_regression": {"passed": True, "comparisons": primary_regression},
        "t3_original_and_recovery_both_reported": True,
        "warnings": ["TRAINING_STOCHASTICITY_WARNING", "SOURCE_PROVENANCE_AUDIT_WARNING", "secondary_template_robustness_evaluation"],
        "blockers": [],
        "formal_grid_not_started": True,
    }
    write_json(OUT / "summary.json", summary)
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha_file(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")
    print(canonical({"summary_completed": True, "status": summary["status"], "signals": signals}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "primary", "secondary", "summary"), required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model")
    parser.add_argument("--template-id", choices=("TEMPLATE_1", "TEMPLATE_2"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "primary":
        materialize_primary()
    elif args.phase == "secondary":
        if args.task is None or args.model is None or args.template_id is None:
            raise RuntimeError("--secondary requires --task, --model, and --template-id")
        evaluate_slot(args.task, args.model, args.template_id)
    else:
        summarize()


if __name__ == "__main__":
    main()
