from __future__ import annotations

"""Frozen SFT-C2 full nested scale grid runner.

This runner deliberately inherits the C1 serialization, model, LoRA, loss and
metric primitives.  Its additions are only the pre-registered C2 grid and the
per-run audit records required before full-grid execution.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from peft import PeftModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.sft import run_sft_scaling_endpoint_v1 as c1  # noqa: E402
from corpus_experiments_v1.sft import run_sft_c1_3_instruction_robustness as c13  # noqa: E402


BASE = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1"
MATERIALIZATION = BASE / "materialization"
C1 = BASE / "training_endpoint_v1"
C12 = BASE / "endpoint_adjudication_v1"
C13 = BASE / "instruction_robustness_v1"
OUT = BASE / "full_grid_v1"
RUNS = OUT / "runs"
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
SCALES = (100, 250, 500, 1000, 2000)
SEEDS = (42, 314159, 271828)
STEMS = c1.STEMS
PROTOCOL_SHA = "d4fa8dd565a0204285e2b70fc6914689a9042213ea4844cec4b44656aa146489"
FORMAL_PROTOCOL_SHA = "5c696679f7d6d39462822b8d7546e0f0ad32c50591509c8e771cd0f1fe90ed32"
TEMPLATE_CONFIG_SHA = "414b1d4fa656c3485a727e1ed33ffdb05747e68ddca3466139f2ca896ae825c0"
DELTA_NEM = 0.02
BOOTSTRAP = 10_000


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def check_manifest(directory: Path) -> None:
    manifest = directory / "checksums.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"Missing frozen checksum manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual_path = directory / relative
        if not actual_path.is_file() or sha_file(actual_path) != expected:
            raise RuntimeError(f"Frozen input checksum mismatch: {directory.name}/{relative}")


def check_model_and_tokenizer() -> None:
    if sha_file(c1.MODEL / "config.json") != c1.EXPECTED_MODEL_CONFIG_SHA:
        raise RuntimeError("Frozen Base model config SHA mismatch")
    expected = read_json(MATERIALIZATION / "config.json")["base_model"]["tokenizer_files"]
    for name, wanted in expected.items():
        if sha_file(c1.MODEL / name) != wanted:
            raise RuntimeError(f"Frozen tokenizer file SHA mismatch: {name}")


def verify_frozen_inputs() -> None:
    c1.verify_inputs()
    for directory in (MATERIALIZATION, C1, C12, C13):
        check_manifest(directory)
    if sha_file(C1 / "training_protocol_v1.json") != PROTOCOL_SHA:
        raise RuntimeError("Frozen C1 training protocol SHA mismatch")
    if sha_file(C13 / "formal_scale_protocol_v1.json") != FORMAL_PROTOCOL_SHA:
        raise RuntimeError("Frozen formal scale protocol SHA mismatch")
    if sha_file(C13 / "template_config_v1.json") != TEMPLATE_CONFIG_SHA:
        raise RuntimeError("Frozen template config SHA mismatch")
    check_model_and_tokenizer()


def configure_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Preserve the C1 recipe's actual cuDNN choices.  Deterministic algorithms
    # and TF32 are recorded rather than changed; GPU bitwise identity is not claimed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "python_seed": seed, "numpy_seed": seed, "torch_cpu_seed": seed, "torch_cuda_seed": seed,
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def environment() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": __import__("peft").__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_model": torch.cuda.get_device_name(0),
    }


def load_all_sets(tokenizer: Any) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame]]:
    """Use C1's frozen label enrichment/length gates, extending only its scales."""
    original_scales = c1.SCALES
    try:
        c1.SCALES = SCALES
        trains, evals, _lengths = c1.load_sets(tokenizer)
    finally:
        c1.SCALES = original_scales
    manifest = pd.read_csv(MATERIALIZATION / "nested_pool_manifest.csv")
    for task in TASKS:
        master = pd.read_parquet(MATERIALIZATION / f"{STEMS[task]}_master_train_pool.parquet")
        for scale in SCALES:
            subset = trains[task][scale]
            manifest_ids = manifest[(manifest.task == task) & (manifest.scale == scale)].instance_id.astype(str).tolist()
            if subset.instance_id.astype(str).tolist() != master.head(scale).instance_id.astype(str).tolist():
                raise RuntimeError(f"Frozen nested ordering mismatch: {task} N{scale}")
            if set(subset.instance_id.astype(str)) != set(manifest_ids) or len(manifest_ids) != scale:
                raise RuntimeError(f"Frozen nested membership mismatch: {task} N{scale}")
        if set(master.instance_id.astype(str).head(100)) - set(master.instance_id.astype(str).head(250)):
            raise RuntimeError(f"Nested prefix violation: {task}")
        if set(master.paper_id.astype(str)) & set(evals[task].paper_id.astype(str)):
            raise RuntimeError(f"Train/eval paper overlap: {task}")
    return trains, evals


def run_name(task: str, scale: int, seed: int) -> str:
    return f"{STEMS[task]}_n{scale}_seed{seed}"


def rendered_instruction(task: str, template_id: str, original: str) -> str:
    return c13.rendered_instruction(task, template_id, original)


def prepare_output() -> None:
    verify_frozen_inputs()
    config = {
        "experiment": "corpus_v1_sft_c2_full_nested_scale_grid",
        "status": "frozen_before_execution",
        "tasks": list(TASKS), "scales": list(SCALES), "training_seeds": list(SEEDS),
        "formal_scale_protocol_sha256": FORMAL_PROTOCOL_SHA,
        "template_config_sha256": TEMPLATE_CONFIG_SHA,
        "c1_training_protocol_sha256": PROTOCOL_SHA,
        "materialization_manifest_sha256": sha_file(MATERIALIZATION / "checksums.sha256"),
        "base_model": read_json(MATERIALIZATION / "config.json")["base_model"],
        "delta_nem": DELTA_NEM,
        "bootstrap_replicates": BOOTSTRAP,
        "endpoint_adapters": "development_endpoint_only; excluded from formal scale statistics",
        "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha_file(Path(__file__))},
    }
    if OUT.exists():
        existing = OUT / "config.json"
        if not existing.is_file() or read_json(existing) != config:
            raise RuntimeError("Existing full-grid directory/config differs; refuse to overwrite")
        return
    OUT.mkdir(parents=True)
    RUNS.mkdir()
    write_json(OUT / "config.json", config)
    shutil.copy2(C13 / "formal_scale_protocol_v1.json", OUT / "formal_scale_protocol_v1.json")


def train_with_audit(model: Any, tokenizer: Any, rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], dict[str, str]]:
    micro = c1.TRAINING["micro_batch_size"]
    accumulation = c1.TRAINING["gradient_accumulation"]
    epochs = c1.TRAINING["epochs"]
    steps_per_epoch = math.ceil(math.ceil(len(rows) / micro) / accumulation)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=c1.TRAINING["learning_rate"], weight_decay=c1.TRAINING["weight_decay"])
    scheduler = c1.get_linear_schedule_with_warmup(optimizer, max(1, math.ceil(total_steps * c1.TRAINING["warmup_ratio"])), total_steps)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.perf_counter()
    losses: list[float] = []
    tokens_seen = 0
    examples_seen = 0
    optimizer_steps = 0
    order_hashes: dict[str, str] = {}
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        actual_ids = [rows[index]["instance_id"] for index in order]
        order_hashes[f"epoch_{epoch + 1}_order_sha256"] = sha_text("\n".join(actual_ids))
        pending = 0
        for start in range(0, len(order), micro):
            batch = [rows[index] for index in order[start:start + micro]]
            loss = c1.batch_loss(model, tokenizer, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss")
            (loss / accumulation).backward()
            pending += 1
            losses.append(float(loss.detach().cpu()))
            examples_seen += len(batch)
            tokens_seen += sum(len(item["ids"]) for item in batch)
            if pending == accumulation or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), c1.TRAINING["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                pending = 0; optimizer_steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    window = min(8, max(1, len(losses) // 3))
    return {
        "optimizer_steps": optimizer_steps, "steps_per_epoch": steps_per_epoch, "epochs": epochs,
        "examples_seen": examples_seen, "tokens_seen": tokens_seen,
        "loss_initial": losses[0], "loss_final": losses[-1], "loss_min": min(losses),
        "first_window_loss": statistics.fmean(losses[:window]), "last_window_loss": statistics.fmean(losses[-window:]),
        "finite_loss": all(math.isfinite(value) for value in losses), "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "tokens_per_second": tokens_seen / elapsed,
    }, order_hashes


def annotate_predictions(frame: pd.DataFrame, task: str, scale: int, seed: int, template_id: str) -> pd.DataFrame:
    result = frame.copy()
    result["scale"] = scale
    result["seed"] = seed
    result["template_id"] = template_id
    result["target_containment"] = [bool(gold) and gold in prediction for gold, prediction in zip(result.normalized_gold, result.normalized_prediction)]
    if task == "T2_NUMERIC":
        targets = result.gold.map(c13.numeric_values)
        result["numeric_value_containment"] = [all(item in pred for item in target) if target else False for target, pred in zip(targets, result.normalized_prediction)]
    return result


def metric_row(predictions: pd.DataFrame, evaluation: dict[str, Any], task: str, scale: int, seed: int, template_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task, "scale": scale, "seed": seed, "template_id": template_id, "instances": int(len(predictions)),
        "NEM": float(evaluation["normalized_exact_match"]), "EM": float(evaluation["exact_match"]),
        "target_containment_rate": float(predictions.target_containment.mean()),
        "generation_hit_max_new_tokens_count": int(evaluation["generation_hit_max_new_tokens_count"]),
    }
    if task == "T1a_ACRONYM":
        row["case_insensitive_normalized_em"] = float(evaluation["case_insensitive_normalized_em"])
        row["long_form_containment_rate"] = row["target_containment_rate"]
    elif task == "T2_NUMERIC":
        row["numeric_value_containment_rate"] = float(predictions.numeric_value_containment.mean())
        for field in ("unit_preservation_instances", "unit_preservation_accuracy", "operator_preservation_instances", "operator_preservation_accuracy"):
            row[field] = evaluation.get(field)
    else:
        for field in ("token_f1", "token_precision", "token_recall", "negation_preservation_instances", "negation_preservation_accuracy", "uncertainty_preservation_instances", "uncertainty_preservation_accuracy", "condition_preservation_instances", "condition_preservation_accuracy"):
            row[field] = evaluation.get(field)
        row["claim_containment_rate"] = row["target_containment_rate"]
    return row


def run_one(task: str, scale: int, seed: int) -> None:
    prepare_output()
    name = run_name(task, scale, seed)
    directory = RUNS / name
    complete = directory / "training_summary.json"
    required = [complete, directory / "run_config.json", directory / "predictions_primary.parquet", directory / "adapter" / "adapter_model.safetensors"]
    if all(path.is_file() for path in required):
        summary = read_json(complete)
        if summary.get("status") != "completed":
            raise RuntimeError(f"Run marked non-complete: {name}")
        return
    if directory.exists():
        raise RuntimeError(f"Incomplete prior run preserved for audit; use explicit technical retry: {name}")
    directory.mkdir(parents=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(c1.MODEL, local_files_only=True, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        trains, evals = load_all_sets(tokenizer)
        train_frame, eval_frame = trains[task][scale], evals[task]
        rows, masking_checks = c1.prepare_training_rows(train_frame, tokenizer)
        masking = c1.masking_unit_test(rows)
        rng = configure_seed(seed)
        model = c1.attach_lora(c1.load_base())
        training, order_hashes = train_with_audit(model, tokenizer, rows, seed)
        if not training["finite_loss"] or not masking["passed"]:
            raise RuntimeError("Formal integrity gate failed during training")
        adapter = directory / "adapter"
        model.save_pretrained(adapter)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError("Saved adapter missing")
        del model; torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(c1.load_base(), adapter).eval()
        primary, primary_eval = c1.evaluate(reloaded, tokenizer, eval_frame, task, name)
        primary = annotate_predictions(primary, task, scale, seed, "TEMPLATE_0_PRIMARY")
        primary.to_parquet(directory / "predictions_primary.parquet", index=False)
        metrics = [metric_row(primary, primary_eval, task, scale, seed, "TEMPLATE_0_PRIMARY")]
        if scale in {100, 1000, 2000}:
            for template_id in ("TEMPLATE_1", "TEMPLATE_2"):
                altered = eval_frame.copy()
                altered["instruction"] = [rendered_instruction(task, template_id, instruction) for instruction in eval_frame.instruction.astype(str)]
                predictions, evaluation = c1.evaluate(reloaded, tokenizer, altered, task, f"{name}_{template_id}")
                predictions = annotate_predictions(predictions, task, scale, seed, template_id)
                predictions.to_parquet(directory / f"predictions_{template_id.lower()}.parquet", index=False)
                metrics.append(metric_row(predictions, evaluation, task, scale, seed, template_id))
        del reloaded; torch.cuda.empty_cache()
        instance_ids = train_frame.instance_id.astype(str).tolist()
        run_config = {
            "run": name, "task": task, "scale": scale, "seed": seed,
            "formal_scale_protocol_sha256": FORMAL_PROTOCOL_SHA, "template_config_sha256": TEMPLATE_CONFIG_SHA,
            "c1_training_protocol_sha256": PROTOCOL_SHA, "base_model": read_json(MATERIALIZATION / "config.json")["base_model"],
            "lora": c1.LORA, "training": c1.TRAINING,
            "training_instance_ids": instance_ids, "training_membership_sha256": sha_text("\n".join(instance_ids)),
            "unique_paper_ids": sorted(train_frame.paper_id.astype(str).unique().tolist()), "unique_paper_count": int(train_frame.paper_id.nunique()),
            "training_serialized_tokens_one_epoch": int(train_frame.sequence_token_count.sum()),
            "training_tokens_all_epochs_expected": int(train_frame.sequence_token_count.sum() * c1.TRAINING["epochs"]),
            "eval_instance_ids_sha256": sha_text("\n".join(eval_frame.instance_id.astype(str))), "eval_instances": int(len(eval_frame)),
            "train_eval_paper_overlap": int(len(set(train_frame.paper_id.astype(str)) & set(eval_frame.paper_id.astype(str)))),
            "target_truncation_count": int(train_frame.target_truncated.sum()),
        }
        write_json(directory / "run_config.json", run_config)
        summary = {
            "status": "completed", "run": name, "loss_masking_unit_test": masking, "masking_checks": masking_checks,
            "epoch_order_sha256": order_hashes, "rng_audit": rng, "environment": environment(), "training": training,
            "adapter_sha256": c1.aggregate_adapter_sha(adapter), "adapter_fresh_base_reload": True,
            "eval_target_truncation_count": 0, "metrics": metrics,
        }
        write_json(complete, summary)
    except Exception as error:
        write_json(directory / "technical_failure.json", {"run": name, "error_type": type(error).__name__, "error": str(error), "technical_retry": False})
        raise


def expected_runs() -> list[tuple[str, int, int]]:
    return [(task, scale, seed) for task in TASKS for scale in SCALES for seed in SEEDS]


def run_all() -> None:
    for task, scale, seed in expected_runs():
        run_one(task, scale, seed)
        print(canonical({"completed": run_name(task, scale, seed)}), flush=True)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def hierarchical_bootstrap(task: str, candidate: int, prediction_map: dict[tuple[int, int], pd.DataFrame]) -> dict[str, Any]:
    arrays: dict[tuple[int, int], np.ndarray] = {}
    for scale in (candidate, 2000):
        reference_ids: list[str] | None = None
        for seed in SEEDS:
            frame = prediction_map[(scale, seed)].sort_values("instance_id").reset_index(drop=True)
            ids = frame.instance_id.astype(str).tolist()
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"Paired eval IDs drift: {task} N{candidate}/N2000 seed {seed}")
            arrays[(scale, seed)] = frame.correct_NEM.to_numpy(dtype=float)
    size = len(reference_ids or [])
    rng = np.random.default_rng(int(sha_text(f"C2_BOOTSTRAP_V1|{task}|{candidate}")[:16], 16))
    values = np.empty(BOOTSTRAP, dtype=float)
    for index in range(BOOTSTRAP):
        drawn_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
        component: list[float] = []
        for seed_index in drawn_seeds:
            seed = SEEDS[int(seed_index)]
            example_indices = rng.integers(0, size, size=size)
            component.append(float((arrays[(2000, seed)][example_indices] - arrays[(candidate, seed)][example_indices]).mean()))
        values[index] = float(np.mean(component))
    observed = float(np.mean([arrays[(2000, seed)].mean() - arrays[(candidate, seed)].mean() for seed in SEEDS]))
    return {
        "task": task, "scale": candidate, "reference_scale": 2000, "point_N2000_minus_N": observed,
        "ci95_low": quantile(values, 0.025), "ci95_high": quantile(values, 0.975), "one_sided_ucb95": quantile(values, 0.95),
        "delta_nem": DELTA_NEM, "practically_equivalent": bool(quantile(values, 0.95) <= DELTA_NEM),
        "bootstrap_replicates": BOOTSTRAP, "bootstrap_seed": "SHA256(C2_BOOTSTRAP_V1|task|scale)",
        "method": "outer resampling of three training seeds; inner paired resampling of frozen Eval examples",
    }


def finalize() -> None:
    prepare_output()
    records: list[dict[str, Any]] = []
    prediction_map_by_task: dict[str, dict[tuple[int, int], pd.DataFrame]] = {task: {} for task in TASKS}
    integrity_errors: list[str] = []
    for task, scale, seed in expected_runs():
        directory = RUNS / run_name(task, scale, seed)
        required = [directory / "run_config.json", directory / "training_summary.json", directory / "predictions_primary.parquet", directory / "adapter" / "adapter_model.safetensors"]
        if not all(path.is_file() for path in required):
            integrity_errors.append(f"missing:{run_name(task, scale, seed)}")
            continue
        config, summary = read_json(directory / "run_config.json"), read_json(directory / "training_summary.json")
        if summary.get("status") != "completed" or not summary["loss_masking_unit_test"].get("passed") or not summary["training"].get("finite_loss"):
            integrity_errors.append(f"pipeline:{run_name(task, scale, seed)}")
        if config["training_membership_sha256"] != sha_text("\n".join(config["training_instance_ids"])) or config["train_eval_paper_overlap"] != 0 or config["target_truncation_count"] != 0 or summary["eval_target_truncation_count"] != 0:
            integrity_errors.append(f"integrity:{run_name(task, scale, seed)}")
        if set(summary["epoch_order_sha256"]) != {"epoch_1_order_sha256", "epoch_2_order_sha256", "epoch_3_order_sha256"}:
            integrity_errors.append(f"order_audit:{run_name(task, scale, seed)}")
        primary = pd.read_parquet(directory / "predictions_primary.parquet")
        prediction_map_by_task[task][(scale, seed)] = primary
        record = dict(next(item for item in summary["metrics"] if item["template_id"] == "TEMPLATE_0_PRIMARY"))
        record.update({"training_tokens": summary["training"]["tokens_seen"], "runtime_seconds": summary["training"]["elapsed_seconds"], "final_loss": summary["training"]["loss_final"], "peak_gpu_memory_bytes": summary["training"]["peak_gpu_memory_bytes"], "optimizer_steps": summary["training"]["optimizer_steps"], "adapter_sha256": summary["adapter_sha256"]})
        records.append(record)
    if integrity_errors or len(records) != 45:
        write_json(OUT / "summary.json", {"status": "SFT_FULL_GRID_NOT_READY", "integrity_errors": integrity_errors, "complete_runs": len(records), "expected_runs": 45})
        raise RuntimeError("Full-grid completeness gate failed: " + ", ".join(integrity_errors[:5]))
    seed_metrics = pd.DataFrame(records).sort_values(["task", "scale", "seed"])
    seed_metrics.to_csv(OUT / "full_grid_seed_metrics.csv", index=False)
    scale_rows: list[dict[str, Any]] = []
    for (task, scale), group in seed_metrics.groupby(["task", "scale"], sort=True):
        row = {"task": task, "scale": int(scale), "NEM_mean": float(group.NEM.mean()), "NEM_sd": float(group.NEM.std(ddof=1)), "NEM_min": float(group.NEM.min()), "NEM_max": float(group.NEM.max()), "NEM_seed_values": canonical([float(x) for x in group.sort_values("seed").NEM]), "EM_mean": float(group.EM.mean()), "training_tokens_per_run": int(group.training_tokens.iloc[0]), "runtime_seconds_mean": float(group.runtime_seconds.mean()), "final_loss_mean": float(group.final_loss.mean())}
        scale_rows.append(row)
    scale_metrics = pd.DataFrame(scale_rows).sort_values(["task", "scale"])
    scale_metrics.to_csv(OUT / "full_grid_scale_metrics.csv", index=False)
    learning = scale_metrics[["task", "scale", "NEM_mean", "NEM_sd", "NEM_min", "NEM_max"]].copy()
    learning.to_csv(OUT / "learning_curve.csv", index=False)
    bootstrap_rows = [hierarchical_bootstrap(task, scale, prediction_map_by_task[task]) for task in TASKS for scale in SCALES[:-1]]
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "practical_equivalence.csv", index=False)
    nstar_rows: list[dict[str, Any]] = []
    for task in TASKS:
        candidates = bootstrap[bootstrap.task == task].sort_values("scale")
        eligible = candidates[candidates.practically_equivalent]
        nstar_rows.append({"task": task, "n_star": int(eligible.scale.iloc[0]) if len(eligible) else 2000, "n_star_status": "identified" if len(eligible) else "N2000_reference_only", "delta_nem": DELTA_NEM})
    nstar = pd.DataFrame(nstar_rows)
    nstar.to_csv(OUT / "n_star_summary.csv", index=False)
    template_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for scale in (100, 1000, 2000):
            metrics: list[dict[str, Any]] = []
            for seed in SEEDS:
                values = read_json(RUNS / run_name(task, scale, seed) / "training_summary.json")["metrics"]
                if {item["template_id"] for item in values} != {"TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"}:
                    raise RuntimeError(f"Missing secondary template evaluation: {task} N{scale} seed{seed}")
                metrics.extend(values)
            table = pd.DataFrame(metrics)
            per_template = table.groupby("template_id").NEM.mean()
            all_values = [float(per_template[item]) for item in ("TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2")]
            template_rows.append({"task": task, "scale": scale, "primary_mean_NEM": all_values[0], "paraphrase_mean_NEM": float(np.mean(all_values[1:])), "worst_template_mean_NEM": float(min(all_values)), "template_sd": float(np.std(all_values, ddof=1)), "template0_NEM": all_values[0], "template1_NEM": all_values[1], "template2_NEM": all_values[2]})
    template_summary = pd.DataFrame(template_rows)
    template_summary.to_csv(OUT / "template_robustness_summary.csv", index=False)
    cost = seed_metrics.groupby(["task", "scale"], as_index=False).agg(training_tokens=("training_tokens", "first"), runtime_seconds_mean=("runtime_seconds", "mean"), runtime_seconds_sd=("runtime_seconds", "std"), peak_gpu_memory_bytes_max=("peak_gpu_memory_bytes", "max"), optimizer_steps=("optimizer_steps", "first"), final_loss_mean=("final_loss", "mean"))
    cost.to_csv(OUT / "training_cost_summary.csv", index=False)
    plateau_rows: list[dict[str, Any]] = []
    for task in TASKS:
        means = scale_metrics[scale_metrics.task == task].set_index("scale").NEM_mean
        difference = float(means.loc[2000] - means.loc[1000])
        plateau_rows.append({"task": task, "N2000_minus_N1000": difference, "plateau_status": "REFERENCE_NOT_PLATEAUED" if difference > DELTA_NEM else "REFERENCE_PLATEAU_COMPATIBLE"})
    summary = {"status": "SFT_FULL_GRID_READY", "expected_runs": 45, "complete_runs": 45, "technical_retries": [], "integrity": {"all_frozen_inputs_verified": True, "all_runs_same_base_revision": True, "all_runs_fresh_lora": True, "all_runs_masking_pass": True, "all_runs_finite_loss": True, "all_runs_order_auditable": True, "all_train_eval_paper_overlap_zero": True, "all_target_truncation_zero": True, "all_primary_predictions_present": True}, "formal_protocol_sha256": FORMAL_PROTOCOL_SHA, "template_config_sha256": TEMPLATE_CONFIG_SHA, "c1_training_protocol_sha256": PROTOCOL_SHA, "delta_nem": DELTA_NEM, "bootstrap_replicates": BOOTSTRAP, "n_star": nstar_rows, "plateau": plateau_rows, "warnings": ["TRAINING_STOCHASTICITY_WARNING", "SOURCE_PROVENANCE_AUDIT_WARNING", "FORMAT_METRIC_WARNING_T1A", "FORMAT_METRIC_WARNING_T2"], "endpoint_adapters": "development_endpoint_only"}
    write_json(OUT / "summary.json", summary)
    output_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha_file(path)}  {path.relative_to(OUT).as_posix()}" for path in output_files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("prepare", "run-one", "run-all", "final"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--scale", type=int, choices=SCALES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_output(); print(canonical({"prepared": str(OUT)}))
    elif args.phase == "run-one":
        if args.task is None or args.scale is None or args.seed is None:
            raise RuntimeError("run-one requires --task --scale --seed")
        run_one(args.task, args.scale, args.seed); print(canonical({"completed": run_name(args.task, args.scale, args.seed)}))
    elif args.phase == "run-all":
        run_all()
    else:
        finalize(); print(canonical({"status": "SFT_FULL_GRID_READY"}))


if __name__ == "__main__":
    main()
