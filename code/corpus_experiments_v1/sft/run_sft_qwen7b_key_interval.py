from __future__ import annotations

"""Qwen2.5-7B key-interval model-size sensitivity experiment."""

import argparse
import gc
import hashlib
import json
import os
import math
import os
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.sft import run_sft_c2_full_grid as c2  # noqa: E402
from corpus_experiments_v1.sft import run_sft_c3_scale_extension as c3  # noqa: E402
from corpus_experiments_v1.sft import run_sft_c5_within_family_replication as frozen_sets  # noqa: E402


BASE = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1"
MATERIALIZATION = BASE / "materialization"
C2_OUT = BASE / "full_grid_v1"
C3_OUT = BASE / "scale_extension_v1"
PHI_OUT = BASE / "second_model_phi4mini_v1"
OUT = BASE / "qwen7b_key_interval_v1"
RUNS = OUT / "runs"
MODEL = Path(os.environ.get("QWEN7B_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"))
REPOSITORY_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
SCALES = {
    "T1a_ACRONYM": (1000, 2000),
    "T2_NUMERIC": (3000, 3678),
    "T3_CLAIM": (2000, 2286),
}
SEEDS = (42, 314159, 271828)
LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
TRAINING = dict(c2.c1.TRAINING)
TRAINING["micro_batch_size"] = 1
TRAINING["gradient_accumulation"] = 8
TRAINING["effective_batch_size"] = 8
BOOTSTRAP = 10_000
DELTA_NEM = 0.02
CACHE_TRIM_EVERY_OPTIMIZER_STEPS = 8
PROTOCOL_NAME = "qwen7b_key_interval_protocol.json"
MODEL_FILES = {
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ordered_hash(values: pd.Series | list[str]) -> str:
    return sha_text("\n".join(str(value) for value in values))


def set_hash(values: pd.Series | list[str]) -> str:
    return sha_text("\n".join(sorted(set(str(value) for value in values))))


def check_manifest(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"Frozen checksum mismatch: {directory.name}/{relative}")


def parent_run(task: str, scale: int, seed: int) -> Path:
    use_c2 = task == "T1a_ACRONYM" or (task == "T3_CLAIM" and scale == 2000)
    return (C2_OUT if use_c2 else C3_OUT) / "runs" / f"{c2.STEMS[task]}_n{scale}_seed{seed}"


def tokenizer() -> Any:
    result = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    result.pad_token = result.eos_token
    if not result.chat_template:
        raise RuntimeError("Official Qwen tokenizer chat template missing")
    return result


def all_sets(tok: Any) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame]]:
    trains, evals = frozen_sets.all_sets(tok)
    return trains, evals


def verify_model_assets() -> None:
    if not MODEL.is_dir():
        raise RuntimeError(f"Official local model directory missing: {MODEL}")
    for name, expected in MODEL_FILES.items():
        path = MODEL / name
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"Official Qwen2.5-7B asset mismatch: {name}")


def frozen_membership(tok: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trains, evals = all_sets(tok)
    membership: dict[str, Any] = {}
    eval_info: dict[str, Any] = {}
    parent_configs: dict[str, Any] = {}
    for task in TASKS:
        small, large = SCALES[task]
        if trains[task][small].instance_id.astype(str).tolist() != trains[task][large].head(small).instance_id.astype(str).tolist():
            raise RuntimeError(f"Nested training prefix mismatch: {task}")
        membership[task] = {}
        for scale in (small, large):
            frame = trains[task][scale]
            instance_hash = ordered_hash(frame.instance_id.astype(str).tolist())
            seed_hashes = []
            for seed in SEEDS:
                config_path = parent_run(task, scale, seed) / "run_config.json"
                config = read_json(config_path)
                if config.get("training_membership_sha256") != instance_hash:
                    raise RuntimeError(f"Parent membership drift: {task} N{scale} seed{seed}")
                seed_hashes.append(sha_file(config_path))
            membership[task][str(scale)] = {
                "instances": int(len(frame)),
                "unique_papers": int(frame.paper_id.astype(str).nunique()),
                "train_instance_ids_sha256": instance_hash,
                "train_paper_ids_sha256": set_hash(frame.paper_id.astype(str).tolist()),
                "target_truncation_count": int(frame.target_truncated.astype(bool).sum()),
            }
            parent_configs[f"{task}|{scale}"] = seed_hashes
        evaluation = evals[task]
        eval_hash = ordered_hash(evaluation.instance_id.astype(str).tolist())
        for scale in (small, large):
            for seed in SEEDS:
                config = read_json(parent_run(task, scale, seed) / "run_config.json")
                if config.get("eval_instance_ids_sha256") != eval_hash:
                    raise RuntimeError(f"Parent Eval drift: {task} N{scale} seed{seed}")
        overlap = len(set(trains[task][large].paper_id.astype(str)) & set(evaluation.paper_id.astype(str)))
        if overlap:
            raise RuntimeError(f"Train/Eval paper overlap: {task}")
        eval_info[task] = {
            "instances": int(len(evaluation)),
            "unique_papers": int(evaluation.paper_id.astype(str).nunique()),
            "eval_instance_ids_sha256": eval_hash,
            "eval_paper_ids_sha256": set_hash(evaluation.paper_id.astype(str).tolist()),
            "train_eval_paper_overlap": overlap,
        }
    return membership, eval_info, parent_configs


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
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def protocol() -> dict[str, Any]:
    tok = tokenizer()
    membership, eval_info, parent_configs = frozen_membership(tok)
    config = read_json(MODEL / "config.json")
    return {
        "experiment": "QWEN25_7B_KEY_INTERVAL_MODEL_SCALE_SENSITIVITY",
        "status": "frozen_before_smoke_and_formal_training",
        "research_role": "within_family_parameter_scale_sensitivity_key_interval_replication",
        "not_full_scaling_curve": True,
        "does_not_redefine_n_star": True,
        "model": {
            "repository_id": REPOSITORY_ID,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "local_path": str(MODEL),
            "official_huggingface_repository": True,
            "downloaded_once_then_local_only": True,
            "local_files_only": True,
            "license": "Apache-2.0",
            "architecture": config["architectures"][0],
            "model_type": config["model_type"],
            "parameter_count": 7_615_616_512,
            "layer_count": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "attention_heads": config["num_attention_heads"],
            "key_value_heads": config["num_key_value_heads"],
            "tokenizer_class": tok.__class__.__name__,
            "official_chat_template_sha256": sha_text(str(tok.chat_template)),
            "dtype": "float16",
            "quantization": "none",
            "attention_implementation": "sdpa",
            "asset_sha256": MODEL_FILES,
        },
        "conditions": {task: list(scales) for task, scales in SCALES.items()},
        "seeds": list(SEEDS),
        "formal_runs": 18,
        "membership": membership,
        "evaluation_membership": eval_info,
        "parent_run_config_sha256_by_seed": parent_configs,
        "lora": LORA,
        "training": TRAINING,
        "memory_implementation": {
            "base_fp16_peak_allocated_bytes_load_test": 15_276_322_304,
            "micro_batch_size_changed_from_3B": [2, 1],
            "gradient_accumulation_changed_from_3B": [4, 8],
            "effective_batch_size_unchanged": 8,
            "eval_batch_size_same_as_3B": 4,
            "gradient_checkpointing": True,
            "reason": "7B FP16 base allocation plus WDDM resident memory leaves insufficient safe margin for micro-batch 2; a pre-formal runner probe exposed and corrected a stale training-model reference before any formal result was committed",
            "scientific_recipe_changed": False,
            "pre_formal_resource_probe_archives": [
                str(BASE / "qwen7b_key_interval_v1_pre_eval_memory_repair_20260819"),
                str(BASE / "qwen7b_key_interval_v1_pre_release_bugfix_20260819"),
            ],
        },
        "smoke": {
            "task": "T1a_ACRONYM",
            "scale": 1000,
            "seed": 42,
            "optimizer_steps": 8,
            "development_only": True,
            "adapter_retained": False,
            "formal_fresh_base_required": True,
        },
        "evaluation": {
            "template": "TEMPLATE_0_PRIMARY",
            "primary_metric": "NEM",
            "normalization": "identical frozen Qwen2.5-3B RQ3 evaluator",
            "eval_instances": {task: eval_info[task]["instances"] for task in TASKS},
            "greedy": True,
            "eval_batch_size": 4,
            "generation_max_new_tokens": c2.c1.GENERATION_MAX_NEW_TOKENS,
        },
        "statistics": {
            "hierarchical_bootstrap": BOOTSTRAP,
            "outer": "training seed",
            "inner": "paired frozen Eval example",
            "difference": "large_minus_small",
            "practical_equivalence_delta_nem": DELTA_NEM,
            "criterion": "one-sided UCB95(large-small) <= delta",
        },
        "frozen_parent_manifests": {
            "C2_checksums_sha256": sha_file(C2_OUT / "checksums.sha256"),
            "C3_checksums_sha256": sha_file(C3_OUT / "checksums.sha256"),
            "Phi_checksums_sha256": sha_file(PHI_OUT / "checksums.sha256"),
        },
        "environment": environment(),
        "prohibited": [
            "resampling or rematerialization",
            "new tasks or scales",
            "modification of Qwen2.5-3B or Phi results",
            "quantization or QLoRA",
            "performance-based tuning",
            "paper modification",
        ],
    }


def prepare() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    expected = protocol()
    if OUT.exists():
        target = OUT / PROTOCOL_NAME
        if not target.is_file() or read_json(target) != expected:
            raise RuntimeError("Existing Qwen7B protocol differs; refuse overwrite")
        return
    c3.verify_inputs()
    check_manifest(C3_OUT)
    check_manifest(PHI_OUT)
    verify_model_assets()
    OUT.mkdir(parents=True)
    RUNS.mkdir()
    write_json(OUT / PROTOCOL_NAME, expected)


def load_base() -> Any:
    return AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")


def attach_lora(model: Any) -> Any:
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(LORA["target_modules"]) - leaf_names
    if missing:
        raise RuntimeError(f"Frozen LoRA target modules absent: {sorted(missing)}")
    config = LoraConfig(
        r=LORA["r"],
        lora_alpha=LORA["alpha"],
        lora_dropout=LORA["dropout"],
        target_modules=LORA["target_modules"],
        task_type="CAUSAL_LM",
    )
    result = get_peft_model(model, config)
    result.gradient_checkpointing_enable()
    result.enable_input_require_grads()
    result.config.use_cache = False
    return result


def train_steps(
    model: Any,
    tok: Any,
    rows: list[dict[str, Any]],
    seed: int,
    max_steps: int | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    micro = int(TRAINING["micro_batch_size"])
    accumulation = int(TRAINING["gradient_accumulation"])
    epochs = int(TRAINING["epochs"])
    steps_per_epoch = math.ceil(math.ceil(len(rows) / micro) / accumulation)
    formal_steps = steps_per_epoch * epochs
    total_steps = max_steps if max_steps is not None else formal_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    scheduler = c2.c1.get_linear_schedule_with_warmup(
        optimizer,
        max(1, math.ceil(total_steps * TRAINING["warmup_ratio"])),
        total_steps,
    )
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.perf_counter()
    losses: list[float] = []
    tokens_seen = 0
    examples_seen = 0
    optimizer_steps = 0
    order_hashes: dict[str, str] = {}
    epochs_started = 0
    for epoch in range(epochs):
        epochs_started += 1
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        order_hashes[f"epoch_{epoch + 1}_order_sha256"] = sha_text(
            "\n".join(rows[index]["instance_id"] for index in order)
        )
        pending = 0
        for start in range(0, len(order), micro):
            batch = [rows[index] for index in order[start : start + micro]]
            loss = c2.c1.batch_loss(model, tok, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf training loss")
            (loss / accumulation).backward()
            pending += 1
            losses.append(float(loss.detach().cpu()))
            examples_seen += len(batch)
            tokens_seen += sum(len(item["ids"]) for item in batch)
            if pending == accumulation or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                optimizer_steps += 1
                # Windows/WDDM retained nearly all 24 GB after repeated steps and the
                # host later bugchecked.  This only releases *unused* allocator cache;
                # it does not change the forward/backward graph, gradients, RNG, or
                # optimizer/scheduler sequence.
                if optimizer_steps % CACHE_TRIM_EVERY_OPTIMIZER_STEPS == 0:
                    torch.cuda.empty_cache()
                if optimizer_steps >= total_steps:
                    break
        if optimizer_steps >= total_steps:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    window = min(8, max(1, len(losses) // 3))
    return (
        {
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": total_steps,
            "formal_expected_optimizer_steps": formal_steps,
            "steps_per_epoch": steps_per_epoch,
            "epochs_started": epochs_started,
            "epochs_completed": epochs if max_steps is None else None,
            "examples_seen": examples_seen,
            "tokens_seen": tokens_seen,
            "loss_initial": losses[0],
            "loss_final": losses[-1],
            "loss_min": min(losses),
            "first_window_loss": statistics.fmean(losses[:window]),
            "last_window_loss": statistics.fmean(losses[-window:]),
            "finite_loss": all(math.isfinite(value) for value in losses),
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_free_before_training_bytes": int(free_before),
            "gpu_total_memory_bytes": int(total_memory),
            "tokens_per_second": tokens_seen / elapsed,
            "cuda_cache_trim_every_optimizer_steps": CACHE_TRIM_EVERY_OPTIMIZER_STEPS,
            "cuda_cache_trim_scientific_recipe_changed": False,
        },
        order_hashes,
    )


def smoke() -> None:
    prepare()
    target = OUT / "smoke_summary.json"
    if (
        target.is_file()
        and read_json(target).get("status") == "SMOKE_PASS"
        and read_json(target).get("checks", {}).get("fresh_base_adapter_reload") is True
    ):
        return
    tok = tokenizer()
    trains, evals = all_sets(tok)
    rows, masking_checks = c2.c1.prepare_training_rows(trains["T1a_ACRONYM"][1000], tok)
    masking = c2.c1.masking_unit_test(rows)
    rng_audit = c2.configure_seed(42)
    model = attach_lora(load_base())
    training, order = train_steps(model, tok, rows, 42, max_steps=8)
    with tempfile.TemporaryDirectory(prefix="qwen7b_smoke_adapter_", dir=OUT) as temporary:
        adapter = Path(temporary) / "adapter"
        model.save_pretrained(adapter)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(load_base(), adapter).eval()
        torch.cuda.reset_peak_memory_stats()
        predictions, evaluation = c2.c1.evaluate(
            reloaded,
            tok,
            evals["T1a_ACRONYM"].head(8).copy(),
            "T1a_ACRONYM",
            "qwen7b_smoke_development_only",
        )
        evaluation_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
    passed = bool(
        masking["passed"]
        and masking_checks["assistant_only_masking"]
        and training["finite_loss"]
        and training["optimizer_steps"] == 8
        and len(predictions) == 8
        and evaluation["instances"] == 8
    )
    write_json(
        target,
        {
            "status": "SMOKE_PASS" if passed else "SMOKE_FAIL",
            "development_only": True,
            "scientific_result": False,
            "adapter_retained": False,
            "checks": {
                "official_model_local_only_load": True,
                "official_chat_template": bool(tok.chat_template),
                "lora_targets": LORA["target_modules"],
                "assistant_only_masking": masking,
                "masking_checks": masking_checks,
                "forward_backward_finite": training["finite_loss"],
                "generation": len(predictions) == 8,
                "evaluation_parsing": evaluation["instances"] == 8,
                "fresh_base_adapter_reload": True,
            },
            "training": training,
            "evaluation_peak_gpu_memory_bytes": evaluation_peak_gpu_memory_bytes,
            "epoch_order_sha256": order,
            "rng_audit": rng_audit,
            "environment": environment(),
            "formal_run_requires_fresh_base": True,
        },
    )
    if not passed:
        raise RuntimeError("Qwen7B smoke failed")


def run_name(task: str, scale: int, seed: int) -> str:
    return f"{c2.STEMS[task]}_n{scale}_seed{seed}"


def expected_runs() -> list[tuple[str, int, int]]:
    return [(task, scale, seed) for task in TASKS for scale in SCALES[task] for seed in SEEDS]


def run_one(task: str, scale: int, seed: int) -> None:
    prepare()
    if not (OUT / "smoke_summary.json").is_file() or read_json(OUT / "smoke_summary.json").get("status") != "SMOKE_PASS":
        raise RuntimeError("Smoke PASS is required before formal training")
    if scale not in SCALES[task] or seed not in SEEDS:
        raise RuntimeError("Unregistered task/scale/seed")
    directory = RUNS / run_name(task, scale, seed)
    required = [
        directory / "run_config.json",
        directory / "training_summary.json",
        directory / "predictions_primary.parquet",
        directory / "adapter/adapter_model.safetensors",
    ]
    if all(path.is_file() for path in required):
        if read_json(directory / "training_summary.json").get("status") != "completed":
            raise RuntimeError(f"Completed files but invalid status: {directory.name}")
        return
    if directory.exists():
        raise RuntimeError(f"Incomplete formal run preserved; inspect before retry: {directory}")
    temporary = RUNS / f".tmp_{directory.name}"
    if temporary.exists():
        failure = temporary / "technical_failure.json"
        if failure.exists():
            raise RuntimeError(f"Recorded technical failure requires inspection: {failure}")
        resolved = temporary.resolve()
        if resolved.parent != RUNS.resolve() or not resolved.name.startswith(".tmp_"):
            raise RuntimeError(f"Unsafe temporary path: {resolved}")
        shutil.rmtree(resolved)
    temporary.mkdir()
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        tok = tokenizer()
        trains, evals = all_sets(tok)
        train_frame = trains[task][scale]
        eval_frame = evals[task]
        rows, masking_checks = c2.c1.prepare_training_rows(train_frame, tok)
        masking = c2.c1.masking_unit_test(rows)
        rng_audit = c2.configure_seed(seed)
        model = attach_lora(load_base())
        training, order = train_steps(model, tok, rows, seed)
        if not training["finite_loss"] or not masking["passed"]:
            raise RuntimeError("Formal training gate failed")
        adapter = temporary / "adapter"
        model.save_pretrained(adapter)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(load_base(), adapter).eval()
        torch.cuda.reset_peak_memory_stats()
        predictions, evaluation = c2.c1.evaluate(
            reloaded, tok, eval_frame.copy(), task, f"{directory.name}_TEMPLATE_0_PRIMARY"
        )
        evaluation_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
        predictions = c2.annotate_predictions(predictions, task, scale, seed, "TEMPLATE_0_PRIMARY")
        predictions.to_parquet(temporary / "predictions_primary.parquet", index=False)
        metric = c2.metric_row(predictions, evaluation, task, scale, seed, "TEMPLATE_0_PRIMARY")
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        ids = train_frame.instance_id.astype(str).tolist()
        papers = train_frame.paper_id.astype(str).tolist()
        eval_ids = eval_frame.instance_id.astype(str).tolist()
        eval_papers = eval_frame.paper_id.astype(str).tolist()
        write_json(
            temporary / "run_config.json",
            {
                "run": directory.name,
                "task": task,
                "scale": scale,
                "seed": seed,
                "protocol_sha256": sha_file(OUT / PROTOCOL_NAME),
                "base_model": read_json(OUT / PROTOCOL_NAME)["model"],
                "lora": LORA,
                "training": TRAINING,
                "training_instance_ids": ids,
                "training_instance_ids_sha256": ordered_hash(ids),
                "training_paper_ids_sha256": set_hash(papers),
                "eval_instance_ids_sha256": ordered_hash(eval_ids),
                "eval_paper_ids_sha256": set_hash(eval_papers),
                "eval_instances": len(eval_frame),
                "train_eval_paper_overlap": len(set(papers) & set(eval_papers)),
                "target_truncation_count": int(train_frame.target_truncated.astype(bool).sum()),
                "fresh_base_lora": True,
                "smoke_adapter_reused": False,
                "formal_started_at": started_wall,
                "formal_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        write_json(
            temporary / "training_summary.json",
            {
                "status": "completed",
                "run": directory.name,
                "loss_masking_unit_test": masking,
                "masking_checks": masking_checks,
                "epoch_order_sha256": order,
                "rng_audit": rng_audit,
                "environment": environment(),
                "training": training,
                "adapter_sha256": c2.c1.aggregate_adapter_sha(adapter),
                "adapter_fresh_base_reload": True,
                "eval_target_truncation_count": evaluation["eval_target_truncation_count"],
                "evaluation_peak_gpu_memory_bytes": evaluation_peak_gpu_memory_bytes,
                "metrics": [metric],
            },
        )
        temporary.replace(directory)
    except Exception as error:
        write_json(
            temporary / "technical_failure.json",
            {
                "run": directory.name,
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_result": False,
            },
        )
        raise


def bootstrap_task(task: str, predictions: dict[tuple[int, int], pd.DataFrame]) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    small, large = SCALES[task]
    arrays: dict[tuple[int, int], np.ndarray] = {}
    ids: list[str] | None = None
    for scale in (small, large):
        for seed in SEEDS:
            frame = predictions[(scale, seed)].sort_values("instance_id").reset_index(drop=True)
            current = frame.instance_id.astype(str).tolist()
            if ids is None:
                ids = current
            elif current != ids:
                raise RuntimeError(f"Paired Eval IDs drift: {task}")
            arrays[(scale, seed)] = frame.correct_NEM.to_numpy(float)
    size = len(ids or [])
    rng = np.random.default_rng(int(sha_text(f"QWEN7B_KEY_INTERVAL_BOOTSTRAP_V1|{task}")[:16], 16))
    differences = np.empty(BOOTSTRAP)
    scale_values = {small: np.empty(BOOTSTRAP), large: np.empty(BOOTSTRAP)}
    for index in range(BOOTSTRAP):
        drawn_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
        components = {small: [], large: []}
        paired_differences = []
        for seed_index in drawn_seeds:
            seed = SEEDS[int(seed_index)]
            example_indices = rng.integers(0, size, size=size)
            small_value = float(arrays[(small, seed)][example_indices].mean())
            large_value = float(arrays[(large, seed)][example_indices].mean())
            components[small].append(small_value)
            components[large].append(large_value)
            paired_differences.append(large_value - small_value)
        scale_values[small][index] = float(np.mean(components[small]))
        scale_values[large][index] = float(np.mean(components[large]))
        differences[index] = float(np.mean(paired_differences))
    point = float(np.mean([arrays[(large, seed)].mean() - arrays[(small, seed)].mean() for seed in SEEDS]))
    ucb95 = float(np.quantile(differences, 0.95))
    comparison = {
        "task": task,
        "small_scale": small,
        "large_scale": large,
        "comparison": "large_minus_small",
        "point_difference": point,
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "one_sided_ucb95": ucb95,
        "delta_nem": DELTA_NEM,
        "practically_equivalent": bool(ucb95 <= DELTA_NEM),
        "bootstrap_replicates": BOOTSTRAP,
        "bootstrap_seed": "SHA256(QWEN7B_KEY_INTERVAL_BOOTSTRAP_V1|task)",
        "method": "outer resampling of three training seeds; inner paired resampling of frozen Eval examples",
    }
    intervals = {
        scale: {
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for scale, values in scale_values.items()
    }
    return comparison, intervals


def frozen_3b_row(task: str) -> dict[str, Any]:
    small, large = SCALES[task]
    if task == "T1a_ACRONYM":
        metrics = pd.read_csv(C2_OUT / "full_grid_scale_metrics.csv")
        boot = pd.read_csv(C2_OUT / "practical_equivalence.csv")
        selected = boot[(boot.task == task) & (boot.scale == small)].iloc[0]
        difference = float(selected.point_N2000_minus_N)
    else:
        metrics = pd.read_csv(C3_OUT / "combined_scale_metrics.csv")
        boot = pd.read_csv(C3_OUT / "practical_equivalence.csv")
        selected = boot[(boot.task == task) & (boot.scale == small) & (boot.reference_scale == large)].iloc[0]
        difference = float(selected.mean_NEM_difference_reference_minus_scale)
    values = metrics[metrics.task == task].set_index("scale")
    return {
        "task": task,
        "model": "Qwen2.5-3B-Instruct",
        "research_dimension": "within_family_parameter_scale",
        "N_small": small,
        "N_large": large,
        "score_small": float(values.loc[small].NEM_mean),
        "score_large": float(values.loc[large].NEM_mean),
        "difference_large_minus_small": difference,
        "ci95_low": float(selected.ci95_low),
        "ci95_high": float(selected.ci95_high),
        "one_sided_ucb95": float(selected.one_sided_ucb95),
        "delta": DELTA_NEM,
        "equivalent": bool(selected.practically_equivalent),
        "source": "frozen_C2_C3",
    }


def phi_relation_rows() -> list[dict[str, Any]]:
    metrics = pd.read_csv(PHI_OUT / "scale_metrics.csv")
    boots = pd.read_csv(PHI_OUT / "bootstrap.csv")
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        small, large = SCALES[task]
        values = metrics[metrics.task == task].set_index("scale")
        selected = boots[boots.task == task].iloc[0]
        rows.append(
            {
                "task": task,
                "model": "Phi-4-mini-instruct",
                "research_dimension": "cross_family_directional_replication_frozen",
                "N_small": small,
                "N_large": large,
                "score_small": float(values.loc[small].NEM_mean),
                "score_large": float(values.loc[large].NEM_mean),
                "difference_large_minus_small": float(selected.point_difference),
                "ci95_low": float(selected.ci95_low),
                "ci95_high": float(selected.ci95_high),
                "one_sided_ucb95": np.nan,
                "delta": DELTA_NEM,
                "equivalent": np.nan,
                "source": "frozen_Phi_C5_directional_result_no_new_equivalence_test",
            }
        )
    return rows


def finalize() -> None:
    prepare()
    protocol_data = read_json(OUT / PROTOCOL_NAME)
    protocol_sha = sha_file(OUT / PROTOCOL_NAME)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    prediction_maps: dict[str, dict[tuple[int, int], pd.DataFrame]] = {task: {} for task in TASKS}
    for task, scale, seed in expected_runs():
        directory = RUNS / run_name(task, scale, seed)
        required = [
            directory / "run_config.json",
            directory / "training_summary.json",
            directory / "predictions_primary.parquet",
            directory / "adapter/adapter_model.safetensors",
        ]
        if not all(path.is_file() for path in required):
            errors.append(f"missing:{directory.name}")
            continue
        config = read_json(directory / "run_config.json")
        summary = read_json(directory / "training_summary.json")
        expected_train = protocol_data["membership"][task][str(scale)]
        expected_eval = protocol_data["evaluation_membership"][task]
        expected_steps = math.ceil(scale / TRAINING["micro_batch_size"] / TRAINING["gradient_accumulation"]) * TRAINING["epochs"]
        if summary.get("status") != "completed" or not summary["loss_masking_unit_test"].get("passed") or not summary["training"].get("finite_loss"):
            errors.append(f"pipeline:{directory.name}")
        checks = [
            config.get("protocol_sha256") == protocol_sha,
            config.get("training_instance_ids_sha256") == expected_train["train_instance_ids_sha256"],
            config.get("training_paper_ids_sha256") == expected_train["train_paper_ids_sha256"],
            config.get("eval_instance_ids_sha256") == expected_eval["eval_instance_ids_sha256"],
            config.get("eval_paper_ids_sha256") == expected_eval["eval_paper_ids_sha256"],
            config.get("train_eval_paper_overlap") == 0,
            config.get("target_truncation_count") == 0,
            config.get("fresh_base_lora") is True,
            config.get("smoke_adapter_reused") is False,
            summary.get("adapter_fresh_base_reload") is True,
            summary.get("eval_target_truncation_count") == 0,
            summary["training"].get("optimizer_steps") == expected_steps,
            set(summary.get("epoch_order_sha256", {})) == {
                "epoch_1_order_sha256",
                "epoch_2_order_sha256",
                "epoch_3_order_sha256",
            },
        ]
        if not all(checks):
            errors.append(f"integrity:{directory.name}")
        predictions = pd.read_parquet(directory / "predictions_primary.parquet")
        if len(predictions) != expected_eval["instances"] or ordered_hash(predictions.instance_id.astype(str).tolist()) != expected_eval["eval_instance_ids_sha256"]:
            errors.append(f"predictions:{directory.name}")
        prediction_maps[task][(scale, seed)] = predictions
        metric = dict(summary["metrics"][0])
        metric.update(
            {
                "model": "Qwen2.5-7B-Instruct",
                "runtime_seconds": summary["training"]["elapsed_seconds"],
                "training_tokens": summary["training"]["tokens_seen"],
                "optimizer_steps": summary["training"]["optimizer_steps"],
                "final_loss": summary["training"]["loss_final"],
                "peak_gpu_memory_bytes": summary["training"]["peak_gpu_memory_bytes"],
                "evaluation_peak_gpu_memory_bytes": summary["evaluation_peak_gpu_memory_bytes"],
                "adapter_sha256": summary["adapter_sha256"],
                "train_instance_ids_sha256": config["training_instance_ids_sha256"],
                "eval_instance_ids_sha256": config["eval_instance_ids_sha256"],
            }
        )
        records.append(metric)
    if errors or len(records) != 18:
        write_json(
            OUT / "summary.json",
            {
                "status": "QWEN7B_KEY_INTERVAL_NOT_READY",
                "expected_runs": 18,
                "complete_runs": len(records),
                "errors": errors,
            },
        )
        raise RuntimeError("Qwen7B completeness gate failed: " + ", ".join(errors[:8]))

    run_table = pd.DataFrame(records).sort_values(["task", "scale", "seed"])
    run_table.to_csv(OUT / "qwen7b_key_interval_runs.csv", index=False)
    boot_rows: list[dict[str, Any]] = []
    intervals: dict[tuple[str, int], dict[str, float]] = {}
    for task in TASKS:
        boot, task_intervals = bootstrap_task(task, prediction_maps[task])
        boot_rows.append(boot)
        for scale, values in task_intervals.items():
            intervals[(task, scale)] = values
    bootstrap = pd.DataFrame(boot_rows).sort_values("task")
    bootstrap.to_csv(OUT / "qwen7b_key_interval_bootstrap.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (task, scale), group in run_table.groupby(["task", "scale"], sort=True):
        interval = intervals[(task, int(scale))]
        summary_rows.append(
            {
                "task": task,
                "model": "Qwen2.5-7B-Instruct",
                "scale": int(scale),
                "NEM_mean": float(group.NEM.mean()),
                "NEM_sd": float(group.NEM.std(ddof=1)),
                "NEM_ci95_low": interval["ci95_low"],
                "NEM_ci95_high": interval["ci95_high"],
                "seed_values": canonical([float(value) for value in group.sort_values("seed").NEM]),
                "runtime_seconds_mean": float(group.runtime_seconds.mean()),
                "peak_gpu_memory_bytes_max": int(group.peak_gpu_memory_bytes.max()),
                "evaluation_peak_gpu_memory_bytes_max": int(group.evaluation_peak_gpu_memory_bytes.max()),
                "training_tokens_per_run": int(group.training_tokens.iloc[0]),
            }
        )
    scale_table = pd.DataFrame(summary_rows).sort_values(["task", "scale"])
    scale_table.to_csv(OUT / "qwen7b_key_interval_summary.csv", index=False)

    comparison_rows: list[dict[str, Any]] = []
    for task in TASKS:
        comparison_rows.append(frozen_3b_row(task))
        small, large = SCALES[task]
        values = scale_table[scale_table.task == task].set_index("scale")
        boot = bootstrap[bootstrap.task == task].iloc[0]
        comparison_rows.append(
            {
                "task": task,
                "model": "Qwen2.5-7B-Instruct",
                "research_dimension": "within_family_parameter_scale",
                "N_small": small,
                "N_large": large,
                "score_small": float(values.loc[small].NEM_mean),
                "score_large": float(values.loc[large].NEM_mean),
                "difference_large_minus_small": float(boot.point_difference),
                "ci95_low": float(boot.ci95_low),
                "ci95_high": float(boot.ci95_high),
                "one_sided_ucb95": float(boot.one_sided_ucb95),
                "delta": DELTA_NEM,
                "equivalent": bool(boot.practically_equivalent),
                "source": "qwen7b_key_interval_v1",
            }
        )
    comparison_rows.extend(phi_relation_rows())
    comparison = pd.DataFrame(comparison_rows).sort_values(["task", "research_dimension", "model"])
    comparison.to_csv(OUT / "cross_model_scale_comparison.csv", index=False)

    equivalence = {row.task: bool(row.practically_equivalent) for row in bootstrap.itertuples(index=False)}
    frozen_equivalence = {
        row["task"]: bool(row["equivalent"])
        for row in comparison_rows
        if row["model"] == "Qwen2.5-3B-Instruct"
    }
    relative_pattern_consistent = equivalence == frozen_equivalence
    small_scale_improvements = {}
    for task in TASKS:
        group = comparison[(comparison.task == task) & comparison.model.isin(["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"])].set_index("model")
        small_scale_improvements[task] = float(group.loc["Qwen2.5-7B-Instruct"].score_small - group.loc["Qwen2.5-3B-Instruct"].score_small)
    total_runtime = float(run_table.runtime_seconds.sum())
    write_json(
        OUT / "summary.json",
        {
            "status": "QWEN7B_KEY_INTERVAL_READY",
            "expected_runs": 18,
            "complete_runs": 18,
            "model_local_path": str(MODEL),
            "model_downloaded_once_for_experiment": True,
            "all_formal_loads_local_files_only": True,
            "smoke_pass": True,
            "smoke_excluded_from_scientific_results": True,
            "integrity": {
                "membership_match": True,
                "eval_match": True,
                "paper_overlap_zero": True,
                "masking_pass": True,
                "finite_loss": True,
                "target_truncation_zero": True,
                "predictions_complete": True,
                "fresh_base_and_lora_each_run": True,
                "bootstrap_complete": True,
                "protocol_drift": False,
            },
            "practical_equivalence": equivalence,
            "qwen3b_qwen7b_relative_equivalence_pattern_consistent": relative_pattern_consistent,
            "qwen7b_minus_qwen3b_small_scale_NEM": small_scale_improvements,
            "training_cost": {
                "runtime_seconds": total_runtime,
                "gpu_hours": total_runtime / 3600,
                "peak_gpu_memory_bytes_max": int(run_table.peak_gpu_memory_bytes.max()),
                "evaluation_peak_gpu_memory_bytes_max": int(run_table.evaluation_peak_gpu_memory_bytes.max()),
                "training_tokens_total": int(run_table.training_tokens.sum()),
            },
            "interpretation_constraints": [
                "key-interval replication only, not a full learning curve",
                "does not define a Qwen2.5-7B n_star or optimum scale",
                "Qwen-vs-Phi absolute performance is not a model leaderboard",
                "no changes to frozen Qwen2.5-3B or Phi results",
            ],
        },
    )
    files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and path.suffix.lower() != ".log"
    )
    (OUT / "checksums.sha256").write_text(
        "\n".join(f"{sha_file(path)}  {path.relative_to(OUT).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("prepare", "smoke", "run-one", "run-all", "final"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        print(canonical({"prepared": str(OUT)}))
    elif args.phase == "smoke":
        smoke()
        print(canonical({"smoke": "SMOKE_PASS"}))
    elif args.phase == "run-one":
        if args.task is None or args.scale is None or args.seed is None:
            raise RuntimeError("run-one requires task, scale, and seed")
        run_one(args.task, args.scale, args.seed)
        print(canonical({"completed": run_name(args.task, args.scale, args.seed)}))
    elif args.phase == "run-all":
        for task, scale, seed in expected_runs():
            run_one(task, scale, seed)
            print(canonical({"completed": run_name(task, scale, seed)}), flush=True)
    else:
        finalize()
        print(canonical({"status": "QWEN7B_KEY_INTERVAL_READY"}))


if __name__ == "__main__":
    main()
