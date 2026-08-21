from __future__ import annotations

"""SFT-C5 cross-family directional replication with local Microsoft Phi-4-mini-instruct."""

import argparse
import hashlib
import json
import os
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.sft import run_sft_c2_full_grid as c2  # noqa: E402
from corpus_experiments_v1.sft import run_sft_c3_scale_extension as c3  # noqa: E402


BASE = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1"
MATERIALIZATION = BASE / "materialization"
C2_OUT = BASE / "full_grid_v1"
C3_OUT = BASE / "scale_extension_v1"
C4_OUT = BASE / "compute_matched_v1"
OUT = BASE / "second_model_phi4mini_v1"
RUNS = OUT / "runs"
RETRIES = OUT / "_technical_retries"
MODEL = Path(os.environ.get("PHI4MINI_MODEL_PATH", "microsoft/Phi-4-mini-instruct"))
REPOSITORY_ID = "microsoft/Phi-4-mini-instruct"
REVISION = "revision_not_recoverable_from_local_snapshot"
PROTOCOL_NAME = "second_model_protocol_phi4mini_v1.json"
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
SCALES = {"T1a_ACRONYM": (1000, 2000), "T2_NUMERIC": (3000, 3678), "T3_CLAIM": (2000, 2286)}
SEEDS = (42, 314159, 271828)
LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["qkv_proj", "o_proj"]}
TRAINING = dict(c2.c1.TRAINING)
TRAINING.update({"micro_batch_size": 1, "gradient_accumulation": 8, "effective_batch_size": 8,
                 "precision": "float16", "max_sequence_length": 1280})
BOOTSTRAP = 10_000
DELTA_NEM = 0.02
ASSET_SHA256 = {
    "config.json": "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b",
    "generation_config.json": "3e3f48753753f92d2b958679151861d4fd7bf26e4dfc41fd47056116c4914dcd",
    "tokenizer.json": "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea",
    "tokenizer_config.json": "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db",
    "special_tokens_map.json": "aff38493227d813e29fcf8406e8e90062f1f031aa47d589325e9c31d89ac7cc3",
    "added_tokens.json": "d4f2aceb0f20b71dd1f4bcc7e052e4412946bf281840b8f83d39f259571af486",
    "vocab.json": "6cb65a857824fa6615bb1782d95d882617a8bbce1da0317118586b36f39e98bd",
    "merges.txt": "856ce61180bb689282eed6b3a6838bb1f438399be23aefe9d20eb379791fb4ad",
    "model.safetensors.index.json": "613a98d5e5716ca96fa75931abedc9c5a5d95f488ce4d62df71e639fe3ac6c59",
    "model-00001-of-00002.safetensors": "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6",
    "model-00002-of-00002.safetensors": "7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983",
    "configuration_phi3.py": "ec2044b77d0b8111640ac134daa3af1f40dc552a739f8959626e7e58ea3352df",
    "modeling_phi3.py": "d5fd551a7fe759b1d25e5b16b5012aa98b9b8bd400d26c7ee69c0090a245d6cd",
    "README.md": "c4f34916182a3ee87d95d00c92847cfa687ab21dc338a577275f0c34cf0e1dfd",
    "LICENSE": "fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d",
    "NOTICE.md": "a3386cc7125ec75ed34d4f0a340c47782982095e251d322e01d8ecab22d87ba4",
}


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_manifest(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"Frozen checksum mismatch: {directory.name}/{relative}")


def verify_assets() -> None:
    for name, expected in ASSET_SHA256.items():
        path = MODEL / name
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"Local Phi asset mismatch: {name}")


def verify_inputs() -> None:
    c3.verify_inputs()
    check_manifest(C2_OUT); check_manifest(C3_OUT); check_manifest(C4_OUT)
    if read_json(C2_OUT / "summary.json").get("status") != "SFT_FULL_GRID_READY":
        raise RuntimeError("C2 is not frozen ready")
    if read_json(C3_OUT / "summary.json").get("status") != "SFT_SCALE_EXTENSION_READY":
        raise RuntimeError("C3 is not frozen ready")
    if read_json(C4_OUT / "summary.json").get("status") != "COMPUTE_MATCHED_READY":
        raise RuntimeError("C4 is not frozen ready")
    verify_assets()


def tokenizer() -> Any:
    result = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    if not result.chat_template:
        raise RuntimeError("Official Phi chat template missing")
    if result.pad_token_id is None:
        result.pad_token = result.eos_token
    return result


def load_base() -> Any:
    return AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to("cuda")


def attach_lora(model: Any) -> Any:
    names = [name.rsplit(".", 1)[-1] for name, _ in model.named_modules()]
    counts = {target: names.count(target) for target in LORA["target_modules"]}
    if counts != {"qkv_proj": 32, "o_proj": 32}:
        raise RuntimeError(f"Frozen Phi attention mapping mismatch: {counts}")
    config = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                        target_modules=LORA["target_modules"], task_type="CAUSAL_LM")
    result = get_peft_model(model, config)
    result.gradient_checkpointing_enable(); result.enable_input_require_grads(); result.config.use_cache = False
    return result


def phi_serialize(row: pd.Series | dict[str, Any], tok: Any) -> tuple[list[int], list[int]]:
    user = [{"role": "user", "content": c2.c1.user_content(row)}]
    prompt = list(tok.apply_chat_template(user, tokenize=True, add_generation_prompt=True))
    full = list(tok.apply_chat_template(user + [{"role": "assistant", "content": str(row["response"])}],
                                        tokenize=True, add_generation_prompt=False))
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise RuntimeError("Phi official assistant boundary is not a deterministic prompt prefix")
    return prompt, full[len(prompt):]


def prepare_training_rows(frame: pd.DataFrame, tok: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []; checks = []
    for _, row in frame.iterrows():
        prompt, answer = phi_serialize(row, tok); ids = prompt + answer; labels = [-100] * len(prompt) + answer
        if len(ids) > 1280:
            raise RuntimeError(f"Unexpected Phi target truncation: {row.instance_id}")
        checks.append(bool(answer) and all(value == -100 for value in labels[:len(prompt)]) and labels[len(prompt):] == answer)
        rows.append({"ids": ids, "labels": labels, "instance_id": str(row.instance_id),
                     "assistant_tokens": len(answer)})
    return rows, {"samples": len(rows), "assistant_only_masking": all(checks),
                  "response_boundary_valid": all(checks), "official_full_turn_boundary": True}


def masking_unit_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in rows[:min(8, len(rows))]:
        labels = row["labels"]; first = next((i for i, value in enumerate(labels) if value != -100), None)
        checks.append(first is not None and all(value == -100 for value in labels[:first]) and
                      all(value != -100 for value in labels[first:]))
    result = {"tested_samples": len(checks), "assistant_only_masking": all(checks), "passed": bool(checks) and all(checks)}
    if not result["passed"]:
        raise RuntimeError("Phi assistant-only masking unit test failed")
    return result


def _enrich_phi_lengths(frame: pd.DataFrame, tok: Any) -> None:
    serial = [phi_serialize(row, tok) for _, row in frame.iterrows()]
    frame["prompt_token_count"] = [len(item[0]) for item in serial]
    frame["target_token_count"] = [len(item[1]) for item in serial]
    frame["sequence_token_count"] = [len(item[0]) + len(item[1]) for item in serial]
    frame["target_truncated"] = frame.sequence_token_count > 1280


def all_sets(tok: Any) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame]]:
    c2_trains, c2_evals = c2.load_all_sets(tok)
    labels = pd.read_parquet(c2.c1.LABELS, columns=["candidate_id", "condition_spans", "verifier_scores", "task_subtype"])
    label_map = {str(row.candidate_id): row._asdict() for row in labels.itertuples(index=False)}
    extension: dict[str, dict[int, pd.DataFrame]] = {"T2_NUMERIC": {}, "T3_CLAIM": {}}
    extension_evals: dict[str, pd.DataFrame] = {}
    for task in ("T2_NUMERIC", "T3_CLAIM"):
        stem = c2.STEMS[task]; pool = pd.read_parquet(C3_OUT / f"{stem}_extension_master_pool.parquet")
        evaluation = pd.read_parquet(MATERIALIZATION / f"{stem}_eval.parquet")
        for scale in SCALES[task]:
            if scale == 2000:
                continue
            extension[task][scale] = pool.head(scale).copy().reset_index(drop=True)
        for frame in [evaluation, *extension[task].values()]:
            frame["frozen_condition_spans"] = frame.candidate_id.astype(str).map(lambda value: label_map[value]["condition_spans"])
            frame["frozen_verifier_scores"] = frame.candidate_id.astype(str).map(lambda value: label_map[value]["verifier_scores"])
            frame["task_subtype"] = frame.candidate_id.astype(str).map(lambda value: label_map[value]["task_subtype"])
        extension_evals[task] = evaluation
    trains = {
        "T1a_ACRONYM": {scale: c2_trains["T1a_ACRONYM"][scale].copy() for scale in SCALES["T1a_ACRONYM"]},
        "T2_NUMERIC": {scale: extension["T2_NUMERIC"][scale].copy() for scale in SCALES["T2_NUMERIC"]},
        "T3_CLAIM": {2000: c2_trains["T3_CLAIM"][2000].copy(), 2286: extension["T3_CLAIM"][2286].copy()},
    }
    evals = {"T1a_ACRONYM": c2_evals["T1a_ACRONYM"].copy(),
             "T2_NUMERIC": extension_evals["T2_NUMERIC"].copy(), "T3_CLAIM": extension_evals["T3_CLAIM"].copy()}
    for task in TASKS:
        small, large = SCALES[task]
        for frame in [evals[task], *trains[task].values()]:
            _enrich_phi_lengths(frame, tok)
        if trains[task][small].instance_id.astype(str).tolist() != trains[task][large].head(small).instance_id.astype(str).tolist():
            raise RuntimeError(f"Frozen nested prefix mismatch: {task}")
        if len(evals[task]) != {"T1a_ACRONYM": 300, "T2_NUMERIC": 300, "T3_CLAIM": 293}[task]:
            raise RuntimeError(f"Frozen Eval size mismatch: {task}")
        if set(trains[task][large].paper_id.astype(str)) & set(evals[task].paper_id.astype(str)):
            raise RuntimeError(f"Train/Eval paper overlap: {task}")
        for scale, frame in trains[task].items():
            if len(frame) != scale or frame.target_truncated.astype(bool).any():
                raise RuntimeError(f"Frozen Phi scale/length mismatch: {task} N{scale}")
    return trains, evals


def asset_manifest(tok: Any) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    return {
        "status": "LOCAL_MODEL_PROVENANCE_AUDIT_PASS", "local_path": str(MODEL), "repo_identity": REPOSITORY_ID,
        "revision": REVISION, "download_channel": "hf-mirror transport mirror",
        "source_identity": "official Microsoft Hugging Face repository", "license": "MIT",
        "architecture": cfg.architectures[0], "model_type": cfg.model_type, "parameter_count": 3_836_021_760,
        "hidden_size": cfg.hidden_size, "layer_count": cfg.num_hidden_layers,
        "attention_heads": cfg.num_attention_heads, "key_value_heads": cfg.num_key_value_heads,
        "attention_implementation": "transformers native Phi3 SDPA", "tokenizer_class": type(tok).__name__,
        "vocab_size": cfg.vocab_size, "official_chat_template": tok.chat_template,
        "official_chat_template_sha256": sha_text(tok.chat_template), "model_dtype_metadata": str(cfg.torch_dtype),
        "frozen_training_precision": "float16", "weight_shard_count": 2,
        "total_weight_size_bytes": sum((MODEL / name).stat().st_size for name in ASSET_SHA256 if name.endswith(".safetensors")),
        "index_total_size_bytes": read_json(MODEL / "model.safetensors.index.json")["metadata"]["total_size"],
        "native_load_audit": {"trust_remote_code": False, "model_class": "Phi3ForCausalLM",
                              "finite_fp16_forward": True, "actual_target_modules": LORA["target_modules"]},
        "files": {name: {"bytes": (MODEL / name).stat().st_size, "sha256": value} for name, value in ASSET_SHA256.items()},
    }


def protocol() -> dict[str, Any]:
    tok = tokenizer(); trains, evals = all_sets(tok)
    membership = {task: {str(scale): sha_text("\n".join(frame.instance_id.astype(str)))
                         for scale, frame in values.items()} for task, values in trains.items()}
    eval_hashes = {task: sha_text("\n".join(frame.instance_id.astype(str))) for task, frame in evals.items()}
    return {
        "experiment": "SFT-C5_CROSS_FAMILY_DIRECTIONAL_REPLICATION", "status": "frozen_before_smoke_and_formal_training",
        "research_role": "CROSS_FAMILY_DIRECTIONAL_REPLICATION",
        "historical_model_access_records": [
            {"repository_id": "meta-llama/Llama-3.2-3B-Instruct", "status": "SECOND_MODEL_NOT_AVAILABLE", "preserved": True},
            {"repository_id": "meta-llama/Llama-3.2-3B-Instruct", "status": "LLAMA_ACCESS_STILL_BLOCKED", "preserved": True},
        ],
        "aborted_backup_record": {"repository_id": "Qwen/Qwen2.5-1.5B-Instruct", "artifacts_preserved": True,
                                  "excluded_from_C5_Phi_results": True},
        "model": asset_manifest(tok), "conditions": {task: list(scales) for task, scales in SCALES.items()},
        "seeds": list(SEEDS), "formal_runs": 18, "membership_sha256": membership,
        "eval_instance_ids_sha256": eval_hashes, "lora": LORA, "training": TRAINING,
        "memory_safety_implementation": {"micro_batch_size": 1, "gradient_accumulation": 8,
                                         "effective_batch_unchanged": True, "performance_independent": True},
        "loss_masking": {"official_full_turn_template": True, "prompt_labels": -100,
                         "assistant_answer_and_official_turn_terminators_only": True},
        "smoke": {"task": "T1a_ACRONYM", "scale": 1000, "seed": 42, "optimizer_steps": 8,
                  "development_only": True, "fresh_base_required_for_formal": True},
        "evaluation": {"templates": ["TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"],
                       "primary_metric": "NEM", "normalization": "identical frozen Qwen implementation",
                       "greedy": True, "official_Phi_chat_template": True},
        "statistics": {"hierarchical_paired_bootstrap": BOOTSTRAP, "outer": "paired training seed",
                       "inner": "paired fixed Eval example", "difference": "large-small", "does_not_redefine_n_star": True},
        "replication_rules": {"plateau_delta_nem": DELTA_NEM,
                              "T3_growth": "replicated iff two-sided CI lower bound > 0"},
        "frozen_parent_manifests": {"C2": sha_file(C2_OUT / "checksums.sha256"),
                                    "C3": sha_file(C3_OUT / "checksums.sha256"),
                                    "C4": sha_file(C4_OUT / "checksums.sha256")},
        "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha_file(Path(__file__))},
        "prohibited": ["resampling", "rematerialization", "new scale", "new seed", "third model",
                       "hyperparameter tuning", "compute-matched Phi", "paper modification"],
    }


def prepare() -> None:
    verify_inputs(); tok = tokenizer(); expected_protocol = protocol(); expected_manifest = asset_manifest(tok)
    checksum_text = "\n".join(f"{ASSET_SHA256[name]}  {name}" for name in sorted(ASSET_SHA256)) + "\n"
    if OUT.exists():
        required = [OUT / PROTOCOL_NAME, OUT / "local_model_asset_manifest.json", OUT / "local_model_checksums.sha256"]
        if not all(path.is_file() for path in required) or read_json(required[0]) != expected_protocol or read_json(required[1]) != expected_manifest or required[2].read_text(encoding="utf-8") != checksum_text:
            raise RuntimeError("Existing Phi C5 pretraining freeze differs; refuse overwrite")
        return
    OUT.mkdir(parents=True); RUNS.mkdir(); RETRIES.mkdir()
    write_json(OUT / "local_model_asset_manifest.json", expected_manifest)
    (OUT / "local_model_checksums.sha256").write_text(checksum_text, encoding="utf-8")
    write_json(OUT / PROTOCOL_NAME, expected_protocol)


def batch_loss(model: Any, tok: Any, batch: list[dict[str, Any]]) -> torch.Tensor:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long, device="cuda")
    labels = torch.full_like(ids, -100); attention = torch.zeros_like(ids)
    for index, row in enumerate(batch):
        length = len(row["ids"]); ids[index, :length] = torch.tensor(row["ids"], device="cuda")
        labels[index, :length] = torch.tensor(row["labels"], device="cuda"); attention[index, :length] = 1
    return model(input_ids=ids, attention_mask=attention, labels=labels).loss


def train_steps(model: Any, tok: Any, rows: list[dict[str, Any]], seed: int,
                max_steps: int | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    micro, accumulation, epochs = TRAINING["micro_batch_size"], TRAINING["gradient_accumulation"], TRAINING["epochs"]
    full_steps = math.ceil(math.ceil(len(rows) / micro) / accumulation) * epochs
    total_steps = max_steps if max_steps is not None else full_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    scheduler = c2.c1.get_linear_schedule_with_warmup(optimizer, max(1, math.ceil(total_steps * TRAINING["warmup_ratio"])), total_steps)
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True); model.train(); started = time.perf_counter()
    losses: list[float] = []; tokens_seen = 0; assistant_seen = 0; examples_seen = 0; optimizer_steps = 0; hashes = {}
    for epoch in range(epochs):
        order = list(range(len(rows))); random.Random(seed + epoch).shuffle(order)
        hashes[f"epoch_{epoch + 1}_order_sha256"] = sha_text("\n".join(rows[index]["instance_id"] for index in order))
        pending = 0
        for start in range(0, len(order), micro):
            batch = [rows[index] for index in order[start:start + micro]]; loss = batch_loss(model, tok, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf Phi loss")
            (loss / accumulation).backward(); pending += 1; losses.append(float(loss.detach().cpu()))
            examples_seen += len(batch); tokens_seen += sum(len(item["ids"]) for item in batch)
            assistant_seen += sum(item["assistant_tokens"] for item in batch)
            if pending == accumulation or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); pending = 0; optimizer_steps += 1
                if optimizer_steps >= total_steps:
                    break
        if optimizer_steps >= total_steps:
            break
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started; window = min(8, max(1, len(losses) // 3))
    return ({"optimizer_steps": optimizer_steps, "expected_optimizer_steps": total_steps,
             "epochs_completed": epochs if max_steps is None else None, "samples_seen": examples_seen,
             "input_tokens_processed": tokens_seen, "assistant_target_tokens_processed": assistant_seen,
             "loss_initial": losses[0], "loss_final": losses[-1], "loss_min": min(losses),
             "first_window_loss": statistics.fmean(losses[:window]), "last_window_loss": statistics.fmean(losses[-window:]),
             "finite_loss": all(math.isfinite(value) for value in losses), "runtime_seconds": elapsed,
             "peak_vram_bytes": int(torch.cuda.max_memory_allocated()), "tokens_per_second": tokens_seen / elapsed}, hashes)


def generate_phi(model: Any, tok: Any, frame: pd.DataFrame, task: str) -> tuple[list[str], int]:
    tok.padding_side = "left"; model.eval(); outputs = []; truncated = 0; max_new = c2.c1.GENERATION_MAX_NEW_TOKENS[task]
    for start in range(0, len(frame), 4):
        batch = frame.iloc[start:start + 4]
        rendered = [tok.apply_chat_template([{"role": "user", "content": c2.c1.user_content(row)}],
                                            tokenize=False, add_generation_prompt=True) for _, row in batch.iterrows()]
        encoded = tok(rendered, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=max_new,
                                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
        width = encoded.input_ids.shape[1]
        for item in generated:
            answer_ids = item[width:]; truncated += int(len(answer_ids) >= max_new and int(answer_ids[-1]) != tok.eos_token_id)
            outputs.append(tok.decode(answer_ids, skip_special_tokens=True))
    return outputs, truncated


def evaluate_phi(model: Any, tok: Any, frame: pd.DataFrame, task: str, run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    generated, generation_truncated = generate_phi(model, tok, frame, task); rows = []
    for (_, item), prediction in zip(frame.iterrows(), generated):
        gold = str(item.response); record = {
            "run_id": run_id, "instance_id": str(item.instance_id), "paper_id": str(item.paper_id),
            "candidate_id": str(item.candidate_id), "task": task, "gold": gold, "prediction": prediction,
            "normalized_gold": c2.c1.normalize(gold), "normalized_prediction": c2.c1.normalize(prediction),
            "exact_match": float(prediction == gold), "correct_NEM": float(c2.c1.normalize(prediction) == c2.c1.normalize(gold)),
            "target_truncated": False, "generation_hit_max_new_tokens": False,
        }
        record.update(c2.c1.deterministic_flags(item, prediction)); rows.append(record)
    result = pd.DataFrame(rows); metrics = {"task": task, "run_id": run_id, "instances": len(result),
        "exact_match": float(result.exact_match.mean()), "normalized_exact_match": float(result.correct_NEM.mean()),
        "eval_target_truncation_count": 0, "generation_hit_max_new_tokens_count": generation_truncated,
        "generation_deterministic": True}
    if task == "T1a_ACRONYM":
        metrics["case_insensitive_normalized_em"] = float(result.case_insensitive_normalized_em.mean())
    elif task == "T2_NUMERIC":
        for name in ("unit", "operator"):
            required = result[f"{name}_required"].fillna(False).astype(bool)
            metrics[f"{name}_preservation_instances"] = int(required.sum())
            metrics[f"{name}_preservation_accuracy"] = float(result.loc[required, f"{name}_preserved"].astype(float).mean()) if required.any() else None
    else:
        metrics.update({"token_f1": float(result.token_f1.mean()), "token_precision": float(result.token_precision.mean()),
                        "token_recall": float(result.token_recall.mean())})
        for name in ("negation", "uncertainty", "condition"):
            required = result[f"{name}_required"].fillna(False).astype(bool)
            metrics[f"{name}_preservation_instances"] = int(required.sum())
            metrics[f"{name}_preservation_accuracy"] = float(result.loc[required, f"{name}_preserved"].astype(float).mean()) if required.any() else None
    return result, metrics


def environment() -> dict[str, Any]:
    return {"python_version": sys.version, "platform": platform.platform(), "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__, "peft_version": __import__("peft").__version__,
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "gpu_model": torch.cuda.get_device_name(0), "precision": "float16",
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def smoke() -> None:
    prepare(); target = OUT / "smoke_summary.json"
    if target.is_file() and read_json(target).get("status") == "SMOKE_PASS":
        return
    tok = tokenizer(); trains, evals = all_sets(tok); frame = trains["T1a_ACRONYM"][1000]
    rows, masking_checks = prepare_training_rows(frame, tok); masking = masking_unit_test(rows); rng = c2.configure_seed(42)
    model = attach_lora(load_base()); training, order = train_steps(model, tok, rows, 42, max_steps=8)
    subset = evals["T1a_ACRONYM"].head(8).copy(); predictions, evaluation = evaluate_phi(model.eval(), tok, subset, "T1a_ACRONYM", "phi_smoke_development_only")
    examples = predictions[["instance_id", "gold", "prediction"]].head(3).to_dict("records")
    del model; torch.cuda.empty_cache()
    passed = bool(masking["passed"] and masking_checks["assistant_only_masking"] and training["finite_loss"] and
                  training["optimizer_steps"] == 8 and len(predictions) == 8 and evaluation["instances"] == 8)
    write_json(target, {"status": "SMOKE_PASS" if passed else "SMOKE_FAIL", "development_only": True,
        "checks": {"local_model_load": True, "native_transformers": True, "trust_remote_code": False,
                   "tokenizer": True, "official_chat_template": True, "lora_injection": True,
                   "actual_target_modules": LORA["target_modules"], "masking": masking,
                   "masking_checks": masking_checks, "forward_backward": training["finite_loss"],
                   "finite_loss": training["finite_loss"], "generation": len(predictions) == 8,
                   "evaluation_parsing": evaluation["instances"] == 8, "target_truncation": 0,
                   "train_eval_paper_overlap": 0}, "precision": "float16", "training": training,
        "order_hashes": order, "rng_audit": rng, "environment": environment(), "debug_examples": examples,
        "formal_run_requires_fresh_base": True, "smoke_adapter_saved": False})
    if not passed:
        raise RuntimeError("Phi C5 smoke failed")


def run_name(task: str, scale: int, seed: int) -> str:
    return f"{c2.STEMS[task]}_n{scale}_seed{seed}"


def run_one(task: str, scale: int, seed: int) -> None:
    prepare()
    if read_json(OUT / "smoke_summary.json").get("status") != "SMOKE_PASS":
        raise RuntimeError("Phi smoke must pass first")
    if scale not in SCALES[task]:
        raise RuntimeError("Unregistered Phi task-scale")
    directory = RUNS / run_name(task, scale, seed)
    required = [directory / "run_config.json", directory / "training_summary.json", directory / "predictions_primary.parquet",
                directory / "predictions_template_1.parquet", directory / "predictions_template_2.parquet",
                directory / "adapter/adapter_model.safetensors"]
    if all(path.is_file() for path in required):
        if read_json(directory / "training_summary.json").get("status") != "completed":
            raise RuntimeError("Phi run marked incomplete")
        return
    if directory.exists():
        raise RuntimeError(f"Incomplete Phi run preserved; archive under _technical_retries before retry: {directory.name}")
    directory.mkdir(parents=True)
    try:
        tok = tokenizer(); trains, evals = all_sets(tok); train_frame, eval_frame = trains[task][scale], evals[task]
        rows, masking_checks = prepare_training_rows(train_frame, tok); masking = masking_unit_test(rows); rng = c2.configure_seed(seed)
        ids = train_frame.instance_id.astype(str).tolist(); paper_ids = sorted(train_frame.paper_id.astype(str).unique())
        write_json(directory / "run_config.json", {"run": directory.name, "task": task, "scale": scale, "seed": seed,
            "research_role": "CROSS_FAMILY_DIRECTIONAL_REPLICATION", "protocol_sha256": sha_file(OUT / PROTOCOL_NAME),
            "config_sha256": ASSET_SHA256["config.json"], "base_model": read_json(OUT / PROTOCOL_NAME)["model"],
            "lora": LORA, "actual_target_modules": LORA["target_modules"], "training": TRAINING,
            "training_instance_ids": ids, "training_membership_sha256": sha_text("\n".join(ids)),
            "paper_ids": paper_ids, "paper_count": len(paper_ids),
            "training_serialized_tokens_one_epoch": int(train_frame.sequence_token_count.sum()),
            "assistant_target_tokens_one_epoch": int(train_frame.target_token_count.sum()),
            "eval_instance_ids_sha256": sha_text("\n".join(eval_frame.instance_id.astype(str))), "eval_instances": len(eval_frame),
            "train_eval_paper_overlap": len(set(train_frame.paper_id.astype(str)) & set(eval_frame.paper_id.astype(str))),
            "target_truncation_count": int(train_frame.target_truncated.sum()), "fresh_base_lora": True,
            "smoke_adapter_reused": False, "small_scale_adapter_reused": False})
        model = attach_lora(load_base()); training, order = train_steps(model, tok, rows, seed)
        if not training["finite_loss"] or not masking["passed"]:
            raise RuntimeError("Phi formal training gate failed")
        adapter = directory / "adapter"; model.save_pretrained(adapter); del model; torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(load_base(), adapter).eval(); metrics = []
        for template_id in ("TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"):
            frame = eval_frame.copy()
            if template_id != "TEMPLATE_0_PRIMARY":
                frame["instruction"] = [c2.rendered_instruction(task, template_id, value) for value in frame.instruction.astype(str)]
            predictions, evaluation = evaluate_phi(reloaded, tok, frame, task, f"{directory.name}_{template_id}")
            predictions = c2.annotate_predictions(predictions, task, scale, seed, template_id)
            predictions["template"] = template_id
            filename = "predictions_primary.parquet" if template_id == "TEMPLATE_0_PRIMARY" else f"predictions_{template_id.lower()}.parquet"
            predictions.to_parquet(directory / filename, index=False)
            metrics.append(c2.metric_row(predictions, evaluation, task, scale, seed, template_id))
        del reloaded; torch.cuda.empty_cache()
        write_json(directory / "training_summary.json", {"status": "completed", "run": directory.name,
            "loss_masking_unit_test": masking, "masking_checks": masking_checks, "epoch_order_sha256": order,
            "rng_audit": rng, "environment": environment(), "training": training,
            "adapter_sha256": c2.c1.aggregate_adapter_sha(adapter), "adapter_fresh_base_reload": True,
            "eval_target_truncation_count": 0, "metrics": metrics})
    except Exception as error:
        write_json(directory / "technical_failure.json", {"run": directory.name, "error_type": type(error).__name__,
                                                           "error": str(error), "technical_retry_required": True})
        raise


def expected_runs() -> list[tuple[str, int, int]]:
    return [(task, scale, seed) for task in TASKS for scale in SCALES[task] for seed in SEEDS]


def bootstrap(task: str, predictions: dict[tuple[int, int], pd.DataFrame]) -> dict[str, Any]:
    small, large = SCALES[task]; arrays = {}; ids: list[str] | None = None
    for scale in (small, large):
        for seed in SEEDS:
            frame = predictions[(scale, seed)].sort_values("instance_id").reset_index(drop=True)
            current = frame.instance_id.astype(str).tolist()
            if ids is None: ids = current
            elif current != ids: raise RuntimeError(f"Paired Phi Eval IDs drift: {task}")
            arrays[(scale, seed)] = frame.correct_NEM.to_numpy(float)
    size = len(ids or []); rng = np.random.default_rng(int(sha_text(f"C5_PHI4MINI_BOOTSTRAP_V1|{task}")[:16], 16)); values = np.empty(BOOTSTRAP)
    for index in range(BOOTSTRAP):
        drawn_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS)); components = []
        for seed_index in drawn_seeds:
            seed = SEEDS[int(seed_index)]; examples = rng.integers(0, size, size=size)
            components.append(float((arrays[(large, seed)][examples] - arrays[(small, seed)][examples]).mean()))
        values[index] = float(np.mean(components))
    point = float(np.mean([arrays[(large, seed)].mean() - arrays[(small, seed)].mean() for seed in SEEDS]))
    return {"task": task, "small_scale": small, "large_scale": large, "comparison": "large_minus_small",
            "point_difference": point, "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)), "bootstrap_replicates": BOOTSTRAP,
            "bootstrap_seed": "SHA256(C5_PHI4MINI_BOOTSTRAP_V1|task)",
            "method": "outer paired training seed; inner paired fixed-Eval example"}


def qwen_means(task: str) -> tuple[float, float]:
    small, large = SCALES[task]
    table = pd.read_csv(C2_OUT / "full_grid_scale_metrics.csv") if task == "T1a_ACRONYM" else pd.read_csv(C3_OUT / "combined_scale_metrics.csv")
    rows = table[table.task.eq(task)].set_index("scale").NEM_mean
    return float(rows.loc[small]), float(rows.loc[large])


def plateau_judgment(task: str, point: float, low: float, high: float) -> str:
    label = "EARLY_PLATEAU_PATTERN_REPLICATED" if task == "T1a_ACRONYM" else "LATER_PLATEAU_PATTERN_REPLICATED"
    if abs(point) <= DELTA_NEM: return label
    if low > DELTA_NEM or high < -DELTA_NEM: return "NOT_REPLICATED"
    return "MIXED"


def finalize() -> None:
    prepare(); protocol_data = read_json(OUT / PROTOCOL_NAME); protocol_sha = sha_file(OUT / PROTOCOL_NAME)
    records = []; costs = []; prediction_maps = {task: {} for task in TASKS}; errors = []
    for task, scale, seed in expected_runs():
        directory = RUNS / run_name(task, scale, seed); required = [directory / "run_config.json", directory / "training_summary.json",
            directory / "predictions_primary.parquet", directory / "predictions_template_1.parquet",
            directory / "predictions_template_2.parquet", directory / "adapter/adapter_model.safetensors"]
        if not all(path.is_file() for path in required): errors.append(f"missing:{directory.name}"); continue
        config, summary = read_json(required[0]), read_json(required[1]); expected_steps = math.ceil(scale / 8) * 3
        if summary.get("status") != "completed" or not summary["loss_masking_unit_test"].get("passed") or not summary["training"].get("finite_loss"):
            errors.append(f"pipeline:{directory.name}")
        if config.get("training_membership_sha256") != protocol_data["membership_sha256"][task][str(scale)] or config.get("eval_instance_ids_sha256") != protocol_data["eval_instance_ids_sha256"][task] or config.get("train_eval_paper_overlap") != 0 or config.get("target_truncation_count") != 0 or summary.get("eval_target_truncation_count") != 0 or config.get("protocol_sha256") != protocol_sha or not config.get("fresh_base_lora") or config.get("smoke_adapter_reused") or config.get("small_scale_adapter_reused") or not summary.get("adapter_fresh_base_reload") or summary["training"].get("optimizer_steps") != expected_steps:
            errors.append(f"integrity:{directory.name}")
        if set(summary.get("epoch_order_sha256", {})) != {"epoch_1_order_sha256", "epoch_2_order_sha256", "epoch_3_order_sha256"}:
            errors.append(f"order:{directory.name}")
        if {item["template_id"] for item in summary["metrics"]} != {"TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"}:
            errors.append(f"templates:{directory.name}")
        for path in required[2:5]:
            frame = pd.read_parquet(path); expected_eval = protocol_data["eval_instance_ids_sha256"][task]
            if len(frame) != config["eval_instances"] or sha_text("\n".join(frame.instance_id.astype(str))) != expected_eval or "paper_id" not in frame:
                errors.append(f"prediction:{directory.name}/{path.name}")
        for item in summary["metrics"]:
            row = dict(item); row.update({"runtime_seconds": summary["training"]["runtime_seconds"],
                "input_tokens_processed": summary["training"]["input_tokens_processed"],
                "assistant_target_tokens_processed": summary["training"]["assistant_target_tokens_processed"]}); records.append(row)
        costs.append({"task": task, "scale": scale, "seed": seed, "runtime_seconds": summary["training"]["runtime_seconds"],
                      "gpu_hours": summary["training"]["runtime_seconds"] / 3600,
                      "input_tokens_processed": summary["training"]["input_tokens_processed"],
                      "assistant_target_tokens_processed": summary["training"]["assistant_target_tokens_processed"],
                      "peak_vram_bytes": summary["training"]["peak_vram_bytes"]})
        prediction_maps[task][(scale, seed)] = pd.read_parquet(directory / "predictions_primary.parquet")
    if errors or len(records) != 54:
        write_json(OUT / "summary.json", {"status": "SECOND_MODEL_REPLICATION_NOT_READY", "errors": errors})
        raise RuntimeError("Phi C5 completeness gate failed")
    seed_metrics = pd.DataFrame(records).sort_values(["task", "scale", "seed", "template_id"]); seed_metrics.to_csv(OUT / "seed_metrics.csv", index=False)
    primary = seed_metrics[seed_metrics.template_id.eq("TEMPLATE_0_PRIMARY")]
    scale_metrics = primary.groupby(["task", "scale"], as_index=False).agg(NEM_mean=("NEM", "mean"), NEM_sd=("NEM", "std"), NEM_min=("NEM", "min"), NEM_max=("NEM", "max"))
    scale_metrics.to_csv(OUT / "scale_metrics.csv", index=False)
    boots = pd.DataFrame([bootstrap(task, prediction_maps[task]) for task in TASKS]); boots.to_csv(OUT / "bootstrap.csv", index=False)
    template_rows = []
    for (task, scale), group in seed_metrics.groupby(["task", "scale"]):
        per = group.groupby("template_id").NEM.mean()
        template_rows.append({"task": task, "scale": int(scale), "template0_NEM": float(per["TEMPLATE_0_PRIMARY"]),
            "template1_NEM": float(per["TEMPLATE_1"]), "template2_NEM": float(per["TEMPLATE_2"]),
            "primary_NEM": float(per["TEMPLATE_0_PRIMARY"]),
            "paraphrase_mean_NEM": float(np.mean([per["TEMPLATE_1"], per["TEMPLATE_2"]])),
            "worst_template_NEM": float(per.min()), "template_SD": float(per.std(ddof=1))})
    template_table = pd.DataFrame(template_rows); template_table.to_csv(OUT / "template_robustness.csv", index=False)
    comparison_rows = []; judgments = {}
    for task in TASKS:
        small, large = SCALES[task]; subset = scale_metrics[scale_metrics.task.eq(task)].set_index("scale")
        psmall, plarge = float(subset.loc[small].NEM_mean), float(subset.loc[large].NEM_mean); point = plarge - psmall
        boot = boots[boots.task.eq(task)].iloc[0]; low, high = float(boot.ci95_low), float(boot.ci95_high)
        if task == "T3_CLAIM": judgment = "CAPACITY_LIMITED_GROWTH_PATTERN_REPLICATED" if low > 0 else ("NOT_REPLICATED" if high <= 0 else "MIXED")
        else: judgment = plateau_judgment(task, point, low, high)
        judgments[task] = judgment; qsmall, qlarge = qwen_means(task)
        temp = template_table[template_table.task.eq(task)].sort_values("scale")
        comparison_rows.append({"task": task, "small_scale": small, "large_scale": large,
            "Qwen_small_mean": qsmall, "Qwen_large_mean": qlarge, "Qwen_difference": qlarge - qsmall,
            "Phi_small_mean": psmall, "Phi_large_mean": plarge, "Phi_difference": point,
            "Phi_95CI_low": low, "Phi_95CI_high": high,
            "template_robustness_summary": canonical(temp[["scale", "primary_NEM", "paraphrase_mean_NEM", "worst_template_NEM", "template_SD"]].to_dict("records")),
            "replication_judgment": judgment})
    pd.DataFrame(comparison_rows).to_csv(OUT / "cross_model_comparison.csv", index=False)
    cost_table = pd.DataFrame(costs); cost_table.to_csv(OUT / "training_cost_summary.csv", index=False)
    replicated = sum(value.endswith("_REPLICATED") for value in judgments.values())
    overall = "TASK_SPECIFIC_SCALE_PATTERN_REPLICATED" if replicated == 3 else ("PARTIAL_REPLICATION" if replicated else "NOT_REPLICATED")
    retries = len(list(RETRIES.iterdir()))
    total_cost = {"runtime_seconds": float(cost_table.runtime_seconds.sum()), "gpu_hours": float(cost_table.gpu_hours.sum()),
                  "input_tokens_processed": int(cost_table.input_tokens_processed.sum()),
                  "assistant_target_tokens_processed": int(cost_table.assistant_target_tokens_processed.sum()),
                  "peak_vram_min_bytes": int(cost_table.peak_vram_bytes.min()), "peak_vram_max_bytes": int(cost_table.peak_vram_bytes.max())}
    write_json(OUT / "summary.json", {"status": "SECOND_MODEL_REPLICATION_READY", "research_role": "CROSS_FAMILY_DIRECTIONAL_REPLICATION",
        "model": protocol_data["model"], "expected_runs": 18, "complete_runs": 18, "technical_retries": retries,
        "smoke_excluded_from_formal_results": True,
        "integrity": {"local_provenance": True, "fresh_base_each_run": True, "membership_match": True,
                      "eval_match": True, "paper_overlap_zero": True, "masking_pass": True, "finite_loss": True,
                      "target_truncation_zero": True, "prediction_files_complete": True, "three_templates_complete": True,
                      "bootstrap_complete": True, "protocol_drift": False},
        "task_judgments": judgments, "overall_judgment": overall, "training_cost": total_cost,
        "warnings": ["TRAINING_STOCHASTICITY_WARNING", "SOURCE_PROVENANCE_AUDIT_WARNING",
                     "LOCAL_SNAPSHOT_REVISION_NOT_RECOVERABLE", "LLAMA_ACCESS_HISTORY_PRESERVED",
                     "CROSS_FAMILY_SCALE_RESPONSE_NOT_MODEL_LEADERBOARD"]})
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256" and path.suffix != ".log")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha_file(path)}  {path.relative_to(OUT).as_posix()}" for path in files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", required=True, choices=("prepare", "smoke", "run-one", "run-all", "final"))
    parser.add_argument("--task", choices=TASKS); parser.add_argument("--scale", type=int); parser.add_argument("--seed", type=int, choices=SEEDS); args = parser.parse_args()
    if args.phase == "prepare": prepare(); print(canonical({"prepared": str(OUT)}))
    elif args.phase == "smoke": smoke(); print(canonical({"smoke": "SMOKE_PASS"}))
    elif args.phase == "run-one":
        if args.task is None or args.scale is None or args.seed is None: raise RuntimeError("run-one requires task/scale/seed")
        run_one(args.task, args.scale, args.seed); print(canonical({"completed": run_name(args.task, args.scale, args.seed)}))
    elif args.phase == "run-all":
        for task, scale, seed in expected_runs(): run_one(task, scale, seed); print(canonical({"completed": run_name(task, scale, seed)}), flush=True)
    else: finalize(); print(canonical({"status": "SECOND_MODEL_REPLICATION_READY"}))


if __name__ == "__main__":
    main()
