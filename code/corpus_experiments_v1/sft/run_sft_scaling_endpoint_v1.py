from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[3]
MATERIALIZATION = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1/materialization"
OUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1/training_endpoint_v1"
RUNS = OUT / "runs"
LABELS = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_b_holdout1900_v1_2/holdout_candidate_labels_v1_2.parquet"
MODEL = Path(os.environ.get("QWEN3B_MODEL_PATH", "<LOCAL_PATH_OMITTED>"))

TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
SCALES = (100, 2000)
SEED = 42
MAX_SEQUENCE_LENGTH = 1280
GENERATION_MAX_NEW_TOKENS = {"T1a_ACRONYM": 32, "T2_NUMERIC": 32, "T3_CLAIM": 160}
EXPECTED_MATERIALIZATION = {
    "config.json": "626374e2f1bcb7c8f96ae1a5cc390439568fd3184f0f00f4c2036b0cebfa4a44",
    "materialization_summary.csv": "e13248784a20f41f00c103ffd008e8ea699481dc07da411b5b3b8d789bb8b470",
    "materialized_instances.parquet": "76f8ec7366e0719a8cee3465f83f2ddde8258a4912655575f427a0a552348ae6",
    "nested_pool_manifest.csv": "c126a10bbb76fa37986deeecea433d66d7b406ef561992cf09fc5219ee2be4e1",
    "sft_eval400_manifest.csv": "2ef5bf053f056b3d65464f2938e6d831f46d91c4b067174c65c6eb0330e724cf",
    "sft_train1500_manifest.csv": "cf3d98c4b68c3a407117057052005b48a44ddf9de45b3eaa8b5eb38f38279abe",
    "summary.json": "82020bdc9035f56bda514b256e27a387298d012fabaef07bfd6d5c561b83d71f",
    "t1a_eval.parquet": "9bb2e42ec9cb2262266538488ce427636a03a4908cc8b75544d569afa8732be1",
    "t1a_master_train_pool.parquet": "f96a26e868a126915b62e057e5cf22dfad455b4463de09d57c222b438649f738",
    "t2_eval.parquet": "c045dff34d18d66fe39de2d1645c53b92cd1f274b7e38b1678bc474710418688",
    "t2_master_train_pool.parquet": "ad09714ae29e47c589888a0e00fb7434dcccb4328c932f5fed502deaa0692b63",
    "t3_eval.parquet": "74a4e70d15da23537ed122066a2faca0237b27e72855850b547f18db1b7ceb8f",
    "t3_master_train_pool.parquet": "5bf1438f539f13141efe48f5819add219d07172916b937df31af4e8d7f0a5e3a",
}
EXPECTED_LABELS_SHA = "d766b8b356a6c102255d0aed25697adca229c66c04d068c3da986b25f7502446"
EXPECTED_MODEL_CONFIG_SHA = "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b"
EXPECTED_TOKENIZER_SHA = "badd01fb58846dadbe2b7ce2a21a2e8e2e7257c87bf971f5689be8f4c87fa786"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
TRAINING = {
    "seed": SEED,
    "epochs": 3,
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "weight_decay": 0.0,
    "scheduler": "linear",
    "warmup_ratio": 0.03,
    "micro_batch_size": 2,
    "gradient_accumulation": 4,
    "effective_batch_size": 8,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "precision": "float16",
    "assistant_only_loss": True,
    "fixed_epoch_protocol": True,
}
STEMS = {"T1a_ACRONYM": "t1a", "T2_NUMERIC": "t2", "T3_CLAIM": "t3"}
WORD_RE = re.compile(r"\w+", re.UNICODE)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?:\s*[–—-]\s*[-+]?\d+(?:[.,]\d+)?)?")


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


def normalize(value: Any, casefold: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold() if casefold else text


def tokens(value: Any) -> list[str]:
    return WORD_RE.findall(normalize(value, casefold=True))


def token_f1(prediction: str, gold: str) -> tuple[float, float, float]:
    left, right = tokens(prediction), tokens(gold)
    if not left or not right:
        score = float(left == right)
        return score, score, score
    overlap = sum((Counter(left) & Counter(right)).values())
    precision, recall = overlap / len(left), overlap / len(right)
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def parse_json(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


def configure_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def verify_inputs() -> None:
    listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (MATERIALIZATION / "checksums.sha256").read_text(encoding="utf-8").splitlines()}
    if listed != EXPECTED_MATERIALIZATION:
        raise RuntimeError("Frozen materialization checksum manifest differs from C0 record")
    for name, expected in EXPECTED_MATERIALIZATION.items():
        if sha_file(MATERIALIZATION / name) != expected:
            raise RuntimeError(f"Frozen materialization hash mismatch: {name}")
    if sha_file(LABELS) != EXPECTED_LABELS_SHA:
        raise RuntimeError("Frozen v1.2 label hash mismatch")
    if not MODEL.is_dir() or sha_file(MODEL / "config.json") != EXPECTED_MODEL_CONFIG_SHA:
        raise RuntimeError("Frozen Qwen2.5-3B snapshot/config mismatch")


def user_content(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['instruction']}\n\nPassage:\n{row['context']}"


def serialize(row: pd.Series | dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_content(row)}], tokenize=True, add_generation_prompt=True)
    answer = tokenizer(str(row["response"]) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return list(prompt), list(answer)


def load_sets(tokenizer: Any) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    labels = pd.read_parquet(LABELS, columns=["candidate_id", "condition_spans", "verifier_scores", "task_subtype"])
    label_map = {str(row.candidate_id): row._asdict() for row in labels.itertuples(index=False)}
    trains: dict[str, dict[int, pd.DataFrame]] = {}
    evals: dict[str, pd.DataFrame] = {}
    length_info: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        stem = STEMS[task]
        master = pd.read_parquet(MATERIALIZATION / f"{stem}_master_train_pool.parquet")
        evaluation = pd.read_parquet(MATERIALIZATION / f"{stem}_eval.parquet")
        manifest = pd.read_csv(MATERIALIZATION / "nested_pool_manifest.csv")
        expected_eval = set(evaluation.instance_id.astype(str))
        if len(evaluation) != len(expected_eval):
            raise RuntimeError(f"Duplicate eval instance IDs: {task}")
        trains[task] = {}
        for scale in SCALES:
            subset = master.head(scale).copy().reset_index(drop=True)
            expected = set(manifest[(manifest.task == task) & (manifest.scale == scale)].instance_id.astype(str))
            if len(subset) != scale or set(subset.instance_id.astype(str)) != expected:
                raise RuntimeError(f"Nested membership drift: {task} N{scale}")
            if scale == 100 and not set(subset.instance_id.astype(str)) < set(master.head(2000).instance_id.astype(str)):
                raise RuntimeError(f"N100 is not a strict subset of N2000: {task}")
            trains[task][scale] = subset
        if set(master.paper_id.astype(str)) & set(evaluation.paper_id.astype(str)):
            raise RuntimeError(f"Train/eval paper leakage: {task}")
        for frame in [*trains[task].values(), evaluation]:
            frame["frozen_condition_spans"] = frame.candidate_id.astype(str).map(lambda x: label_map[x]["condition_spans"])
            frame["frozen_verifier_scores"] = frame.candidate_id.astype(str).map(lambda x: label_map[x]["verifier_scores"])
            frame["task_subtype"] = frame.candidate_id.astype(str).map(lambda x: label_map[x]["task_subtype"])
            serial = [serialize(row, tokenizer) for _, row in frame.iterrows()]
            frame["prompt_token_count"] = [len(x[0]) for x in serial]
            frame["target_token_count"] = [len(x[1]) for x in serial]
            frame["sequence_token_count"] = [len(x[0]) + len(x[1]) for x in serial]
            frame["target_truncated"] = frame.sequence_token_count > MAX_SEQUENCE_LENGTH
            if bool(frame.target_truncated.any()):
                raise RuntimeError(f"Target truncation under frozen C1 max sequence length: {task}")
        evals[task] = evaluation
        all_frames = [trains[task][2000], evaluation]
        seq = pd.concat(all_frames).sequence_token_count
        response = pd.concat(all_frames).target_token_count
        length_info[task] = {
            "sequence_max": int(seq.max()), "sequence_p995": float(seq.quantile(0.995)),
            "target_max": int(response.max()), "target_p995": float(response.quantile(0.995)),
            "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS[task],
        }
    return trains, evals, length_info


def initialize_protocol() -> dict[str, Any]:
    verify_inputs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    _trains, _evals, lengths = load_sets(tokenizer)
    protocol = {
        "experiment": "corpus_v1_sft_c1_baseline_endpoint_v1",
        "code": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha_file(Path(__file__))},
        "frozen_materialization": {"path": str(MATERIALIZATION.relative_to(ROOT)), "checksums": EXPECTED_MATERIALIZATION},
        "model": {
            "repository_id": "Qwen/Qwen2.5-3B-Instruct", "revision": REVISION, "snapshot": str(MODEL),
            "config_sha256": EXPECTED_MODEL_CONFIG_SHA, "tokenizer_bundle_sha256": EXPECTED_TOKENIZER_SHA,
            "dtype": "float16", "quantization": "none", "local_files_only": True,
        },
        "tasks": list(TASKS), "endpoint_scales": list(SCALES), "seed": SEED,
        "lora": LORA, "training": TRAINING,
        "loss_masking": {
            "chat_template": "official tokenizer.apply_chat_template",
            "user_content": "instruction + \\n\\nPassage:\\n + context",
            "assistant_target": "exact frozen response + EOS",
            "prompt_labels": -100, "assistant_labels": "token IDs", "unit_test_required_before_training": True,
        },
        "evaluation": {
            "primary_metric": "normalized_exact_match", "normalization": "Unicode NFKC; trim; collapse whitespace; case-sensitive",
            "generation": {"do_sample": False, "num_beams": 1, "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS},
            "eval_batch_size": 4, "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "target_truncation_allowed": False, "public_datasets_used": False,
        },
        "pre_output_length_freeze": {
            "max_sequence_length_rule": "max frozen serialized train/eval sequence plus deterministic headroom, rounded to 1280",
            "max_new_tokens_rule": "ceil(p99.5 frozen target tokens * 1.25) plus fixed margin; fixed before Base generation",
            "per_task": lengths,
        },
        "scale_protocol": {
            "type": "fixed_epoch_practical_SFT", "interpretation": "unique SFT instances increase under one shared training recipe; optimization tokens naturally increase",
            "future_n_star_delta_nem": 0.02,
            "n_star_not_computed_in_C1": True,
        },
        "frozen_length_distribution": lengths,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "training_protocol_v1.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        expected_without_code = dict(protocol); frozen_without_code = dict(existing)
        expected_without_code.pop("code", None); frozen_without_code.pop("code", None)
        if frozen_without_code != expected_without_code:
            raise RuntimeError("Existing C1 training protocol differs; refuse to alter a frozen protocol")
        # Later recovery/audit code may differ, but never mutates the protocol
        # that was frozen before Base generation.
        protocol = existing
    else:
        write_json(target, protocol)
    return protocol


def load_base() -> Any:
    return AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")


def attach_lora(model: Any) -> Any:
    config = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                        target_modules=LORA["target_modules"], task_type="CAUSAL_LM")
    result = get_peft_model(model, config)
    result.gradient_checkpointing_enable()
    result.enable_input_require_grads()
    result.config.use_cache = False
    return result


def prepare_training_rows(frame: pd.DataFrame, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks: list[bool] = []
    for _, row in frame.iterrows():
        prompt, answer = serialize(row, tokenizer)
        ids, labels = prompt + answer, [-100] * len(prompt) + answer
        if len(ids) > MAX_SEQUENCE_LENGTH:
            raise RuntimeError(f"Unexpected target truncation: {row.instance_id}")
        checks.append(bool(answer) and all(x == -100 for x in labels[:len(prompt)]) and labels[len(prompt):] == answer)
        rows.append({"ids": ids, "labels": labels, "instance_id": str(row.instance_id)})
    return rows, {"samples": len(rows), "assistant_only_masking": all(checks), "response_boundary_valid": all(bool(x["labels"]) for x in rows)}


def masking_unit_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in rows[: min(8, len(rows))]:
        labels = row["labels"]
        first_target = next((i for i, value in enumerate(labels) if value != -100), None)
        checks.append(first_target is not None and all(v == -100 for v in labels[:first_target]) and all(v != -100 for v in labels[first_target:]))
    result = {"tested_samples": len(checks), "assistant_only_masking": all(checks), "passed": bool(checks) and all(checks)}
    if not result["passed"]:
        raise RuntimeError("Assistant-only loss masking unit test failed")
    return result


def batch_loss(model: Any, tokenizer: Any, batch: list[dict[str, Any]]) -> torch.Tensor:
    width = max(len(x["ids"]) for x in batch)
    ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(batch):
        length = len(row["ids"])
        ids[index, :length] = torch.tensor(row["ids"], dtype=torch.long, device="cuda")
        labels[index, :length] = torch.tensor(row["labels"], dtype=torch.long, device="cuda")
        attention[index, :length] = 1
    return model(input_ids=ids, attention_mask=attention, labels=labels).loss


def train(model: Any, tokenizer: Any, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    micro, accumulation, epochs = TRAINING["micro_batch_size"], TRAINING["gradient_accumulation"], TRAINING["epochs"]
    steps_per_epoch = math.ceil(math.ceil(len(rows) / micro) / accumulation)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
    warmup = max(1, math.ceil(total_steps * TRAINING["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True); model.train(); started = time.perf_counter()
    losses: list[float] = []; examples_seen = 0; tokens_seen = 0; optimizer_steps = 0
    for epoch in range(epochs):
        order = list(range(len(rows))); random.Random(seed + epoch).shuffle(order); pending = 0
        for start in range(0, len(order), micro):
            batch = [rows[i] for i in order[start:start + micro]]
            loss = batch_loss(model, tokenizer, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss")
            (loss / accumulation).backward(); pending += 1
            losses.append(float(loss.detach().cpu())); examples_seen += len(batch); tokens_seen += sum(len(x["ids"]) for x in batch)
            if pending == accumulation or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); pending = 0; optimizer_steps += 1
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    window = min(8, max(1, len(losses) // 3))
    return {
        "optimizer_steps": optimizer_steps, "steps_per_epoch": steps_per_epoch, "epochs": epochs,
        "examples_seen": examples_seen, "tokens_seen": tokens_seen, "loss_initial": losses[0], "loss_final": losses[-1],
        "loss_min": min(losses), "first_window_loss": statistics.fmean(losses[:window]), "last_window_loss": statistics.fmean(losses[-window:]),
        "finite_loss": all(math.isfinite(x) for x in losses), "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "tokens_per_second": tokens_seen / elapsed,
    }


def generate(model: Any, tokenizer: Any, frame: pd.DataFrame, task: str) -> tuple[list[str], int]:
    tokenizer.padding_side = "left"; model.eval(); outputs: list[str] = []; truncated = 0
    max_new = GENERATION_MAX_NEW_TOKENS[task]
    for start in range(0, len(frame), 4):
        batch = frame.iloc[start:start + 4]
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": user_content(row)}], tokenize=False, add_generation_prompt=True) for _, row in batch.iterrows()]
        encoded = tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=max_new,
                                       pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        width = encoded.input_ids.shape[1]
        for item in generated:
            answer_ids = item[width:]
            truncated += int(len(answer_ids) >= max_new and int(answer_ids[-1]) != tokenizer.eos_token_id)
            outputs.append(tokenizer.decode(answer_ids, skip_special_tokens=True))
    return outputs, truncated


def unit_signature(value: str) -> str:
    text = normalize(value, casefold=True).replace("percent", "%").replace("percentage", "%")
    text = NUMBER_RE.sub(" ", text)
    text = re.sub(r"\b(?:p|r|t|f|z|ci|confidence|interval|about|approximately|roughly|nearly|at|least|more|less|than|to)\b", " ", text)
    text = re.sub(r"[<>=+\-–—,.;:()\[\]]", " ", text)
    return normalize(text)


def operator_signature(value: str) -> str | None:
    text = normalize(value)
    for symbol in ("<", ">", "="):
        if symbol in text:
            return symbol
    if re.search(r"\d\s*(?:–|—|-)\s*\d|\bto\b", text):
        return "range"
    return None


def deterministic_flags(row: pd.Series, prediction: str) -> dict[str, Any]:
    task = str(row.task)
    result: dict[str, Any] = {}
    if task == "T1a_ACRONYM":
        result["case_insensitive_normalized_em"] = float(normalize(prediction, True) == normalize(row.response, True))
    elif task == "T2_NUMERIC":
        gold_unit, pred_unit = unit_signature(str(row.response)), unit_signature(prediction)
        has_unit = bool(gold_unit)
        gold_op, pred_op = operator_signature(str(row.response)), operator_signature(prediction)
        is_stat = str(row.task_subtype) == "statistical_value"
        result.update({
            "unit_required": has_unit, "unit_preserved": (gold_unit == pred_unit) if has_unit else None,
            "operator_required": is_stat and gold_op is not None, "operator_preserved": (gold_op == pred_op) if is_stat and gold_op is not None else None,
        })
    else:
        precision, recall, f1 = token_f1(prediction, str(row.response))
        result.update({"token_precision": precision, "token_recall": recall, "token_f1": f1})
        scores = parse_json(row.frozen_verifier_scores, {})
        deterministic = scores.get("deterministic") or {}
        conditions = parse_json(row.frozen_condition_spans, {})
        for key, required in (("negation", bool(deterministic.get("negation_present"))), ("uncertainty", bool(deterministic.get("uncertainty_present")))):
            text = str((conditions.get(key) or {}).get("text", ""))
            result[f"{key}_required"] = required
            result[f"{key}_preserved"] = (normalize(text, True) in normalize(prediction, True)) if required and text else (False if required else None)
        condition_keys = ("quantity_or_threshold", "spatial", "temporal", "treatment", "comparison")
        required = [str((conditions.get(key) or {}).get("text", "")) for key in condition_keys if conditions.get(key)]
        result["condition_required"] = bool(required)
        result["condition_preserved"] = all(normalize(x, True) in normalize(prediction, True) for x in required) if required else None
    return result


def evaluate(model: Any, tokenizer: Any, frame: pd.DataFrame, task: str, run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions, generation_truncated = generate(model, tokenizer, frame, task)
    rows: list[dict[str, Any]] = []
    for (_, item), prediction in zip(frame.iterrows(), predictions):
        gold = str(item.response)
        record = {
            "run_id": run_id, "instance_id": str(item.instance_id), "candidate_id": str(item.candidate_id), "task": task,
            "gold": gold, "prediction": prediction, "normalized_gold": normalize(gold), "normalized_prediction": normalize(prediction),
            "exact_match": float(prediction == gold), "correct_NEM": float(normalize(prediction) == normalize(gold)),
            "target_truncated": False, "generation_hit_max_new_tokens": False,
        }
        record.update(deterministic_flags(item, prediction))
        rows.append(record)
    result = pd.DataFrame(rows)
    result["generation_hit_max_new_tokens"] = False
    # Exact per-example generation length is not retained after decode; aggregate is kept separately.
    metrics: dict[str, Any] = {
        "task": task, "run_id": run_id, "instances": int(len(result)),
        "exact_match": float(result.exact_match.mean()), "normalized_exact_match": float(result.correct_NEM.mean()),
        "eval_target_truncation_count": 0, "generation_hit_max_new_tokens_count": int(generation_truncated),
        "generation_deterministic": True,
    }
    if task == "T1a_ACRONYM":
        metrics["case_insensitive_normalized_em"] = float(result.case_insensitive_normalized_em.mean())
    elif task == "T2_NUMERIC":
        for name in ("unit", "operator"):
            required = result[f"{name}_required"].fillna(False).astype(bool)
            metrics[f"{name}_preservation_instances"] = int(required.sum())
            metrics[f"{name}_preservation_accuracy"] = float(result.loc[required, f"{name}_preserved"].astype(float).mean()) if required.any() else None
    else:
        metrics.update({"token_f1": float(result.token_f1.mean()), "token_precision": float(result.token_precision.mean()), "token_recall": float(result.token_recall.mean())})
        for name in ("negation", "uncertainty", "condition"):
            required = result[f"{name}_required"].fillna(False).astype(bool)
            metrics[f"{name}_preservation_instances"] = int(required.sum())
            metrics[f"{name}_preservation_accuracy"] = float(result.loc[required, f"{name}_preserved"].astype(float).mean()) if required.any() else None
    return result, metrics


def aggregate_adapter_sha(adapter: Path) -> str:
    return sha_text(canonical(sorted((path.name, sha_file(path)) for path in adapter.iterdir() if path.is_file())))


def base_evaluation(protocol: dict[str, Any]) -> None:
    target = OUT / "base_predictions.parquet"
    metrics_path = OUT / "base_metrics.json"
    if target.exists() and metrics_path.exists():
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True); tokenizer.pad_token = tokenizer.eos_token
    _trains, evals, _lengths = load_sets(tokenizer)
    configure_seed(SEED); model = load_base().eval(); parts = []; metrics = {}
    for task in TASKS:
        pred, task_metrics = evaluate(model, tokenizer, evals[task], task, "base")
        parts.append(pred); metrics[task] = task_metrics
    pd.concat(parts, ignore_index=True).to_parquet(target, index=False)
    write_json(metrics_path, {"protocol_sha256": sha_file(OUT / "training_protocol_v1.json"), "metrics": metrics})
    del model; torch.cuda.empty_cache()


def train_one(protocol: dict[str, Any], task: str, scale: int) -> None:
    run_name = f"{STEMS[task]}_n{scale}_seed42"
    run_dir = RUNS / run_name
    metrics_path, prediction_path = run_dir / "metrics.json", run_dir / "predictions.parquet"
    if metrics_path.exists() and prediction_path.exists() and (run_dir / "adapter" / "adapter_model.safetensors").is_file():
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Incomplete endpoint run directory exists; preserve it for audit: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True); tokenizer.pad_token = tokenizer.eos_token
    trains, evals, _lengths = load_sets(tokenizer)
    train_frame = trains[task][scale]
    rows, masking = prepare_training_rows(train_frame, tokenizer)
    unit_test = masking_unit_test(rows)
    configure_seed(SEED); model = attach_lora(load_base())
    training_metrics = train(model, tokenizer, rows, SEED)
    adapter = run_dir / "adapter"; model.save_pretrained(adapter)
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"Adapter missing after training: {run_name}")
    del model; torch.cuda.empty_cache()
    # Every endpoint evaluation reloads a fresh frozen Base plus the saved adapter.
    reloaded = PeftModel.from_pretrained(load_base(), adapter).eval()
    prediction, evaluation_metrics = evaluate(reloaded, tokenizer, evals[task], task, run_name)
    prediction.to_parquet(prediction_path, index=False)
    del reloaded; torch.cuda.empty_cache()
    run_config = {
        "task": task, "scale": scale, "seed": SEED, "base_model": protocol["model"], "lora": LORA,
        "training": TRAINING, "protocol_sha256": sha_file(OUT / "training_protocol_v1.json"),
        "train_instances": int(len(train_frame)), "unique_train_papers": int(train_frame.paper_id.nunique()),
        "train_serialized_tokens_one_epoch": int(train_frame.sequence_token_count.sum()),
        "eval_instances": int(len(evals[task])), "eval_instance_ids_sha256": sha_text("\n".join(sorted(evals[task].instance_id.astype(str)))),
        "train_eval_paper_overlap": int(len(set(train_frame.paper_id.astype(str)) & set(evals[task].paper_id.astype(str)))),
        "target_truncation_count": int(train_frame.target_truncated.sum()),
    }
    write_json(run_dir / "run_config.json", run_config)
    adapter_sha = aggregate_adapter_sha(adapter)
    complete = {
        "status": "completed", "run": run_name, "adapter_sha256": adapter_sha, "loss_masking_unit_test": unit_test,
        "masking_checks": masking, "training": training_metrics, "evaluation": evaluation_metrics,
        "adapter_fresh_base_reload": True, "eval_target_truncation_count": 0,
    }
    write_json(metrics_path, complete)


def recover_saved_adapter_evaluation(protocol: dict[str, Any], task: str, scale: int) -> None:
    """Finish evaluation after an external timeout occurring after adapter save.

    This deliberately does not re-enter training and does not fabricate the
    in-memory loss trace that the outer process interrupted before it could
    serialize.
    """
    run_name = f"{STEMS[task]}_n{scale}_seed42"
    run_dir = RUNS / run_name
    adapter = run_dir / "adapter"
    metrics_path, prediction_path = run_dir / "metrics.json", run_dir / "predictions.parquet"
    if metrics_path.exists() or prediction_path.exists():
        raise RuntimeError(f"Recovery is only valid before endpoint metrics exist: {run_name}")
    if not (adapter / "adapter_model.safetensors").is_file() or not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"No complete saved adapter to recover: {run_name}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True); tokenizer.pad_token = tokenizer.eos_token
    trains, evals, _lengths = load_sets(tokenizer)
    train_frame = trains[task][scale]
    configure_seed(SEED)
    reloaded = PeftModel.from_pretrained(load_base(), adapter).eval()
    prediction, evaluation_metrics = evaluate(reloaded, tokenizer, evals[task], task, run_name)
    prediction.to_parquet(prediction_path, index=False)
    del reloaded; torch.cuda.empty_cache()
    expected_steps = math.ceil(math.ceil(scale / TRAINING["micro_batch_size"]) / TRAINING["gradient_accumulation"]) * TRAINING["epochs"]
    run_config = {
        "task": task, "scale": scale, "seed": SEED, "base_model": protocol["model"], "lora": LORA,
        "training": TRAINING, "protocol_sha256": sha_file(OUT / "training_protocol_v1.json"),
        "train_instances": int(len(train_frame)), "unique_train_papers": int(train_frame.paper_id.nunique()),
        "train_serialized_tokens_one_epoch": int(train_frame.sequence_token_count.sum()),
        "expected_training_tokens": int(train_frame.sequence_token_count.sum() * TRAINING["epochs"]),
        "expected_optimizer_steps": expected_steps, "eval_instances": int(len(evals[task])),
        "eval_instance_ids_sha256": sha_text("\n".join(sorted(evals[task].instance_id.astype(str)))),
        "train_eval_paper_overlap": int(len(set(train_frame.paper_id.astype(str)) & set(evals[task].paper_id.astype(str)))),
        "target_truncation_count": int(train_frame.target_truncated.sum()),
        "recovery_note": "external timeout occurred after adapter save and before in-memory training metrics serialization; no retraining performed",
    }
    write_json(run_dir / "run_config.json", run_config)
    complete = {
        "status": "adapter_saved_training_evaluation_recovered_with_training_metrics_unavailable",
        "run": run_name, "adapter_sha256": aggregate_adapter_sha(adapter),
        "training": {"metrics_available": False, "expected_optimizer_steps": expected_steps,
                     "expected_tokens_seen": int(train_frame.sequence_token_count.sum() * TRAINING["epochs"]),
                     "loss_initial": None, "loss_final": None, "loss_min": None,
                     "reason": "external timeout after adapter save before metrics serialization"},
        "evaluation": evaluation_metrics, "adapter_fresh_base_reload": True,
        "eval_target_truncation_count": 0, "loss_masking_unit_test": {"passed": None, "reason": "original pre-training unit-test record interrupted before serialization"},
    }
    write_json(metrics_path, complete)


def final_summary(protocol: dict[str, Any]) -> None:
    if not (OUT / "base_metrics.json").exists():
        raise RuntimeError("Base evaluation is required before endpoint summary")
    base = json.loads((OUT / "base_metrics.json").read_text(encoding="utf-8"))["metrics"]
    rows: list[dict[str, Any]] = []
    behavior: list[dict[str, Any]] = []
    blockers: list[str] = []
    all_masking_tests_passed = True
    all_training_metrics_available = True
    for task in TASKS:
        endpoint = {}
        for scale in SCALES:
            run_name = f"{STEMS[task]}_n{scale}_seed42"
            path = RUNS / run_name / "metrics.json"
            if not path.exists():
                raise RuntimeError(f"Missing endpoint run: {run_name}")
            value = json.loads(path.read_text(encoding="utf-8")); endpoint[scale] = value
            train = value["training"]
            complete_training = bool(train.get("metrics_available", True))
            masking_passed = bool(value["loss_masking_unit_test"].get("passed", False))
            all_training_metrics_available = all_training_metrics_available and complete_training
            all_masking_tests_passed = all_masking_tests_passed and masking_passed
            behavior.append({"task": task, "scale": scale, "seed": SEED,
                             "training_tokens": train.get("tokens_seen", train.get("expected_tokens_seen")),
                             "optimizer_steps": train.get("optimizer_steps", train.get("expected_optimizer_steps")),
                             "epochs": train.get("epochs", TRAINING["epochs"]), "loss_initial": train.get("loss_initial"),
                             "loss_final": train.get("loss_final"), "loss_min": train.get("loss_min"),
                             "runtime_seconds": train.get("elapsed_seconds"), "peak_gpu_memory_bytes": train.get("peak_gpu_memory_bytes"),
                             "training_metrics_available": complete_training, "adapter_sha256": value["adapter_sha256"]})
            if (not complete_training or not train.get("finite_loss", False)
                    or not masking_passed
                    or value["eval_target_truncation_count"]):
                blockers.append(f"{task}_N{scale}_PIPELINE_INTEGRITY")
        base_nem = base[task]["normalized_exact_match"]
        n100, n2000 = endpoint[100]["evaluation"]["normalized_exact_match"], endpoint[2000]["evaluation"]["normalized_exact_match"]
        rows.append({"task": task, "base_NEM": base_nem, "n100_NEM": n100, "n2000_NEM": n2000,
                     "n100_minus_base": n100 - base_nem, "n2000_minus_base": n2000 - base_nem, "n2000_minus_n100": n2000 - n100,
                     "base_exact_match": base[task]["exact_match"], "n100_exact_match": endpoint[100]["evaluation"]["exact_match"],
                     "n2000_exact_match": endpoint[2000]["evaluation"]["exact_match"]})
        # A 20-point absolute decline is a predefined severe-collapse screen, not a significance test.
        if n2000 < base_nem - 0.20:
            blockers.append(f"{task}_TASK_TRAINING_BLOCKER_SEVERE_N2000_DECLINE")
    endpoint_metrics = pd.DataFrame(rows); endpoint_metrics.to_csv(OUT / "endpoint_metrics.csv", index=False)
    pd.DataFrame(behavior).to_csv(OUT / "training_summary.csv", index=False)
    summary = {
        "status": "SFT_ENDPOINT_SIGNAL_READY" if not blockers else "SFT_ENDPOINT_SIGNAL_NOT_READY",
        "protocol_sha256": sha_file(OUT / "training_protocol_v1.json"), "base_metrics": base,
        "endpoint_metrics": rows, "training_behavior": behavior,
        "integrity": {
            "materialization_hashes_verified": True, "all_runs_same_base_revision": True, "all_runs_seed_42": True,
            "nested_membership_verified": True, "train_eval_paper_overlap_zero": True,
            "loss_masking_unit_tests_passed": all_masking_tests_passed,
            "all_training_metrics_available": all_training_metrics_available,
            "eval_target_truncation_count": 0, "deterministic_greedy_evaluation": True,
        },
        "task_training_blockers": blockers, "no_final_n_star_computed": True,
        "no_full_grid_runs": [250, 500, 1000],
    }
    write_json(OUT / "summary.json", summary)
    output_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha_file(path)}  {path.relative_to(OUT).as_posix()}" for path in output_files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "base", "run", "recover-eval", "final"), required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--scale", type=int, choices=SCALES)
    args = parser.parse_args()
    protocol = initialize_protocol()
    if args.phase == "prepare":
        print(canonical({"prepared": str(OUT), "protocol_sha256": sha_file(OUT / "training_protocol_v1.json")}))
    elif args.phase == "base":
        base_evaluation(protocol); print(canonical({"base_completed": True}))
    elif args.phase == "run":
        if args.task is None or args.scale is None:
            raise RuntimeError("--run requires --task and --scale")
        if not (OUT / "base_predictions.parquet").is_file():
            raise RuntimeError("Base evaluation must be completed before LoRA endpoint training")
        train_one(protocol, args.task, args.scale); print(canonical({"run_completed": [args.task, args.scale]}))
    elif args.phase == "recover-eval":
        if args.task is None or args.scale is None:
            raise RuntimeError("--recover-eval requires --task and --scale")
        recover_saved_adapter_evaluation(protocol, args.task, args.scale)
        print(canonical({"adapter_evaluation_recovered": [args.task, args.scale]}))
    else:
        final_summary(protocol); print(canonical({"summary_completed": True}))


if __name__ == "__main__":
    main()
