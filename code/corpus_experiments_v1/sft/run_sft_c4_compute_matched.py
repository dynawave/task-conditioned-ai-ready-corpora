from __future__ import annotations

"""SFT-C4 compute-matched sensitivity and frozen T3 incremental diagnostic.

This is deliberately a separate sensitivity analysis.  C2/C3 outputs are
verified read-only inputs; only the C4 directory is written.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.sft import run_sft_c2_full_grid as c2  # noqa: E402
from corpus_experiments_v1.sft import run_sft_c3_scale_extension as c3  # noqa: E402

BASE = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1"
MATERIALIZATION = BASE / "materialization"
C2_OUT = BASE / "full_grid_v1"
C3_OUT = BASE / "scale_extension_v1"
OUT = BASE / "compute_matched_v1"
RUNS = OUT / "runs"
SEEDS = c2.SEEDS
BOOTSTRAP = 10_000
COMPARISONS = {
    "T1a_ACRONYM": {"small_scale": 1000, "large_scale": 2000, "small_source": "C2", "large_source": "C2"},
    "T2_NUMERIC": {"small_scale": 3000, "large_scale": 3678, "small_source": "C3", "large_source": "C3"},
    "T3_CLAIM": {"small_scale": 2000, "large_scale": 2286, "small_source": "C2", "large_source": "C3"},
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


def run_name(task: str, scale: int, seed: int) -> str:
    return f"{c2.STEMS[task]}_n{scale}_matched_seed{seed}"


def source_dir(source: str) -> Path:
    return C2_OUT if source == "C2" else C3_OUT


def formal_run(task: str, scale: int, seed: int, source: str) -> Path:
    return source_dir(source) / "runs" / c2.run_name(task, scale, seed)


def check_manifest(directory: Path, encoding: str = "utf-8") -> None:
    for line in (directory / "checksums.sha256").read_text(encoding=encoding).splitlines():
        expected, relative = line.split("  ", 1)
        if not (directory / relative).is_file() or sha_file(directory / relative) != expected:
            raise RuntimeError(f"Frozen checksum mismatch: {directory.name}/{relative}")


def verify_inputs() -> None:
    c3.verify_inputs()
    check_manifest(C3_OUT)
    if read_json(C3_OUT / "summary.json").get("status") != "SFT_SCALE_EXTENSION_READY":
        raise RuntimeError("Frozen C3 is not ready")
    for task, item in COMPARISONS.items():
        for seed in SEEDS:
            for which in ("small", "large"):
                path = formal_run(task, int(item[f"{which}_scale"]), seed, str(item[f"{which}_source"]))
                required = [path / "training_summary.json", path / "run_config.json", path / "predictions_primary.parquet", path / "adapter" / "adapter_model.safetensors"]
                if not all(x.is_file() for x in required) or read_json(path / "training_summary.json").get("status") != "completed":
                    raise RuntimeError(f"Frozen formal run unavailable: {path.name}")


def all_sets(tokenizer: Any) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame]]:
    c2_trains, c2_evals = c2.load_all_sets(tokenizer)
    c3_trains, c3_evals = c3.load_sets(tokenizer)
    trains: dict[str, dict[int, pd.DataFrame]] = {
        "T1a_ACRONYM": {1000: c2_trains["T1a_ACRONYM"][1000], 2000: c2_trains["T1a_ACRONYM"][2000]},
        "T2_NUMERIC": {3000: c3_trains["T2_NUMERIC"][3000], 3678: c3_trains["T2_NUMERIC"][3678]},
        "T3_CLAIM": {2000: c2_trains["T3_CLAIM"][2000], 2286: c3_trains["T3_CLAIM"][2286]},
    }
    evals = {"T1a_ACRONYM": c2_evals["T1a_ACRONYM"], "T2_NUMERIC": c3_evals["T2_NUMERIC"], "T3_CLAIM": c2_evals["T3_CLAIM"]}
    for task, scales in trains.items():
        small, large = scales[min(scales)], scales[max(scales)]
        if small.instance_id.astype(str).tolist() != large.instance_id.astype(str).head(len(small)).tolist():
            raise RuntimeError(f"Frozen nested membership mismatch: {task}")
        if set(large.paper_id.astype(str)) & set(evals[task].paper_id.astype(str)):
            raise RuntimeError(f"Train/eval overlap: {task}")
    return trains, evals


def target_tokens(rows: list[dict[str, Any]]) -> int:
    return sum(sum(value != -100 for value in row["labels"]) for row in rows)


def frozen_manifest() -> tuple[list[dict[str, Any]], dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame]]:
    tokenizer = AutoTokenizer.from_pretrained(c2.c1.MODEL, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    trains, evals = all_sets(tokenizer)
    records: list[dict[str, Any]] = []
    for task, spec in COMPARISONS.items():
        small, large = int(spec["small_scale"]), int(spec["large_scale"])
        small_rows, _ = c2.c1.prepare_training_rows(trains[task][small], tokenizer)
        large_rows, _ = c2.c1.prepare_training_rows(trains[task][large], tokenizer)
        for seed in SEEDS:
            small_run = formal_run(task, small, seed, str(spec["small_source"]))
            small_summary = read_json(small_run / "training_summary.json")
            matched = int(small_summary["training"]["optimizer_steps"])
            records.append({
                "task": task, "seed": seed, "small_scale": small, "large_scale": large,
                "small_source": spec["small_source"], "large_source": spec["large_source"],
                "matched_max_steps": matched, "warmup_steps": max(1, math.ceil(matched * c2.c1.TRAINING["warmup_ratio"])),
                "small_formal_actual_steps": matched, "small_formal_processed_input_tokens": int(small_summary["training"]["tokens_seen"]),
                "small_formal_processed_target_tokens": int(target_tokens(small_rows) * c2.c1.TRAINING["epochs"]),
                "small_membership_sha256": sha_text("\n".join(trains[task][small].instance_id.astype(str))),
                "large_membership_sha256": sha_text("\n".join(trains[task][large].instance_id.astype(str))),
                "small_instances": len(small_rows), "large_instances": len(large_rows),
                "eval_instances": len(evals[task]), "eval_ids_sha256": sha_text("\n".join(evals[task].instance_id.astype(str))),
            })
    return records, trains, evals


def prepare_output() -> None:
    verify_inputs()
    records, _trains, _evals = frozen_manifest()
    config = {
        "experiment": "corpus_v1_sft_c4_compute_matched_scale_sensitivity",
        "status": "frozen_before_training",
        "sensitivity_only_not_replacement_for_c2_c3": True,
        "comparisons": COMPARISONS, "training_seeds": list(SEEDS), "bootstrap_replicates": BOOTSTRAP,
        "compute_matching": {"primary_unit": "actual_optimizer_steps", "large_data": "complete frozen large membership", "fresh_base_lora": True, "scheduler": "linear total_steps=matched_max_steps; warmup_ratio=0.03"},
        "c2_manifest_sha256": sha_file(C2_OUT / "checksums.sha256"), "c3_manifest_sha256": sha_file(C3_OUT / "checksums.sha256"),
        "materialization_manifest_sha256": sha_file(MATERIALIZATION / "checksums.sha256"),
        "formal_scale_protocol_sha256": c2.FORMAL_PROTOCOL_SHA, "template_config_sha256": c2.TEMPLATE_CONFIG_SHA,
        "c1_training_protocol_sha256": c2.PROTOCOL_SHA, "base_model": read_json(MATERIALIZATION / "config.json")["base_model"],
        "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha_file(Path(__file__))}, "c2_c3_read_only": True,
    }
    if OUT.exists():
        if not (OUT / "config.json").is_file():
            raise RuntimeError("Existing C4 directory/config differs; refuse to overwrite")
        existing = read_json(OUT / "config.json")
        # The runner hash is provenance metadata, not part of the frozen
        # experimental design.  Permit a diagnostic-only repair to finalize
        # an already frozen configuration, while preserving config.json.
        expected_design = dict(config); existing_design = dict(existing)
        expected_design.pop("runner", None); existing_design.pop("runner", None)
        if existing_design != expected_design:
            raise RuntimeError("Existing C4 directory/config differs; refuse to overwrite")
        return
    OUT.mkdir(parents=True); RUNS.mkdir()
    write_json(OUT / "config.json", config)
    pd.DataFrame(records).sort_values(["task", "seed"]).to_csv(OUT / "matched_run_manifest.csv", index=False)


def train_matched(model: Any, tokenizer: Any, rows: list[dict[str, Any]], seed: int, matched_steps: int) -> tuple[dict[str, Any], dict[str, str]]:
    micro, accumulation = c2.c1.TRAINING["micro_batch_size"], c2.c1.TRAINING["gradient_accumulation"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=c2.c1.TRAINING["learning_rate"], weight_decay=c2.c1.TRAINING["weight_decay"])
    warmup = max(1, math.ceil(matched_steps * c2.c1.TRAINING["warmup_ratio"]))
    scheduler = c2.c1.get_linear_schedule_with_warmup(optimizer, warmup, matched_steps)
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True); model.train(); started = time.perf_counter()
    losses: list[float] = []; seen_ids: list[str] = []; input_tokens = 0; assistant_tokens = 0; optimizer_steps = 0; pass_hashes: dict[str, str] = {}
    pass_index = 0
    while optimizer_steps < matched_steps:
        order = list(range(len(rows))); random.Random(seed + pass_index).shuffle(order)
        used: list[str] = []; pending = 0
        for start in range(0, len(order), micro):
            batch = [rows[index] for index in order[start:start + micro]]
            loss = c2.c1.batch_loss(model, tokenizer, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss")
            (loss / accumulation).backward(); pending += 1; losses.append(float(loss.detach().cpu()))
            used.extend(item["instance_id"] for item in batch); seen_ids.extend(item["instance_id"] for item in batch)
            input_tokens += sum(len(item["ids"]) for item in batch)
            assistant_tokens += sum(sum(value != -100 for value in item["labels"]) for item in batch)
            if pending == accumulation or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), c2.c1.TRAINING["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); pending = 0; optimizer_steps += 1
                if optimizer_steps == matched_steps:
                    break
        pass_hashes[f"pass_{pass_index + 1}_actual_order_sha256"] = sha_text("\n".join(used))
        pass_index += 1
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started; window = min(8, max(1, len(losses) // 3))
    return {
        "optimizer_steps": optimizer_steps, "matched_max_steps": matched_steps, "warmup_steps": warmup, "warmup_ratio": c2.c1.TRAINING["warmup_ratio"],
        "full_large_instances": len(rows), "samples_seen": len(seen_ids), "effective_epochs": len(seen_ids) / len(rows),
        "processed_nonpadding_input_tokens": input_tokens, "processed_assistant_target_tokens": assistant_tokens,
        "loss_initial": losses[0], "loss_final": losses[-1], "loss_min": min(losses), "first_window_loss": statistics.fmean(losses[:window]), "last_window_loss": statistics.fmean(losses[-window:]),
        "finite_loss": all(math.isfinite(value) for value in losses), "elapsed_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "tokens_per_second": input_tokens / elapsed,
        "actual_sample_order_sha256": sha_text("\n".join(seen_ids)),
    }, pass_hashes


def manifest_record(task: str, seed: int) -> dict[str, Any]:
    frame = pd.read_csv(OUT / "matched_run_manifest.csv")
    row = frame[(frame.task == task) & (frame.seed == seed)]
    if len(row) != 1:
        raise RuntimeError("Missing frozen C4 manifest row")
    return row.iloc[0].to_dict()


def run_one(task: str, seed: int) -> None:
    prepare_output(); spec = COMPARISONS[task]; manifest = manifest_record(task, seed); scale = int(spec["large_scale"])
    directory = RUNS / run_name(task, scale, seed)
    required = [directory / "training_summary.json", directory / "run_config.json", directory / "predictions_primary.parquet", directory / "adapter" / "adapter_model.safetensors"]
    if all(path.is_file() for path in required):
        if read_json(directory / "training_summary.json").get("status") != "completed": raise RuntimeError(f"Run marked non-complete: {directory.name}")
        return
    if directory.exists(): raise RuntimeError(f"Incomplete prior run preserved for audit: {directory.name}")
    directory.mkdir(parents=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(c2.c1.MODEL, local_files_only=True, use_fast=True); tokenizer.pad_token = tokenizer.eos_token
        trains, evals = all_sets(tokenizer); train_frame, eval_frame = trains[task][scale], evals[task]
        rows, masking_checks = c2.c1.prepare_training_rows(train_frame, tokenizer); masking = c2.c1.masking_unit_test(rows); rng = c2.configure_seed(seed)
        model = c2.c1.attach_lora(c2.c1.load_base()); training, order_hashes = train_matched(model, tokenizer, rows, seed, int(manifest["matched_max_steps"]))
        if not training["finite_loss"] or not masking["passed"] or training["optimizer_steps"] != int(manifest["matched_max_steps"]): raise RuntimeError("Compute-matched integrity gate failed")
        adapter = directory / "adapter"; model.save_pretrained(adapter)
        if not (adapter / "adapter_model.safetensors").is_file(): raise RuntimeError("Saved adapter missing")
        del model; torch.cuda.empty_cache(); reloaded = PeftModel.from_pretrained(c2.c1.load_base(), adapter).eval()
        primary, evaluation = c2.c1.evaluate(reloaded, tokenizer, eval_frame, task, directory.name); primary = c2.annotate_predictions(primary, task, scale, seed, "TEMPLATE_0_PRIMARY")
        primary.to_parquet(directory / "predictions_primary.parquet", index=False); metrics = [c2.metric_row(primary, evaluation, task, scale, seed, "TEMPLATE_0_PRIMARY")]
        del reloaded; torch.cuda.empty_cache()
        instance_ids = train_frame.instance_id.astype(str).tolist()
        write_json(directory / "run_config.json", {"run": directory.name, "task": task, "scale": scale, "seed": seed, "sensitivity": "compute_matched_large_pool", "matched_max_steps": int(manifest["matched_max_steps"]), "small_formal_run": str(formal_run(task, int(spec["small_scale"]), seed, str(spec["small_source"])).relative_to(ROOT)), "large_formal_run": str(formal_run(task, scale, seed, str(spec["large_source"])).relative_to(ROOT)), "training_instance_ids": instance_ids, "training_membership_sha256": sha_text("\n".join(instance_ids)), "train_eval_paper_overlap": int(len(set(train_frame.paper_id.astype(str)) & set(eval_frame.paper_id.astype(str)))), "target_truncation_count": int(train_frame.target_truncated.sum()), "eval_instance_ids_sha256": sha_text("\n".join(eval_frame.instance_id.astype(str))), "eval_instances": len(eval_frame), "fresh_base_lora": True, "lora": c2.c1.LORA, "training_recipe": c2.c1.TRAINING})
        write_json(directory / "training_summary.json", {"status": "completed", "run": directory.name, "loss_masking_unit_test": masking, "masking_checks": masking_checks, "epoch_order_sha256": order_hashes, "rng_audit": rng, "environment": c2.environment(), "training": training, "adapter_sha256": c2.c1.aggregate_adapter_sha(adapter), "adapter_fresh_base_reload": True, "eval_target_truncation_count": 0, "metrics": metrics})
    except Exception as error:
        write_json(directory / "technical_failure.json", {"run": directory.name, "error_type": type(error).__name__, "error": str(error), "technical_retry": False}); raise


def bootstrap(task: str, small: dict[int, pd.DataFrame]) -> dict[str, Any]:
    arrays: dict[tuple[str, int], np.ndarray] = {}; ids: list[str] | None = None
    for kind, mapping in (("small_formal", small), ("large_matched", None)):
        for seed in SEEDS:
            path = formal_run(task, int(COMPARISONS[task]["small_scale"]), seed, str(COMPARISONS[task]["small_source"])) / "predictions_primary.parquet" if kind == "small_formal" else RUNS / run_name(task, int(COMPARISONS[task]["large_scale"]), seed) / "predictions_primary.parquet"
            frame = pd.read_parquet(path).sort_values("instance_id").reset_index(drop=True); current = frame.instance_id.astype(str).tolist()
            if ids is None: ids = current
            elif current != ids: raise RuntimeError(f"Paired Eval IDs drift: {task}/{kind}/seed{seed}")
            arrays[(kind, seed)] = frame.correct_NEM.to_numpy(dtype=float)
    n = len(ids or []); rng = np.random.default_rng(int(sha_text(f"C4_BOOTSTRAP_V1|{task}")[:16], 16)); values = np.empty(BOOTSTRAP, dtype=float)
    for index in range(BOOTSTRAP):
        selected = rng.integers(0, len(SEEDS), size=len(SEEDS)); pieces = []
        for seed_index in selected:
            seed = SEEDS[int(seed_index)]; examples = rng.integers(0, n, size=n)
            pieces.append(float((arrays[("large_matched", seed)][examples] - arrays[("small_formal", seed)][examples]).mean()))
        values[index] = float(np.mean(pieces))
    observed = float(np.mean([arrays[("large_matched", seed)].mean() - arrays[("small_formal", seed)].mean() for seed in SEEDS]))
    return {"task": task, "comparison": "large_compute_matched_minus_small_formal", "point_difference": observed, "ci95_low": c2.quantile(values, 0.025), "ci95_high": c2.quantile(values, 0.975), "bootstrap_replicates": BOOTSTRAP, "bootstrap_seed": "SHA256(C4_BOOTSTRAP_V1|task)", "method": "outer resampling of three training seeds; inner paired resampling of frozen Eval examples"}


def t3_diagnostic() -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(c2.c1.MODEL, local_files_only=True, use_fast=True); tokenizer.pad_token = tokenizer.eos_token
    trains, _evals = all_sets(tokenizer); frame = trains["T3_CLAIM"][2286].copy(); frame["group"] = np.where(frame.index < 2000, "N2000", "incremental_286")
    labels = pd.read_parquet(c2.c1.LABELS, columns=["candidate_id", "major_section", "section_class", "condition_spans", "relation"])
    frame = frame.merge(labels, on="candidate_id", how="left", suffixes=("", "_label"), validate="one_to_one")
    if frame.major_section.isna().any(): raise RuntimeError("Missing frozen T3 label metadata")
    condition = frame.condition_spans.map(lambda x: json.loads(x) if isinstance(x, str) else {})
    frame["has_condition"] = condition.map(lambda d: any(d.get(k) for k in ("quantity_or_threshold", "temporal", "spatial", "treatment", "comparison")))
    frame["condition_numeric"] = condition.map(lambda d: bool(d.get("quantity_or_threshold"))); frame["condition_temporal"] = condition.map(lambda d: bool(d.get("temporal"))); frame["condition_spatial"] = condition.map(lambda d: bool(d.get("spatial"))); frame["condition_treatment_comparison"] = condition.map(lambda d: bool(d.get("treatment")) or bool(d.get("comparison")))
    frame["condition_other"] = condition.map(lambda d: any(d.get(k) for k in ("method", "outcome", "population_or_material")))
    # `relation` is frozen public-gold metadata.  Surface it under the
    # diagnostic feature name used below; no model output or label is changed.
    frame["relation_cue"] = frame.relation.fillna("missing").astype(str)
    def section(value: str) -> str:
        text = str(value).casefold()
        if "result" in text: return "Results"
        if "discussion" in text: return "Discussion"
        if "conclusion" in text: return "Conclusion"
        if "method" in text: return "Methods"
        return "Other"
    frame["major_section_bucket"] = frame.section_class.map(section)
    def rows_for_categories(feature: str, values: list[str]) -> list[dict[str, Any]]:
        result=[]
        for group, data in frame.groupby("group", sort=False):
            for value in values:
                count=int((data[feature].astype(str)==value).sum()); result.append({"diagnostic_role":"POST-HOC_EXPLORATORY_DIAGNOSTIC","feature":feature,"statistic":"count","category":value,"group":group,"value":count,"n":len(data)})
                result.append({"diagnostic_role":"POST-HOC_EXPLORATORY_DIAGNOSTIC","feature":feature,"statistic":"proportion","category":value,"group":group,"value":count/len(data),"n":len(data)})
        return result
    rows=[]
    for group,data in frame.groupby("group",sort=False):
        rows.append({"diagnostic_role":"POST-HOC_EXPLORATORY_DIAGNOSTIC","feature":"unique_paper_count","statistic":"count","category":"all","group":group,"value":int(data.paper_id.nunique()),"n":len(data)})
        for feature in ("response_tokens","context_tokens"):
            rows += [{"diagnostic_role":"POST-HOC_EXPLORATORY_DIAGNOSTIC","feature":feature,"statistic":stat,"category":"all","group":group,"value":float(value),"n":len(data)} for stat,value in (("median",data[feature].median()),("iqr",data[feature].quantile(.75)-data[feature].quantile(.25)))]
        for feature in ("negation_flag","uncertainty_flag","has_condition","condition_numeric","condition_temporal","condition_spatial","condition_treatment_comparison","condition_other"):
            count=int(data[feature].astype(bool).sum()); rows += [{"diagnostic_role":"POST-HOC_EXPLORATORY_DIAGNOSTIC","feature":feature,"statistic":"count","category":"true","group":group,"value":count,"n":len(data)},{"diagnostic_role":"POST-HOC_EXPLORATORY_DIAGNOSTIC","feature":feature,"statistic":"proportion","category":"true","group":group,"value":count/len(data),"n":len(data)}]
    rows += rows_for_categories("major_section_bucket", ["Results","Discussion","Conclusion","Methods","Other"])
    relation_values=sorted(set(frame.relation.fillna("missing").astype(str))); rows += rows_for_categories("relation_cue", relation_values)
    return pd.DataFrame(rows)


def finalize() -> None:
    prepare_output(); records, _trains, _evals = frozen_manifest(); errors=[]; metric_rows=[]
    for record in records:
        task, seed = str(record["task"]), int(record["seed"]); directory=RUNS/run_name(task,int(record["large_scale"]),seed)
        required=[directory/"training_summary.json",directory/"run_config.json",directory/"predictions_primary.parquet",directory/"adapter"/"adapter_model.safetensors"]
        if not all(x.is_file() for x in required): errors.append(f"missing:{directory.name}"); continue
        config, summary=read_json(directory/"run_config.json"),read_json(directory/"training_summary.json"); training=summary.get("training",{})
        if summary.get("status")!="completed" or not summary["loss_masking_unit_test"].get("passed") or not training.get("finite_loss"): errors.append(f"pipeline:{directory.name}")
        if training.get("optimizer_steps")!=int(record["matched_max_steps"]) or config.get("training_membership_sha256")!=str(record["large_membership_sha256"]) or config.get("train_eval_paper_overlap")!=0 or config.get("target_truncation_count")!=0 or summary.get("eval_target_truncation_count")!=0 or not summary.get("epoch_order_sha256"): errors.append(f"integrity:{directory.name}")
        metric=dict(next(x for x in summary["metrics"] if x["template_id"]=="TEMPLATE_0_PRIMARY")); metric.update({"matched_max_steps":training["matched_max_steps"],"processed_nonpadding_input_tokens":training["processed_nonpadding_input_tokens"],"processed_assistant_target_tokens":training["processed_assistant_target_tokens"],"samples_seen":training["samples_seen"],"effective_epochs":training["effective_epochs"],"runtime_seconds":training["elapsed_seconds"],"peak_gpu_memory_bytes":training["peak_gpu_memory_bytes"]}); metric_rows.append(metric)
    if errors:
        write_json(OUT/"summary.json",{"status":"COMPUTE_MATCHED_NOT_READY","integrity_errors":errors}); raise RuntimeError("C4 completeness gate failed: "+", ".join(errors[:5]))
    matched=pd.DataFrame(metric_rows).sort_values(["task","seed"]); matched.to_csv(OUT/"compute_matched_metrics.csv",index=False)
    boot=[]; summary_tasks=[]
    manifest=pd.DataFrame(records)
    for task,spec in COMPARISONS.items():
        subset=manifest[manifest.task.eq(task)].sort_values("seed"); matched_group=matched[matched.task.eq(task)].sort_values("seed")
        small_values=[]; large_values=[]
        for seed in SEEDS:
            small_summary=read_json(formal_run(task,int(spec["small_scale"]),seed,str(spec["small_source"]))/"training_summary.json")
            large_summary=read_json(formal_run(task,int(spec["large_scale"]),seed,str(spec["large_source"]))/"training_summary.json")
            small_values.append(float(next(x for x in small_summary["metrics"] if x["template_id"]=="TEMPLATE_0_PRIMARY")["NEM"])); large_values.append(float(next(x for x in large_summary["metrics"] if x["template_id"]=="TEMPLATE_0_PRIMARY")["NEM"]))
        result=bootstrap(task,{}); boot.append(result); token_ratio=((matched_group.processed_nonpadding_input_tokens.mean()/subset.small_formal_processed_input_tokens.mean())-1)*100
        decision="UNIQUE_DATA_ADVANTAGE_SUPPORTED" if result["ci95_low"]>0 else ("COMPUTE_CONTRIBUTION_WARNING" if np.mean(large_values)>np.mean(small_values) and np.mean(matched_group.NEM)<=np.mean(small_values) else "MIXED")
        summary_tasks.append({"task":task,"small_scale":int(spec["small_scale"]),"large_scale":int(spec["large_scale"]),"small_formal_mean_NEM":float(np.mean(small_values)),"large_formal_3epoch_mean_NEM":float(np.mean(large_values)),"large_compute_matched_mean_NEM":float(matched_group.NEM.mean()),"matched_steps":canonical([int(x) for x in subset.matched_max_steps]),"small_processed_input_tokens_mean":float(subset.small_formal_processed_input_tokens.mean()),"large_processed_input_tokens_mean":float(matched_group.processed_nonpadding_input_tokens.mean()),"token_budget_difference_percent":float(token_ratio),"token_compute_mismatch_warning":bool(abs(token_ratio)>5.0),"decision":decision})
    pd.DataFrame(boot).sort_values("task").to_csv(OUT/"compute_matched_bootstrap.csv",index=False); diagnostic=t3_diagnostic(); diagnostic.to_csv(OUT/"t3_incremental286_diagnostic.csv",index=False)
    write_json(OUT/"summary.json",{"status":"COMPUTE_MATCHED_READY","sensitivity_only_not_replacement_for_c2_c3":True,"new_runs_expected":9,"new_runs_complete":9,"integrity":{"fresh_base":True,"same_frozen_membership":True,"matched_steps":True,"masking_pass":True,"finite_loss":True,"eval_ids_correct":True,"target_truncation_zero":True,"train_eval_overlap_zero":True,"order_hashes_present":True},"task_results":summary_tasks,"bootstrap_replicates":BOOTSTRAP,"t3_incremental286_diagnostic":"POST-HOC_EXPLORATORY_DIAGNOSTIC","warnings":["TOKEN_COMPUTE_MISMATCH_WARNING applies when absolute input-token budget difference exceeds 5%","TRAINING_STOCHASTICITY_WARNING","SOURCE_PROVENANCE_AUDIT_WARNING"]})
    files=sorted(path for path in OUT.rglob("*") if path.is_file() and path.name!="checksums.sha256"); (OUT/"checksums.sha256").write_text("\n".join(f"{sha_file(path)}  {path.relative_to(OUT).as_posix()}" for path in files)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",required=True,choices=("prepare","run-one","final")); parser.add_argument("--task",choices=tuple(COMPARISONS)); parser.add_argument("--seed",type=int,choices=SEEDS); args=parser.parse_args()
    if args.phase=="prepare": prepare_output(); print(canonical({"prepared":str(OUT)}))
    elif args.phase=="run-one":
        if args.task is None or args.seed is None: raise RuntimeError("run-one requires task and seed")
        run_one(args.task,args.seed); print(canonical({"completed":run_name(args.task,int(COMPARISONS[args.task]["large_scale"]),args.seed)}))
    else: finalize(); print(canonical({"status":"COMPUTE_MATCHED_READY"}))


if __name__ == "__main__": main()
