from __future__ import annotations

"""No-SFT control for the frozen corpus_v1 RQ3 evaluation protocol.

This runner performs no training and never attaches or loads a PEFT adapter.
It reuses the frozen C2 input gates and the C1 Base loader/evaluator directly.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from corpus_experiments_v1.sft import run_sft_c2_full_grid as c2


c1 = c2.c1
ROOT = c2.ROOT
OUT = c2.BASE / "no_sft_control_v1"
SUMMARY_PATH = OUT / "no_sft_control_summary.json"
PREDICTIONS_PATH = OUT / "no_sft_control_predictions.parquet"
RUN_ID = "qwen25_3b_no_sft_control_template0_primary"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_eval_reference(task: str, eval_ids_sha256: str, eval_n: int) -> dict[str, Any]:
    observed_hashes: set[str] = set()
    observed_ns: set[int] = set()
    for scale in c2.SCALES:
        for seed in c2.SEEDS:
            config_path = c2.RUNS / c2.run_name(task, scale, seed) / "run_config.json"
            if not config_path.is_file():
                raise RuntimeError(f"Missing frozen C2 run config: {config_path}")
            config = read_json(config_path)
            observed_hashes.add(str(config["eval_instance_ids_sha256"]))
            observed_ns.add(int(config["eval_instances"]))
    if observed_hashes != {eval_ids_sha256} or observed_ns != {eval_n}:
        raise RuntimeError(f"Frozen C2 evaluation membership mismatch: {task}")
    return {
        "formal_c2_run_configs_checked": len(c2.SCALES) * len(c2.SEEDS),
        "formal_c2_eval_instance_ids_sha256": next(iter(observed_hashes)),
        "formal_c2_eval_n": next(iter(observed_ns)),
    }


def n100_reference(task: str) -> dict[str, Any]:
    values: list[float] = []
    for seed in c2.SEEDS:
        summary_path = c2.RUNS / c2.run_name(task, 100, seed) / "training_summary.json"
        summary = read_json(summary_path)
        primary = [item for item in summary["metrics"] if item["template_id"] == "TEMPLATE_0_PRIMARY"]
        if len(primary) != 1:
            raise RuntimeError(f"Invalid frozen N100 primary metric: {summary_path}")
        values.append(float(primary[0]["NEM"]))
    return {
        "seeds": list(c2.SEEDS),
        "seed_NEM": values,
        "mean_NEM": float(statistics.fmean(values)),
    }


def assert_existing_base_match(predictions: pd.DataFrame) -> None:
    frozen_path = c2.C1 / "base_predictions.parquet"
    if not frozen_path.is_file():
        raise RuntimeError(f"Missing frozen C1 Base predictions: {frozen_path}")
    frozen = pd.read_parquet(frozen_path)
    key = ["task", "instance_id"]
    fields = key + ["prediction", "normalized_prediction", "correct_NEM", "exact_match"]
    left = predictions[fields].sort_values(key).reset_index(drop=True)
    right = frozen[fields].sort_values(key).reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Current No-SFT predictions differ from frozen C1 Base predictions")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Output already exists; refuse to overwrite: {OUT}")

    c2.verify_frozen_inputs()
    tokenizer = AutoTokenizer.from_pretrained(c1.MODEL, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    _trains, evals = c2.load_all_sets(tokenizer)

    membership: dict[str, dict[str, Any]] = {}
    for task in c2.TASKS:
        frame = evals[task]
        ids = frame.instance_id.astype(str).tolist()
        ids_hash = c2.sha_text("\n".join(ids))
        membership[task] = {
            "eval_instance_ids_sha256": ids_hash,
            "eval_n": int(len(frame)),
            "eval_file": str((c2.MATERIALIZATION / f"{c2.STEMS[task]}_eval.parquet").relative_to(ROOT)),
            "eval_file_sha256": c2.sha_file(c2.MATERIALIZATION / f"{c2.STEMS[task]}_eval.parquet"),
            **frozen_eval_reference(task, ids_hash, len(frame)),
        }

    started = time.perf_counter()
    model = c1.load_base().eval()
    if isinstance(model, PeftModel) or hasattr(model, "peft_config"):
        raise RuntimeError("No-SFT control unexpectedly loaded a PEFT adapter")

    prediction_parts: list[pd.DataFrame] = []
    task_results: list[dict[str, Any]] = []
    for task in c2.TASKS:
        predictions, evaluation = c1.evaluate(model, tokenizer, evals[task], task, RUN_ID)
        predictions["template_id"] = "TEMPLATE_0_PRIMARY"
        predictions["control"] = "NO_SFT"
        prediction_parts.append(predictions)

        n100 = n100_reference(task)
        no_sft_nem = float(evaluation["normalized_exact_match"])
        task_results.append({
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "task": task,
            "eval_n": int(evaluation["instances"]),
            "NEM": no_sft_nem,
            "exact_match": float(evaluation["exact_match"]),
            "evaluation_template": "TEMPLATE_0_PRIMARY",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "eval_batch_size": 4,
                "max_new_tokens": int(c1.GENERATION_MAX_NEW_TOKENS[task]),
                "generation_hit_max_new_tokens_count": int(evaluation["generation_hit_max_new_tokens_count"]),
                "deterministic": bool(evaluation["generation_deterministic"]),
                "evaluation_seed": None,
                "repeat_count": 1,
            },
            "eval_membership": membership[task],
            "n100_formal_sft": n100,
            "difference_no_sft_minus_n100_mean_NEM": no_sft_nem - float(n100["mean_NEM"]),
            "difference_n100_mean_minus_no_sft_NEM": float(n100["mean_NEM"]) - no_sft_nem,
            "status": "PASS",
        })

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    assert_existing_base_match(all_predictions)
    runtime_seconds = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()

    OUT.mkdir(parents=False, exist_ok=False)
    all_predictions.to_parquet(PREDICTIONS_PATH, index=False)
    base_model = read_json(c2.MATERIALIZATION / "config.json")["base_model"]
    summary = {
        "experiment": "corpus_v1_rq3_no_sft_control_v1",
        "status": "NO_SFT_CONTROL_READY",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": base_model,
        "task_results": task_results,
        "evaluation_protocol": {
            "template": "TEMPLATE_0_PRIMARY",
            "prompt": "official tokenizer.apply_chat_template; frozen instruction + Passage + frozen context",
            "metric": "c1.evaluate normalized exact match",
            "normalization": "Unicode NFKC; trim; collapse whitespace; case-sensitive",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "eval_batch_size": 4,
                "max_new_tokens": c1.GENERATION_MAX_NEW_TOKENS,
            },
        },
        "protocol_checks": {
            "frozen_inputs": "PASS",
            "eval_membership_matches_all_formal_c2_runs": "PASS",
            "prompt_template_matches_formal_rq3": "PASS",
            "metric_function_is_c1_evaluate": "PASS",
            "base_checkpoint_matches_frozen_rq3": "PASS",
            "current_predictions_match_frozen_c1_base": "PASS",
            "adapter_loaded": False,
            "training_performed": False,
            "all_checks_pass": True,
        },
        "runtime_seconds": runtime_seconds,
        "predictions": {
            "path": str(PREDICTIONS_PATH.relative_to(ROOT)),
            "rows": int(len(all_predictions)),
            "sha256": c2.sha_file(PREDICTIONS_PATH),
        },
        "source_code": {
            "runner": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": c2.sha_file(Path(__file__)),
            "c1_runner": str(Path(c1.__file__).relative_to(ROOT)),
            "c1_runner_sha256": c2.sha_file(Path(c1.__file__)),
            "c2_runner": str(Path(c2.__file__).relative_to(ROOT)),
            "c2_runner_sha256": c2.sha_file(Path(c2.__file__)),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "summary": str(SUMMARY_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
