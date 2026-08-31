from __future__ import annotations

"""Frozen C2 hierarchical paired bootstrap for N100 SFT versus No-SFT."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from corpus_experiments_v1.sft import run_sft_c2_full_grid as c2


OUT = c2.BASE / "no_sft_control_v1"
NO_SFT_SUMMARY = OUT / "no_sft_control_summary.json"
RESULT = OUT / "no_sft_vs_n100_bootstrap.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if RESULT.exists():
        raise RuntimeError(f"Result exists; refuse to overwrite: {RESULT}")
    c2.verify_frozen_inputs()

    no_sft_summary = read_json(NO_SFT_SUMMARY)
    if no_sft_summary.get("status") != "NO_SFT_CONTROL_READY":
        raise RuntimeError("No-SFT control is not frozen READY")
    predictions_record = no_sft_summary["predictions"]
    no_sft_path = c2.ROOT / predictions_record["path"]
    if c2.sha_file(no_sft_path) != predictions_record["sha256"]:
        raise RuntimeError("No-SFT prediction SHA mismatch")
    no_sft_all = pd.read_parquet(no_sft_path)

    results: list[dict[str, Any]] = []
    for task in c2.TASKS:
        no_sft_original = no_sft_all[no_sft_all.task.eq(task)].reset_index(drop=True)
        original_ids = no_sft_original.instance_id.astype(str).tolist()
        no_sft_hash = c2.sha_text("\n".join(original_ids))
        no_sft = no_sft_original.sort_values("instance_id").reset_index(drop=True)
        no_sft_ids = no_sft.instance_id.astype(str).tolist()
        task_summary = next(item for item in no_sft_summary["task_results"] if item["task"] == task)
        if no_sft_hash != task_summary["eval_membership"]["eval_instance_ids_sha256"]:
            raise RuntimeError(f"No-SFT eval membership mismatch: {task}")
        if set(no_sft.template_id.astype(str)) != {"TEMPLATE_0_PRIMARY"}:
            raise RuntimeError(f"No-SFT template mismatch: {task}")

        prediction_map: dict[tuple[int, int], pd.DataFrame] = {}
        n100_seed_nem: list[float] = []
        for seed in c2.SEEDS:
            run_dir = c2.RUNS / c2.run_name(task, 100, seed)
            frame = pd.read_parquet(run_dir / "predictions_primary.parquet").sort_values("instance_id").reset_index(drop=True)
            if frame.instance_id.astype(str).tolist() != no_sft_ids:
                raise RuntimeError(f"Paired eval IDs drift: {task} seed {seed}")
            if set(frame.template_id.astype(str)) != {"TEMPLATE_0_PRIMARY"}:
                raise RuntimeError(f"N100 template mismatch: {task} seed {seed}")
            saved = read_json(run_dir / "training_summary.json")
            metric = next(item for item in saved["metrics"] if item["template_id"] == "TEMPLATE_0_PRIMARY")
            recomputed = float(frame.correct_NEM.mean())
            if recomputed != float(metric["NEM"]):
                raise RuntimeError(f"N100 NEM drift: {task} seed {seed}")
            n100_seed_nem.append(recomputed)
            # Reuse c2.hierarchical_bootstrap unchanged: scale 0 is the
            # No-SFT candidate and scale 2000 is a label for the N100 reference.
            prediction_map[(0, seed)] = no_sft
            prediction_map[(2000, seed)] = frame

        raw = c2.hierarchical_bootstrap(task, 0, prediction_map)
        no_sft_nem = float(no_sft.correct_NEM.mean())
        n100_mean = float(sum(n100_seed_nem) / len(n100_seed_nem))
        delta = n100_mean - no_sft_nem
        if delta != float(raw["point_N2000_minus_N"]):
            raise RuntimeError(f"Bootstrap point estimate mismatch: {task}")
        low, high = float(raw["ci95_low"]), float(raw["ci95_high"])
        status = "N100_SFT_GREATER_95CI" if low > 0 else ("NO_SFT_GREATER_95CI" if high < 0 else "CI_INCLUDES_ZERO")
        results.append({
            "task": task,
            "eval_n": int(len(no_sft)),
            "no_sft_NEM": no_sft_nem,
            "n100_seed_NEM": n100_seed_nem,
            "n100_mean_NEM": n100_mean,
            "delta_N100_SFT_minus_No_SFT": delta,
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap_p_value": None,
            "bootstrap_p_value_status": "NOT_SUPPORTED_BY_FROZEN_C2_IMPLEMENTATION",
            "status": status,
            "protocol_check": "PASS",
            "eval_instance_ids_sha256": no_sft_hash,
            "bootstrap_replicates": int(raw["bootstrap_replicates"]),
            "bootstrap_seed": raw["bootstrap_seed"],
            "bootstrap_method": raw["method"],
        })

    payload = {
        "experiment": "corpus_v1_no_sft_vs_n100_bootstrap_v1",
        "comparison": "N100_SFT_MINUS_NO_SFT",
        "metric": "NEM",
        "bootstrap_replicates": c2.BOOTSTRAP,
        "implementation": {
            "function": "run_sft_c2_full_grid.hierarchical_bootstrap",
            "runner_sha256": c2.sha_file(Path(c2.__file__)),
            "adaptation": "No-SFT replicated as the untrained candidate across the outer seed draw; N100 predictions retain the three formal training seeds",
            "p_value": "not produced by the frozen C2 implementation",
        },
        "results": results,
        "protocol_checks": {
            "frozen_c2_inputs": "PASS",
            "no_sft_prediction_sha256": "PASS",
            "paired_eval_instance_ids": "PASS",
            "template_0_primary": "PASS",
            "nem_recomputed_matches_saved_metrics": "PASS",
            "predictions_regenerated": False,
            "model_loaded": False,
            "training_performed": False,
            "all_checks_pass": True,
        },
        "status": "NO_SFT_VS_N100_BOOTSTRAP_READY",
    }
    RESULT.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "result": str(RESULT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
