from __future__ import annotations

"""SFT-C3 post-grid scale extension; C2 is strictly read-only."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))
from corpus_experiments_v1.sft import run_sft_c2_full_grid as c2  # noqa: E402


BASE = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1"
MATERIALIZATION = BASE / "materialization"
C2_OUT = BASE / "full_grid_v1"
OUT = BASE / "scale_extension_v1"
RUNS = OUT / "runs"
TASKS = ("T2_NUMERIC", "T3_CLAIM")
SEEDS = c2.SEEDS
MAX_PER_PAPER = 3
ORDER_SEED = "20260812_SFT_SCALE_V1"
DELTA_NEM = c2.DELTA_NEM
BOOTSTRAP = c2.BOOTSTRAP


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


def stem(task: str) -> str:
    return c2.STEMS[task]


def run_name(task: str, scale: int, seed: int) -> str:
    return f"{stem(task)}_n{scale}_seed{seed}"


def paper_score(task: str, paper_id: str) -> str:
    return sha_text(f"{ORDER_SEED}|{task}|paper|{paper_id}")


def extended_master(task: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_parquet(MATERIALIZATION / "materialized_instances.parquet")
    subset = data[(data.train_or_eval == "train") & (data.task == task)].copy()
    subset = subset.sort_values(["paper_id", "order_score_sha256", "candidate_id"], kind="mergesort")
    groups = {paper: group.to_dict("records") for paper, group in subset.groupby("paper_id", sort=False)}
    papers = sorted(groups, key=lambda paper: (paper_score(task, str(paper)), str(paper)))
    ordered: list[dict[str, Any]] = []
    round_counts: list[int] = []
    for round_index in range(MAX_PER_PAPER):
        picks = [groups[paper][round_index] for paper in papers if len(groups[paper]) > round_index]
        ordered.extend(picks)
        round_counts.append(len(picks))
    pool = pd.DataFrame(ordered)
    pool.insert(0, "extension_master_ordinal", range(1, len(pool) + 1))
    old = pd.read_parquet(MATERIALIZATION / f"{stem(task)}_master_train_pool.parquet")
    prefix_equal = pool.head(2000).instance_id.astype(str).tolist() == old.instance_id.astype(str).tolist()
    if not prefix_equal:
        raise RuntimeError(f"C2 N2000 prefix mismatch for {task}")
    if pool.instance_id.astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate extension instance IDs for {task}")
    if int(pool.paper_id.value_counts().max()) > MAX_PER_PAPER:
        raise RuntimeError(f"Per-paper diversity cap violated for {task}")
    details = {
        "task": task,
        "train_materialized_instances": int(len(subset)),
        "unique_train_papers": int(len(papers)),
        "round_counts": round_counts,
        "max_instances_per_paper": MAX_PER_PAPER,
        "n_max_feasible": int(len(pool)),
        "c2_n2000_prefix_exact_match": prefix_equal,
        "extension_membership_sha256": sha_text("\n".join(pool.instance_id.astype(str))),
    }
    return pool, details


def extension_scales(capacities: dict[str, int]) -> dict[str, tuple[int, ...]]:
    t2_second = 4000 if capacities["T2_NUMERIC"] >= 4000 else capacities["T2_NUMERIC"]
    t2 = tuple(dict.fromkeys((3000, t2_second)))
    t3_cap = capacities["T3_CLAIM"]
    t3 = (t3_cap,) if t3_cap <= 2500 else tuple(dict.fromkeys((2500, t3_cap)))
    if any(scale <= 2000 for scales in (t2, t3) for scale in scales):
        raise RuntimeError("Extension scale must be above C2 N2000")
    return {"T2_NUMERIC": t2, "T3_CLAIM": t3}


def frozen_capacity() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, tuple[int, ...]]]:
    pools: dict[str, pd.DataFrame] = {}
    details: dict[str, Any] = {}
    for task in TASKS:
        pools[task], details[task] = extended_master(task)
    capacities = {task: int(details[task]["n_max_feasible"]) for task in TASKS}
    scales = extension_scales(capacities)
    for task, values in scales.items():
        if max(values) > capacities[task]:
            raise RuntimeError(f"Capacity exceeded for {task}")
    return pools, details, scales


def verify_inputs() -> None:
    c2.verify_frozen_inputs()
    # C2's manifest contents are frozen and valid; tolerate a UTF-8 BOM on
    # its first line when reading it so C3 never rewrites the C2 directory.
    manifest = C2_OUT / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split("  ", 1)
        if not (C2_OUT / relative).is_file() or sha_file(C2_OUT / relative) != expected:
            raise RuntimeError(f"Frozen C2 checksum mismatch: {relative}")
    c2_summary = read_json(C2_OUT / "summary.json")
    if c2_summary.get("status") != "SFT_FULL_GRID_READY" or c2_summary.get("complete_runs") != 45:
        raise RuntimeError("Frozen C2 is not complete/ready")


def prepare_output() -> None:
    verify_inputs()
    pools, details, scales = frozen_capacity()
    capacity = {
        "experiment": "corpus_v1_sft_c3_post_grid_capacity_v1",
        "capacity_decision_is_data_only": True,
        "source": "C0 materialized_instances.parquet with original stable paper round-robin and MAX_INSTANCES_PER_PAPER=3",
        "ordering_seed": ORDER_SEED,
        "tasks": details,
        "selected_extension_scales": {task: list(values) for task, values in scales.items()},
        "secondary_template_scales": {task: [max(values)] for task, values in scales.items()},
    }
    config = {
        "experiment": "corpus_v1_sft_c3_post_grid_scale_extension",
        "status": "frozen_after_capacity_before_training",
        "tasks": list(TASKS),
        "extension_scales": {task: list(values) for task, values in scales.items()},
        "training_seeds": list(SEEDS),
        "secondary_template_max_scales": {task: max(values) for task, values in scales.items()},
        "capacity_summary_sha256": sha_text(json.dumps(capacity, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
        "c2_manifest_sha256": sha_file(C2_OUT / "checksums.sha256"),
        "materialization_manifest_sha256": sha_file(MATERIALIZATION / "checksums.sha256"),
        "formal_scale_protocol_sha256": c2.FORMAL_PROTOCOL_SHA,
        "template_config_sha256": c2.TEMPLATE_CONFIG_SHA,
        "c1_training_protocol_sha256": c2.PROTOCOL_SHA,
        "base_model": read_json(MATERIALIZATION / "config.json")["base_model"],
        "delta_nem": DELTA_NEM,
        "bootstrap_replicates": BOOTSTRAP,
        "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha_file(Path(__file__))},
        "c2_read_only": True,
    }
    if OUT.exists():
        if not (OUT / "config.json").is_file() or read_json(OUT / "config.json") != config:
            raise RuntimeError("Existing C3 directory/config differs; refuse to overwrite")
        return
    OUT.mkdir(parents=True)
    RUNS.mkdir()
    write_json(OUT / "capacity_summary.json", capacity)
    write_json(OUT / "config.json", config)
    shutil.copy2(BASE / "instruction_robustness_v1" / "formal_scale_protocol_v1.json", OUT / "formal_scale_protocol_v1.json")
    manifest_rows: list[pd.DataFrame] = []
    for task, pool in pools.items():
        values = scales[task]
        pool.head(max(values)).to_parquet(OUT / f"{stem(task)}_extension_master_pool.parquet", index=False)
        for scale in values:
            row = pool.head(scale)[["extension_master_ordinal", "instance_id", "candidate_id", "paper_id"]].copy()
            row.insert(0, "scale", scale)
            row.insert(0, "task", task)
            manifest_rows.append(row)
    pd.concat(manifest_rows, ignore_index=True).to_csv(OUT / "extension_pool_manifest.csv", index=False)


def load_sets(tokenizer: Any) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, pd.DataFrame]]:
    pools, _details, scales = frozen_capacity()
    labels = pd.read_parquet(c2.c1.LABELS, columns=["candidate_id", "condition_spans", "verifier_scores", "task_subtype"])
    label_map = {str(row.candidate_id): row._asdict() for row in labels.itertuples(index=False)}
    trains: dict[str, dict[int, pd.DataFrame]] = {}
    evals: dict[str, pd.DataFrame] = {}
    for task in TASKS:
        evaluation = pd.read_parquet(MATERIALIZATION / f"{stem(task)}_eval.parquet")
        trains[task] = {}
        frames: list[pd.DataFrame] = [evaluation]
        for scale in scales[task]:
            subset = pools[task].head(scale).copy().reset_index(drop=True)
            if set(pools[task].head(2000).instance_id.astype(str)) - set(subset.instance_id.astype(str)):
                raise RuntimeError(f"Nested C2 prefix violation: {task} N{scale}")
            trains[task][scale] = subset
            frames.append(subset)
        for frame in frames:
            frame["frozen_condition_spans"] = frame.candidate_id.astype(str).map(lambda x: label_map[x]["condition_spans"])
            frame["frozen_verifier_scores"] = frame.candidate_id.astype(str).map(lambda x: label_map[x]["verifier_scores"])
            frame["task_subtype"] = frame.candidate_id.astype(str).map(lambda x: label_map[x]["task_subtype"])
            serial = [c2.c1.serialize(row, tokenizer) for _, row in frame.iterrows()]
            frame["prompt_token_count"] = [len(x[0]) for x in serial]
            frame["target_token_count"] = [len(x[1]) for x in serial]
            frame["sequence_token_count"] = [len(x[0]) + len(x[1]) for x in serial]
            frame["target_truncated"] = frame.sequence_token_count > c2.c1.MAX_SEQUENCE_LENGTH
            if bool(frame.target_truncated.any()):
                raise RuntimeError(f"Target truncation under frozen protocol: {task}")
        if set(pools[task].paper_id.astype(str)) & set(evaluation.paper_id.astype(str)):
            raise RuntimeError(f"Train/eval paper leakage: {task}")
        evals[task] = evaluation
    return trains, evals


def run_one(task: str, scale: int, seed: int) -> None:
    prepare_output()
    pools, _details, scales = frozen_capacity()
    if scale not in scales[task]:
        raise RuntimeError("Scale not selected by frozen capacity rule")
    directory = RUNS / run_name(task, scale, seed)
    required = [directory / "training_summary.json", directory / "run_config.json", directory / "predictions_primary.parquet", directory / "adapter" / "adapter_model.safetensors"]
    if all(path.is_file() for path in required):
        if read_json(directory / "training_summary.json").get("status") != "completed":
            raise RuntimeError(f"Run marked non-complete: {directory.name}")
        return
    if directory.exists():
        raise RuntimeError(f"Incomplete prior run preserved for audit; explicit technical retry required: {directory.name}")
    directory.mkdir(parents=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(c2.c1.MODEL, local_files_only=True, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        trains, evals = load_sets(tokenizer)
        train_frame, eval_frame = trains[task][scale], evals[task]
        rows, masking_checks = c2.c1.prepare_training_rows(train_frame, tokenizer)
        masking = c2.c1.masking_unit_test(rows)
        rng = c2.configure_seed(seed)
        model = c2.c1.attach_lora(c2.c1.load_base())
        training, order_hashes = c2.train_with_audit(model, tokenizer, rows, seed)
        if not training["finite_loss"] or not masking["passed"]:
            raise RuntimeError("Formal integrity gate failed during training")
        adapter = directory / "adapter"
        model.save_pretrained(adapter)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError("Saved adapter missing")
        del model
        c2.torch.cuda.empty_cache()
        reloaded = c2.PeftModel.from_pretrained(c2.c1.load_base(), adapter).eval()
        primary, primary_eval = c2.c1.evaluate(reloaded, tokenizer, eval_frame, task, directory.name)
        primary = c2.annotate_predictions(primary, task, scale, seed, "TEMPLATE_0_PRIMARY")
        primary.to_parquet(directory / "predictions_primary.parquet", index=False)
        metrics = [c2.metric_row(primary, primary_eval, task, scale, seed, "TEMPLATE_0_PRIMARY")]
        if scale == max(scales[task]):
            for template_id in ("TEMPLATE_1", "TEMPLATE_2"):
                altered = eval_frame.copy()
                altered["instruction"] = [c2.rendered_instruction(task, template_id, item) for item in eval_frame.instruction.astype(str)]
                predictions, evaluation = c2.c1.evaluate(reloaded, tokenizer, altered, task, f"{directory.name}_{template_id}")
                predictions = c2.annotate_predictions(predictions, task, scale, seed, template_id)
                predictions.to_parquet(directory / f"predictions_{template_id.lower()}.parquet", index=False)
                metrics.append(c2.metric_row(predictions, evaluation, task, scale, seed, template_id))
        del reloaded
        c2.torch.cuda.empty_cache()
        instance_ids = train_frame.instance_id.astype(str).tolist()
        run_config = {
            "run": directory.name, "task": task, "scale": scale, "seed": seed,
            "formal_scale_protocol_sha256": c2.FORMAL_PROTOCOL_SHA, "template_config_sha256": c2.TEMPLATE_CONFIG_SHA,
            "c1_training_protocol_sha256": c2.PROTOCOL_SHA, "base_model": read_json(MATERIALIZATION / "config.json")["base_model"],
            "lora": c2.c1.LORA, "training": c2.c1.TRAINING,
            "training_instance_ids": instance_ids, "training_membership_sha256": sha_text("\n".join(instance_ids)),
            "unique_paper_ids": sorted(train_frame.paper_id.astype(str).unique().tolist()), "unique_paper_count": int(train_frame.paper_id.nunique()),
            "training_serialized_tokens_one_epoch": int(train_frame.sequence_token_count.sum()),
            "training_tokens_all_epochs_expected": int(train_frame.sequence_token_count.sum() * c2.c1.TRAINING["epochs"]),
            "eval_instance_ids_sha256": sha_text("\n".join(eval_frame.instance_id.astype(str))), "eval_instances": int(len(eval_frame)),
            "train_eval_paper_overlap": int(len(set(train_frame.paper_id.astype(str)) & set(eval_frame.paper_id.astype(str)))),
            "target_truncation_count": int(train_frame.target_truncated.sum()), "fresh_base_lora": True,
        }
        write_json(directory / "run_config.json", run_config)
        write_json(directory / "training_summary.json", {
            "status": "completed", "run": directory.name, "loss_masking_unit_test": masking, "masking_checks": masking_checks,
            "epoch_order_sha256": order_hashes, "rng_audit": rng, "environment": c2.environment(), "training": training,
            "adapter_sha256": c2.c1.aggregate_adapter_sha(adapter), "adapter_fresh_base_reload": True,
            "eval_target_truncation_count": 0, "metrics": metrics,
        })
    except Exception as error:
        write_json(directory / "technical_failure.json", {"run": directory.name, "error_type": type(error).__name__, "error": str(error), "technical_retry": False})
        raise


def expected_runs() -> list[tuple[str, int, int]]:
    _pools, _details, scales = frozen_capacity()
    return [(task, scale, seed) for task in TASKS for scale in scales[task] for seed in SEEDS]


def bootstrap(task: str, candidate: int, reference: int, prediction_map: dict[tuple[int, int], pd.DataFrame]) -> dict[str, Any]:
    arrays: dict[tuple[int, int], np.ndarray] = {}
    ids: list[str] | None = None
    for scale in (candidate, reference):
        for seed in SEEDS:
            frame = prediction_map[(scale, seed)].sort_values("instance_id").reset_index(drop=True)
            current = frame.instance_id.astype(str).tolist()
            if ids is None:
                ids = current
            elif current != ids:
                raise RuntimeError(f"Paired Eval ID drift: {task} N{candidate}/N{reference} seed {seed}")
            arrays[(scale, seed)] = frame.correct_NEM.to_numpy(dtype=float)
    size = len(ids or [])
    rng = np.random.default_rng(int(sha_text(f"C3_BOOTSTRAP_V1|{task}|{candidate}|{reference}")[:16], 16))
    values = np.empty(BOOTSTRAP, dtype=float)
    for index in range(BOOTSTRAP):
        drawn = rng.integers(0, len(SEEDS), size=len(SEEDS))
        components: list[float] = []
        for seed_index in drawn:
            seed = SEEDS[int(seed_index)]
            example_indices = rng.integers(0, size, size=size)
            components.append(float((arrays[(reference, seed)][example_indices] - arrays[(candidate, seed)][example_indices]).mean()))
        values[index] = float(np.mean(components))
    observed = float(np.mean([arrays[(reference, seed)].mean() - arrays[(candidate, seed)].mean() for seed in SEEDS]))
    return {"task": task, "scale": candidate, "reference_scale": reference, "mean_NEM_difference_reference_minus_scale": observed,
            "ci95_low": c2.quantile(values, 0.025), "ci95_high": c2.quantile(values, 0.975), "one_sided_ucb95": c2.quantile(values, 0.95),
            "delta_nem": DELTA_NEM, "practically_equivalent": bool(c2.quantile(values, 0.95) <= DELTA_NEM),
            "bootstrap_replicates": BOOTSTRAP, "bootstrap_seed": "SHA256(C3_BOOTSTRAP_V1|task|scale|reference)",
            "method": "outer resampling of three training seeds; inner paired resampling of frozen Eval examples"}


def record_from_run(directory: Path) -> dict[str, Any]:
    summary = read_json(directory / "training_summary.json")
    row = dict(next(item for item in summary["metrics"] if item["template_id"] == "TEMPLATE_0_PRIMARY"))
    row.update({"training_tokens": summary["training"]["tokens_seen"], "runtime_seconds": summary["training"]["elapsed_seconds"], "final_loss": summary["training"]["loss_final"], "peak_gpu_memory_bytes": summary["training"]["peak_gpu_memory_bytes"], "optimizer_steps": summary["training"]["optimizer_steps"], "adapter_sha256": summary["adapter_sha256"], "source_experiment": "C3"})
    return row


def finalize() -> None:
    prepare_output()
    pools, details, scales = frozen_capacity()
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    prediction_maps: dict[str, dict[tuple[int, int], pd.DataFrame]] = {task: {} for task in TASKS}
    for task in TASKS:
        for scale in (100, 250, 500, 1000, 2000):
            for seed in SEEDS:
                directory = C2_OUT / "runs" / run_name(task, scale, seed)
                if not (directory / "training_summary.json").is_file() or not (directory / "predictions_primary.parquet").is_file():
                    errors.append(f"c2_missing:{directory.name}")
                    continue
                summary = read_json(directory / "training_summary.json")
                if summary.get("status") != "completed": errors.append(f"c2_pipeline:{directory.name}")
                row = dict(next(item for item in summary["metrics"] if item["template_id"] == "TEMPLATE_0_PRIMARY"))
                row.update({"training_tokens": summary["training"]["tokens_seen"], "runtime_seconds": summary["training"]["elapsed_seconds"], "final_loss": summary["training"]["loss_final"], "peak_gpu_memory_bytes": summary["training"]["peak_gpu_memory_bytes"], "optimizer_steps": summary["training"]["optimizer_steps"], "adapter_sha256": summary["adapter_sha256"], "source_experiment": "C2"})
                records.append(row); prediction_maps[task][(scale, seed)] = pd.read_parquet(directory / "predictions_primary.parquet")
    for task, scale, seed in expected_runs():
        directory = RUNS / run_name(task, scale, seed)
        required = [directory / "training_summary.json", directory / "run_config.json", directory / "predictions_primary.parquet", directory / "adapter" / "adapter_model.safetensors"]
        if not all(path.is_file() for path in required):
            errors.append(f"c3_missing:{directory.name}"); continue
        config, summary = read_json(directory / "run_config.json"), read_json(directory / "training_summary.json")
        if summary.get("status") != "completed" or not summary["loss_masking_unit_test"].get("passed") or not summary["training"].get("finite_loss"): errors.append(f"c3_pipeline:{directory.name}")
        if config["training_membership_sha256"] != sha_text("\n".join(config["training_instance_ids"])) or config["train_eval_paper_overlap"] != 0 or config["target_truncation_count"] != 0 or summary["eval_target_truncation_count"] != 0: errors.append(f"c3_integrity:{directory.name}")
        if set(summary["epoch_order_sha256"]) != {"epoch_1_order_sha256", "epoch_2_order_sha256", "epoch_3_order_sha256"}: errors.append(f"c3_order:{directory.name}")
        if scale == max(scales[task]) and {x["template_id"] for x in summary["metrics"]} != {"TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2"}: errors.append(f"c3_templates:{directory.name}")
        records.append(record_from_run(directory)); prediction_maps[task][(scale, seed)] = pd.read_parquet(directory / "predictions_primary.parquet")
    if errors:
        write_json(OUT / "summary.json", {"status": "SFT_SCALE_EXTENSION_NOT_READY", "integrity_errors": errors})
        raise RuntimeError("C3 completeness gate failed: " + ", ".join(errors[:5]))
    seed_metrics = pd.DataFrame(records).sort_values(["task", "scale", "seed"])
    seed_metrics.to_csv(OUT / "combined_seed_metrics.csv", index=False)
    scale_rows: list[dict[str, Any]] = []
    for (task, scale), group in seed_metrics.groupby(["task", "scale"], sort=True):
        scale_rows.append({"task": task, "scale": int(scale), "NEM_mean": float(group.NEM.mean()), "NEM_sd": float(group.NEM.std(ddof=1)), "NEM_min": float(group.NEM.min()), "NEM_max": float(group.NEM.max()), "NEM_seed_values": canonical([float(x) for x in group.sort_values("seed").NEM]), "EM_mean": float(group.EM.mean()), "training_tokens_per_run": int(group.training_tokens.iloc[0]), "runtime_seconds_mean": float(group.runtime_seconds.mean()), "final_loss_mean": float(group.final_loss.mean())})
    scale_metrics = pd.DataFrame(scale_rows).sort_values(["task", "scale"])
    scale_metrics.to_csv(OUT / "combined_scale_metrics.csv", index=False)
    scale_metrics[["task", "scale", "NEM_mean", "NEM_sd", "NEM_min", "NEM_max"]].to_csv(OUT / "combined_learning_curve.csv", index=False)
    boot_rows: list[dict[str, Any]] = []
    nstar_rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    for task in TASKS:
        reference = max(scales[task])
        candidates = [int(x) for x in scale_metrics[scale_metrics.task == task].scale.tolist() if int(x) != reference]
        boot_rows.extend(bootstrap(task, candidate, reference, prediction_maps[task]) for candidate in candidates)
        group = pd.DataFrame([row for row in boot_rows if row["task"] == task]).sort_values("scale")
        eligible = group[group.practically_equivalent]
        nstar_rows.append({"task": task, "n_star": int(eligible.scale.iloc[0]) if len(eligible) else reference, "n_star_status": "identified" if len(eligible) else "REFERENCE_ONLY_NOT_IDENTIFIED", "reference_scale": reference, "delta_nem": DELTA_NEM})
        means = scale_metrics[scale_metrics.task == task].set_index("scale").NEM_mean
        previous = max(int(value) for value in means.index if int(value) < reference)
        difference = float(means.loc[reference] - means.loc[previous])
        status = "REFERENCE_NOT_PLATEAUED" if difference > DELTA_NEM else "REFERENCE_PLATEAU_COMPATIBLE"
        plateau_rows.append({"task": task, "largest_scale": reference, "previous_scale": previous, "largest_minus_previous": difference, "plateau_status": status, "capacity_limited": reference == int(details[task]["n_max_feasible"]), "conclusion": "PLATEAU_NOT_IDENTIFIED_WITHIN_AVAILABLE_POOL" if status == "REFERENCE_NOT_PLATEAUED" and reference == int(details[task]["n_max_feasible"]) else ""})
    pd.DataFrame(boot_rows).sort_values(["task", "scale"]).to_csv(OUT / "practical_equivalence.csv", index=False)
    pd.DataFrame(nstar_rows).to_csv(OUT / "n_star_summary.csv", index=False)
    pd.DataFrame(plateau_rows).to_csv(OUT / "plateau_summary.csv", index=False)
    template_rows: list[dict[str, Any]] = []
    for task in TASKS:
        scale = max(scales[task]); metrics: list[dict[str, Any]] = []
        for seed in SEEDS: metrics.extend(read_json(RUNS / run_name(task, scale, seed) / "training_summary.json")["metrics"])
        table = pd.DataFrame(metrics).groupby("template_id").NEM.mean()
        values = [float(table[item]) for item in ("TEMPLATE_0_PRIMARY", "TEMPLATE_1", "TEMPLATE_2")]
        template_rows.append({"task": task, "scale": scale, "primary_mean_NEM": values[0], "paraphrase_mean_NEM": float(np.mean(values[1:])), "worst_template_mean_NEM": float(min(values)), "template_sd": float(np.std(values, ddof=1)), "template0_NEM": values[0], "template1_NEM": values[1], "template2_NEM": values[2]})
    pd.DataFrame(template_rows).to_csv(OUT / "template_robustness_summary.csv", index=False)
    seed_metrics.groupby(["task", "scale"], as_index=False).agg(training_tokens=("training_tokens", "first"), runtime_seconds_mean=("runtime_seconds", "mean"), runtime_seconds_sd=("runtime_seconds", "std"), peak_gpu_memory_bytes_max=("peak_gpu_memory_bytes", "max"), optimizer_steps=("optimizer_steps", "first"), final_loss_mean=("final_loss", "mean")).to_csv(OUT / "training_cost_summary.csv", index=False)
    write_json(OUT / "summary.json", {"status": "SFT_SCALE_EXTENSION_READY", "c2_read_only": True, "c2_formal_runs_reused": 30, "c3_expected_new_runs": len(expected_runs()), "c3_complete_new_runs": len(expected_runs()), "capacity": details, "extension_scales": {task: list(value) for task, value in scales.items()}, "integrity": {"all_frozen_inputs_verified": True, "all_new_runs_fresh_lora": True, "all_new_runs_masking_pass": True, "all_new_runs_finite_loss": True, "all_new_runs_order_auditable": True, "all_new_train_eval_paper_overlap_zero": True, "all_new_target_truncation_zero": True, "all_new_primary_predictions_present": True}, "delta_nem": DELTA_NEM, "bootstrap_replicates": BOOTSTRAP, "n_star": nstar_rows, "plateau": plateau_rows, "warnings": ["TRAINING_STOCHASTICITY_WARNING", "SOURCE_PROVENANCE_AUDIT_WARNING", "FORMAT_METRIC_WARNING_T2"], "technical_retries": []})
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("\n".join(f"{sha_file(path)}  {path.relative_to(OUT).as_posix()}" for path in files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("prepare", "run-one", "run-all", "final"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == "prepare": prepare_output(); print(canonical({"prepared": str(OUT)}))
    elif args.phase == "run-one":
        if args.task is None or args.scale is None or args.seed is None: raise RuntimeError("run-one requires task, scale, seed")
        run_one(args.task, args.scale, args.seed); print(canonical({"completed": run_name(args.task, args.scale, args.seed)}))
    elif args.phase == "run-all":
        for task, scale, seed in expected_runs(): run_one(task, scale, seed); print(canonical({"completed": run_name(task, scale, seed)}), flush=True)
    else: finalize(); print(canonical({"status": "SFT_SCALE_EXTENSION_READY"}))


if __name__ == "__main__": main()
