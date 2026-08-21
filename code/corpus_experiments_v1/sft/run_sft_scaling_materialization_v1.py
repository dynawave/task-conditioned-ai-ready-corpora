from __future__ import annotations

import hashlib
import json
import os
import numbers
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / "validation"))

from common import independent_acronym_alignment  # noqa: E402


LABELS_DIR = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_b_holdout1900_v1_2"
CAL_DIR = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/calibration_v1"
OUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_scaling_v1/materialization"
LABELS = LABELS_DIR / "holdout_candidate_labels_v1_2.parquet"
PAPER_METRICS = LABELS_DIR / "paper_task_metrics.parquet"
HOLDOUT_MANIFEST = CAL_DIR / "holdout1900_manifest.csv"
CAL100_MANIFEST = CAL_DIR / "cal100_manifest.csv"
MODEL = Path(os.environ.get("QWEN3B_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"))

SPLIT_SEED = "20260812_SFT_SCALING_V1"
INSTANCE_ORDER_SEED = "20260812_SFT_SCALE_V1"
TEMPLATE_VERSION = "sft_scaling_extractive_v1"
TASKS = ("T1a_ACRONYM", "T2_NUMERIC", "T3_CLAIM")
SCALES = (100, 250, 500, 1000, 2000)
MAX_CONTEXT_TOKENS = 1024
MAX_INSTANCES_PER_PAPER = 3
MAX_EVAL_PER_TASK = 300
EXPECTED = {
    "holdout_manifest": "6ab748e1a5348805bb57b874ead801c56b7d33a607fe2d52add490984dda289b",
    "holdout_membership": "e4066ba08a65e69055eb795f07b7b97715930dcce0f21a48819af35fe4541b4f",
    "labels_v1_2": "d766b8b356a6c102255d0aed25697adca229c66c04d068c3da986b25f7502446",
    "verifier_v1_2": "26a035bf6e9bf7d63d0d401def58d4ddee740b6fe6057925aa368fe50e3b1bd0",
    "canonical": "2c72b3615d5299534d6336c9d4a4b4db0982853011718babeabc2aafa18e007b",
    "model_config": "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b",
    "tokenizer_bundle": "badd01fb58846dadbe2b7ce2a21a2e8e2e7257c87bf971f5689be8f4c87fa786",
}
T3_INSTRUCTION = (
    "Extract the scientific finding stated in the passage.\n"
    "Preserve the wording, conditions, polarity, negation,\n"
    "and uncertainty expressed in the evidence."
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_json(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_audit(value: Any) -> str:
    return re.sub(r"[^a-z0-9%<>=+./-]+", " ", normalize_text(value).casefold()).strip()


def membership_sha(values: list[str]) -> str:
    return sha256_text("\n".join(values))


def task_order_score(task: str, candidate_id: str) -> str:
    return sha256_text(f"{INSTANCE_ORDER_SEED}|{task}|{candidate_id}")


def paper_order_score(task: str, paper_id: str) -> str:
    return sha256_text(f"{INSTANCE_ORDER_SEED}|{task}|paper|{paper_id}")


def instance_id(candidate_id: str) -> str:
    return f"sfti_{sha256_text(TEMPLATE_VERSION + '|' + candidate_id)[:24]}"


def valid_frozen_span(row: pd.Series) -> bool:
    context = str(row.evidence_clean)
    answer = str(row.answer_span)
    start, end = row.answer_start, row.answer_end
    return (
        isinstance(start, numbers.Integral)
        and isinstance(end, numbers.Integral)
        and 0 <= start < end <= len(context)
        and context[start:end] == answer
    )


def tokenizer_file_hashes() -> dict[str, str]:
    files = ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")
    return {name: sha256_file(MODEL / name) for name in files}


def split_sources(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = manifest[["source_id", "paper_id"]].copy()
    split["split_score_sha256"] = split.source_id.map(lambda x: sha256_text(f"{SPLIT_SEED}|{x}"))
    split = split.sort_values(["split_score_sha256", "source_id"], kind="mergesort").reset_index(drop=True)
    split["split_ordinal"] = range(1, len(split) + 1)
    split["split_seed"] = SPLIT_SEED
    split["train_or_eval"] = ["train"] * 1500 + ["eval"] * 400
    return split.iloc[:1500].copy(), split.iloc[1500:].copy()


def token_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [tuple(x) for x in encoded["offset_mapping"]]


def crop_to_required_spans(
    tokenizer: Any, context: str, required: list[tuple[int, int]], max_tokens: int = MAX_CONTEXT_TOKENS
) -> tuple[str | None, bool]:
    token_ids, offsets = token_offsets(tokenizer, context)
    if len(token_ids) <= max_tokens:
        return context, False
    if not required or any(start < 0 or end <= start or end > len(context) for start, end in required):
        return None, False
    req_start = min(start for start, _ in required)
    req_end = max(end for _, end in required)
    covered = [i for i, (start, end) in enumerate(offsets) if end > req_start and start < req_end]
    if not covered:
        return None, False
    left_required, right_required = min(covered), max(covered)
    if right_required - left_required + 1 > max_tokens:
        return None, False
    spare = max_tokens - (right_required - left_required + 1)
    left = max(0, left_required - spare // 2)
    right = min(len(offsets), right_required + 1 + (spare - (left_required - left)))
    left = max(0, right - max_tokens)
    char_start = offsets[left][0]
    char_end = offsets[right - 1][1]
    cropped = context[char_start:char_end]
    for start, end in required:
        if context[start:end] not in cropped:
            return None, False
    if len(tokenizer(cropped, add_special_tokens=False)["input_ids"]) > max_tokens:
        return None, False
    return cropped, True


def nearest_term_span(context: str, term: str, answer_start: int) -> tuple[int, int] | None:
    matches = list(re.finditer(re.escape(term), context))
    if not matches:
        matches = list(re.finditer(re.escape(term), context, re.I))
    if not matches:
        return None
    match = min(matches, key=lambda x: (abs(x.start() - answer_start), x.start()))
    return match.start(), match.end()


def t2_prefix_anchor(row: pd.Series) -> tuple[str, int, int] | None:
    context = str(row.evidence_clean)
    start = int(row.answer_start)
    anchor_start = max(0, int(row.anchor_start))
    sentence_prefix = context[anchor_start:start]
    words = list(WORD_RE.finditer(sentence_prefix))
    if len(words) < 4:
        return None
    # A fixed 8-word suffix is a locatable lexical target binding, not generated prose.
    selected = words[-8:]
    char_start = anchor_start + selected[0].start()
    char_end = anchor_start + selected[-1].end()
    anchor = context[char_start:char_end]
    if str(row.answer_span) in anchor or len(list(WORD_RE.finditer(anchor))) < 4:
        return None
    if context.count(anchor) != 1:
        return None
    return anchor, char_start, char_end


def t3_integrity(row: pd.Series) -> bool:
    if not valid_frozen_span(row):
        return False
    scores = parse_json(row.verifier_scores, {})
    deterministic = scores.get("deterministic") or {}
    if deterministic.get("condition_keys_checked") is None:
        return False
    conditions = parse_json(row.condition_spans, {})
    answer = str(row.answer_span)
    for key in ("quantity_or_threshold", "spatial", "temporal", "treatment", "comparison"):
        span = conditions.get(key)
        if span and str(span.get("text", "")) not in answer:
            return False
    if bool(deterministic.get("negation_present")):
        span = conditions.get("negation")
        if not bool(row.negation_flag) or not span or str(span.get("text", "")) not in answer:
            return False
    if bool(deterministic.get("uncertainty_present")):
        span = conditions.get("uncertainty")
        if not bool(row.uncertainty_flag) or not span or str(span.get("text", "")) not in answer:
            return False
    return True


def base_instance(row: pd.Series, task: str, instruction: str, context: str, response: str) -> dict[str, Any]:
    return {
        "instance_id": instance_id(str(row.candidate_id)),
        "candidate_id": str(row.candidate_id),
        "task": task,
        "task_subtype": str(row.task_subtype),
        "paper_id": str(row.paper_id),
        "source_id": str(row.source_id),
        "section_id": str(row.section_id),
        "paragraph_id": str(row.paragraph_id),
        "instruction": instruction,
        "context": context,
        "response": response,
        "response_normalized": str(row.answer_normalized),
        "page": row.page,
        "bbox": str(row.bbox),
        "provenance": str(row.provenance),
        "verification_scope": str(row.verification_scope_v1_2),
        "unit": None if pd.isna(row.unit) else str(row.unit),
        "negation_flag": bool(row.negation_flag),
        "uncertainty_flag": bool(row.uncertainty_flag),
        "context_normalized_audit": normalize_audit(context),
        "response_normalized_audit": normalize_audit(response),
        "target_traceable": response in context,
        "order_score_sha256": task_order_score(task, str(row.candidate_id)),
    }


def materialize(
    candidates: pd.DataFrame,
    source_tokens: dict[str, int | float],
    split_by_source: dict[str, str],
    tokenizer: Any,
) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, int]]:
    mv = candidates[candidates.machine_verified_v1_2.astype(bool) & candidates.report_task.isin(TASKS)].copy()
    t1_context = defaultdict(lambda: defaultdict(set))
    t2_hard_count = candidates[
        candidates.hard_gate_pass.astype(bool) & (candidates.report_task == "T2_NUMERIC")
    ].groupby(["source_id", "paragraph_id"]).size().to_dict()
    t3_mv_count = mv[mv.report_task == "T3_CLAIM"].groupby(["source_id", "paragraph_id"]).size().to_dict()
    for row in mv[mv.report_task == "T1a_ACRONYM"].itertuples(index=False):
        key = (str(row.source_id), str(row.paragraph_id), normalize_text(row.evidence_clean))
        t1_context[key][normalize_text(row.term).casefold()].add(normalize_text(row.definition).casefold())

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    crop_count: Counter[str] = Counter()
    for row in mv.sort_values("candidate_id", kind="mergesort").itertuples(index=False):
        row = pd.Series(row._asdict())
        task = str(row.report_task)
        context = str(row.evidence_clean)
        response = str(row.answer_span)
        reason: str | None = None
        required: list[tuple[int, int]] = [(int(row.answer_start), int(row.answer_end))]
        instruction = ""
        if not valid_frozen_span(row):
            reason = "FROZEN_TARGET_SPAN_INVALID"
        elif task == "T1a_ACRONYM":
            term = str(row.term)
            definition = str(row.definition)
            key = (str(row.source_id), str(row.paragraph_id), normalize_text(row.evidence_clean))
            if len(t1_context[key][normalize_text(term).casefold()]) > 1:
                reason = "AMBIGUOUS_ACRONYM_CONTEXT"
            elif term not in context or definition not in context or response != definition:
                reason = "T1A_FROZEN_RELATION_NOT_TRACEABLE"
            elif not independent_acronym_alignment(term, definition):
                reason = "T1A_ALIGNMENT_REGRESSION_FAILED"
            else:
                term_span = nearest_term_span(context, term, int(row.answer_start))
                if term_span is None:
                    reason = "T1A_FROZEN_RELATION_NOT_TRACEABLE"
                else:
                    required.append(term_span)
                    instruction = f"What does {term} stand for in the passage?"
        elif task == "T2_NUMERIC":
            group_key = (str(row.source_id), str(row.paragraph_id))
            if t2_hard_count.get(group_key, 0) != 1:
                reason = "NO_UNAMBIGUOUS_NUMERIC_INSTRUCTION"
            else:
                anchor = t2_prefix_anchor(row)
                if anchor is None:
                    reason = "NO_UNAMBIGUOUS_NUMERIC_INSTRUCTION"
                else:
                    anchor_text, anchor_start, anchor_end = anchor
                    required.append((anchor_start, anchor_end))
                    instruction = (
                        f'According to the passage, what value is reported immediately after "{anchor_text}"?'
                    )
        else:
            group_key = (str(row.source_id), str(row.paragraph_id))
            if t3_mv_count.get(group_key, 0) != 1:
                reason = "MULTIPLE_VALID_CLAIMS_IN_CONTEXT"
            elif not t3_integrity(row):
                reason = "T3_FROZEN_INTEGRITY_REGRESSION_FAILED"
            else:
                supports = [str(x) for x in parse_json(row.support_evidence_clean, []) if str(x)]
                if supports:
                    prefix = "Evidence paragraph:\n"
                    support_prefix = "\n\nSupporting evidence:\n"
                    context = prefix + context + support_prefix + "\n\n".join(supports)
                    response_start = len("Evidence paragraph:\n") + int(row.answer_start)
                    required = [(response_start, response_start + len(response))]
                    cursor = len(prefix) + len(str(row.evidence_clean)) + len(support_prefix)
                    for support in supports:
                        required.append((cursor, cursor + len(support)))
                        cursor += len(support) + 2
                instruction = T3_INSTRUCTION

        if reason is None:
            cropped, did_crop = crop_to_required_spans(tokenizer, context, required)
            if cropped is None:
                reason = "CONTEXT_TRUNCATION_INVALID"
            else:
                context = cropped
                crop_count[task] += int(did_crop)
                if response not in context or (task == "T2_NUMERIC" and instruction.split('"')[1] not in context):
                    reason = "CONTEXT_TRUNCATION_INVALID"
        if reason is not None:
            exclusions.append({"candidate_id": str(row.candidate_id), "task": task, "reason": reason})
            continue

        item = base_instance(row, task, instruction, context, response)
        item["source_tokens"] = source_tokens[str(row.source_id)]
        item["train_or_eval"] = split_by_source[str(row.source_id)]
        item["context_tokens"] = len(tokenizer(context, add_special_tokens=False)["input_ids"])
        user_content = instruction + "\n\nPassage:\n" + context
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], tokenize=True, add_generation_prompt=True
        )
        serialized = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}, {"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        )
        item["prompt_tokens"] = len(prompt)
        item["response_tokens"] = len(tokenizer(response, add_special_tokens=False)["input_ids"])
        item["serialized_tokens"] = len(serialized)
        rows.append(item)

    data = pd.DataFrame(rows)
    exact_key = ["train_or_eval", "instruction", "context", "response"]
    data = data.sort_values(["train_or_eval", "order_score_sha256", "candidate_id"], kind="mergesort")
    duplicate = data.duplicated(exact_key, keep="first")
    for candidate_id, task in data.loc[duplicate, ["candidate_id", "task"]].itertuples(index=False):
        exclusions.append({"candidate_id": candidate_id, "task": task, "reason": "EXACT_INSTANCE_DUPLICATE_WITHIN_SPLIT"})
    data = data.loc[~duplicate].copy().reset_index(drop=True)
    return data, exclusions, dict(crop_count)


def train_master(data: pd.DataFrame, task: str) -> pd.DataFrame:
    subset = data[(data.train_or_eval == "train") & (data.task == task)].copy()
    subset = subset.sort_values(["paper_id", "order_score_sha256", "candidate_id"], kind="mergesort")
    grouped = {paper: group.to_dict("records") for paper, group in subset.groupby("paper_id", sort=False)}
    papers = sorted(grouped, key=lambda paper: (paper_order_score(task, paper), paper))
    ordered: list[dict[str, Any]] = []
    for round_index in range(MAX_INSTANCES_PER_PAPER):
        for paper in papers:
            if len(grouped[paper]) > round_index:
                ordered.append(grouped[paper][round_index])
    result = pd.DataFrame(ordered[:2000])
    if not result.empty:
        result.insert(0, "master_ordinal", range(1, len(result) + 1))
    return result


def eval_set(data: pd.DataFrame, task: str) -> pd.DataFrame:
    subset = data[(data.train_or_eval == "eval") & (data.task == task)].copy()
    subset = subset.sort_values(["paper_id", "order_score_sha256", "candidate_id"], kind="mergesort")
    subset = subset.drop_duplicates("paper_id", keep="first")
    subset["eval_order_score_sha256"] = subset.paper_id.map(lambda x: paper_order_score(task, x))
    subset = subset.sort_values(["eval_order_score_sha256", "paper_id"], kind="mergesort").head(MAX_EVAL_PER_TASK)
    subset.insert(0, "eval_ordinal", range(1, len(subset) + 1))
    return subset


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing formal output directory: {OUT}")
    if sha256_file(HOLDOUT_MANIFEST) != EXPECTED["holdout_manifest"]:
        raise RuntimeError("Holdout manifest hash mismatch")
    if sha256_file(LABELS) != EXPECTED["labels_v1_2"]:
        raise RuntimeError("v1.2 label hash mismatch")
    if sha256_file(LABELS_DIR / "machine_verifier_config_v1_2.json") != EXPECTED["verifier_v1_2"]:
        raise RuntimeError("v1.2 verifier hash mismatch")
    if sha256_file(MODEL / "config.json") != EXPECTED["model_config"]:
        raise RuntimeError("Base model config hash mismatch")

    manifest = pd.read_csv(HOLDOUT_MANIFEST)
    cal_ids = set(pd.read_csv(CAL100_MANIFEST).source_id.astype(str))
    holdout_ids = manifest.source_id.astype(str).tolist()
    if len(holdout_ids) != 1900 or len(set(holdout_ids)) != 1900:
        raise RuntimeError("Holdout manifest does not contain 1,900 unique sources")
    if membership_sha(holdout_ids) != EXPECTED["holdout_membership"]:
        raise RuntimeError("Holdout membership hash mismatch")
    if set(holdout_ids) & cal_ids:
        raise RuntimeError("Cal100 overlaps Holdout1900")

    train_manifest, eval_manifest = split_sources(manifest)
    split_by_source = dict(
        zip(
            pd.concat([train_manifest, eval_manifest]).source_id.astype(str),
            pd.concat([train_manifest, eval_manifest]).train_or_eval,
        )
    )
    if set(train_manifest.source_id) & set(eval_manifest.source_id):
        raise RuntimeError("Train/eval source overlap")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    tokenizer_hashes = tokenizer_file_hashes()
    metrics = pd.read_parquet(PAPER_METRICS)
    source_token_rows = metrics.drop_duplicates("source_id")[["source_id", "source_tokens"]]
    source_tokens = dict(zip(source_token_rows.source_id.astype(str), source_token_rows.source_tokens))
    candidates = pd.read_parquet(LABELS)
    materialized, exclusions, crop_count = materialize(candidates, source_tokens, split_by_source, tokenizer)

    masters = {task: train_master(materialized, task) for task in TASKS}
    evals = {task: eval_set(materialized, task) for task in TASKS}
    nested_rows: list[dict[str, Any]] = []
    nested_summary: list[dict[str, Any]] = []
    strict_nested: dict[str, bool] = {}
    for task, master in masters.items():
        prior: set[str] = set()
        ok = True
        for scale in SCALES:
            current = set(master.head(scale).instance_id.astype(str)) if len(master) >= scale else set()
            if scale != SCALES[0] and not prior < current:
                ok = False
            if len(current) != scale:
                ok = False
            if len(master) >= scale:
                for iid in master.head(scale).instance_id.astype(str):
                    nested_rows.append({"task": task, "scale": scale, "instance_id": iid})
                pool = master.head(scale)
                nested_summary.append({
                    "task": task,
                    "scale": scale,
                    "instances": int(len(pool)),
                    "unique_papers": int(pool.paper_id.nunique()),
                    "serialized_tokens": int(pool.serialized_tokens.sum()),
                    "task_subtype_composition": json.dumps(
                        dict(sorted(Counter(pool.task_subtype).items())), ensure_ascii=False, sort_keys=True
                    ),
                })
            else:
                nested_summary.append({
                    "task": task, "scale": scale, "instances": int(len(master)),
                    "unique_papers": int(master.paper_id.nunique()) if len(master) else 0,
                    "serialized_tokens": int(master.serialized_tokens.sum()) if len(master) else 0,
                    "task_subtype_composition": json.dumps(
                        dict(sorted(Counter(master.task_subtype).items())) if len(master) else {}, sort_keys=True
                    ),
                })
            prior = current
        strict_nested[task] = ok

    mv_counts = (
        candidates[candidates.machine_verified_v1_2.astype(bool)]
        .groupby("report_task").size().to_dict()
    )
    exclusion_counts = Counter((x["task"], x["reason"]) for x in exclusions)
    materialization_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_data = materialized[materialized.task == task]
        materialization_rows.append({
            "task": task,
            "mv_source_candidates": int(mv_counts.get(task, 0)),
            "scaling_materializable": int(len(task_data)),
            "train_eligible": int((task_data.train_or_eval == "train").sum()),
            "eval_eligible": int((task_data.train_or_eval == "eval").sum()),
            "context_cropped": int(crop_count.get(task, 0)),
        })
        for (reason_task, reason), count in sorted(exclusion_counts.items()):
            if reason_task == task:
                materialization_rows.append({
                    "task": task, "mv_source_candidates": None, "scaling_materializable": None,
                    "train_eligible": None, "eval_eligible": None, "context_cropped": None,
                    "exclusion_reason": reason, "exclusion_count": int(count),
                })

    train_papers = set(train_manifest.paper_id.astype(str))
    eval_papers = set(eval_manifest.paper_id.astype(str))
    all_master = pd.concat(masters.values(), ignore_index=True)
    all_eval = pd.concat(evals.values(), ignore_index=True)
    exact_cols = ["instruction", "context", "response"]
    train_exact = set(map(tuple, all_master[exact_cols].astype(str).to_numpy()))
    eval_exact = set(map(tuple, all_eval[exact_cols].astype(str).to_numpy()))
    normalized_cross_paper_duplicates = int(
        materialized.groupby(["instruction", "context_normalized_audit", "response_normalized_audit"])
        .paper_id.nunique().gt(1).sum()
    )
    exact_cross_paper_duplicates = int(
        materialized.groupby(exact_cols).paper_id.nunique().gt(1).sum()
    )

    gates = {
        "train_eval_paper_overlap_zero": len(train_papers & eval_papers) == 0,
        "cal100_included_zero": not bool((set(materialized.source_id) & cal_ids)),
        "all_instances_provenance_100_percent": bool(materialized.provenance.astype(str).str.len().gt(2).all()),
        "all_targets_traceable_100_percent": bool(materialized.target_traceable.all()),
        "t1a_ambiguity_zero": exclusion_counts[("T1a_ACRONYM", "AMBIGUOUS_ACRONYM_CONTEXT")] >= 0
        and not bool(materialized[materialized.task == "T1a_ACRONYM"].duplicated(["context", "instruction"], keep=False).any()),
        "t2_instruction_ambiguity_zero": not bool(materialized[materialized.task == "T2_NUMERIC"].duplicated(["context", "instruction"], keep=False).any()),
        "t3_multi_target_ambiguity_zero": not bool(materialized[materialized.task == "T3_CLAIM"].duplicated(["context", "instruction"], keep=False).any()),
        "train_eval_candidate_overlap_zero": not bool(set(all_master.candidate_id) & set(all_eval.candidate_id)),
        "train_eval_exact_overlap_zero": len(train_exact & eval_exact) == 0,
        "all_primary_tasks_have_nested_2000": all(strict_nested.values()),
        "all_eval_sets_use_300_or_all_eligible_papers": all(
            len(evals[task])
            == min(
                MAX_EVAL_PER_TASK,
                materialized[(materialized.train_or_eval == "eval") & (materialized.task == task)].paper_id.nunique(),
            )
            and len(evals[task]) > 0
            for task in TASKS
        ),
    }
    status = "SFT_SCALING_MATERIALIZATION_READY" if all(gates.values()) else "SFT_SCALING_MATERIALIZATION_NOT_READY"

    OUT.mkdir(parents=True)
    train_manifest.to_csv(OUT / "sft_train1500_manifest.csv", index=False)
    eval_manifest.to_csv(OUT / "sft_eval400_manifest.csv", index=False)
    materialized.to_parquet(OUT / "materialized_instances.parquet", index=False)
    file_stem = {"T1a_ACRONYM": "t1a", "T2_NUMERIC": "t2", "T3_CLAIM": "t3"}
    for task in TASKS:
        masters[task].to_parquet(OUT / f"{file_stem[task]}_master_train_pool.parquet", index=False)
        evals[task].to_parquet(OUT / f"{file_stem[task]}_eval.parquet", index=False)
    pd.DataFrame(nested_rows).to_csv(OUT / "nested_pool_manifest.csv", index=False)
    pd.DataFrame(materialization_rows).to_csv(OUT / "materialization_summary.csv", index=False)

    config = {
        "experiment": "corpus_v1_sft_c0_deterministic_materialization_v1",
        "template_version": TEMPLATE_VERSION,
        "primary_scaling_tasks": list(TASKS),
        "t1b_exclusion": {
            "included_in_primary_scaling": False,
            "reasons": [
                "machine-verified pool is substantially smaller",
                "cannot uniformly support the 2,000-instance upper scale",
                "external validation scope is weaker than T1a",
            ],
        },
        "split": {"seed": SPLIT_SEED, "method": "SHA256(seed|source_id), ascending, first 1500 train, last 400 eval"},
        "ordering": {
            "seed": INSTANCE_ORDER_SEED,
            "candidate": "SHA256(seed|task|candidate_id)",
            "paper": "SHA256(seed|task|paper|paper_id)",
            "method": "paper round-robin; at most 3 instances per paper in N2000",
        },
        "materialization_rules": {
            "T1a_ACRONYM": "exact short-long relation; reject same-context same-short distinct-long ambiguity",
            "T2_NUMERIC": "same-paragraph unique hard-pass numeric target plus unique 8-word lexical prefix anchor",
            "T3_CLAIM": "same-paragraph unique MV claim; include all frozen support evidence intact",
            "dedup": "exact instruction+context+response within split, keep lowest stable order",
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "context_crop": "deterministic target/anchor-centered token window; no partial required support",
        },
        "templates": {
            "T1a_ACRONYM": "What does {SHORT_FORM} stand for in the passage?",
            "T2_NUMERIC": "According to the passage, what value is reported immediately after {ANCHOR}?",
            "T3_CLAIM": T3_INSTRUCTION,
        },
        "base_model": {
            "repository_id": "Qwen/Qwen2.5-3B-Instruct",
            "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "snapshot": str(MODEL),
            "dtype": "float16",
            "quantization": "none",
            "config_sha256": EXPECTED["model_config"],
            "tokenizer_bundle_sha256": EXPECTED["tokenizer_bundle"],
            "tokenizer_files": tokenizer_hashes,
            "local_files_only": True,
        },
        "frozen_inputs": {
            "holdout_manifest_sha256": EXPECTED["holdout_manifest"],
            "holdout_membership_sha256": EXPECTED["holdout_membership"],
            "candidate_labels_v1_2_sha256": EXPECTED["labels_v1_2"],
            "machine_verifier_v1_2_sha256": EXPECTED["verifier_v1_2"],
            "canonical_sha256": EXPECTED["canonical"],
        },
        "prohibited_actions_performed": {
            "candidate_detector_rerun": False, "machine_verifier_modified": False,
            "llm_question_generation": False, "llm_judge": False,
            "base_evaluation": False, "training": False,
        },
    }
    write_json(OUT / "config.json", config)
    summary = {
        "status": status,
        "split": {
            "train_papers": len(train_manifest), "eval_papers": len(eval_manifest),
            "paper_overlap": len(train_papers & eval_papers),
            "train_manifest_sha256": sha256_file(OUT / "sft_train1500_manifest.csv"),
            "eval_manifest_sha256": sha256_file(OUT / "sft_eval400_manifest.csv"),
            "cal100_included": int(len(set(materialized.source_id) & cal_ids)),
        },
        "materialization": materialization_rows,
        "nested_pools": nested_summary,
        "strict_nested": strict_nested,
        "eval": {
            task: {
                "n": int(len(frame)), "unique_papers": int(frame.paper_id.nunique()),
                "target_traceability": float(frame.target_traceable.mean()) if len(frame) else None,
                "train_eval_paper_overlap": int(len(set(masters[task].paper_id) & set(frame.paper_id))),
                "train_eval_candidate_overlap": int(len(set(masters[task].candidate_id) & set(frame.candidate_id))),
            }
            for task, frame in evals.items()
        },
        "duplicate_audit": {
            "exact_cross_paper_content_groups": exact_cross_paper_duplicates,
            "normalized_cross_paper_content_groups": normalized_cross_paper_duplicates,
            "train_eval_exact_content_overlap": len(train_exact & eval_exact),
        },
        "gates": gates,
        "warnings": [],
        "blockers": [key for key, passed in gates.items() if not passed],
    }
    write_json(OUT / "summary.json", summary)

    output_files = sorted(path for path in OUT.iterdir() if path.name != "checksums.sha256")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in output_files]
    (OUT / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(OUT), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
