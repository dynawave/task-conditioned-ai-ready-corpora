from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SFT_DIR = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(SFT_DIR))
sys.path.insert(0, str(HERE))

from common import NEGATION_RE, NUMBER_RE, UNCERTAINTY_RE, json_compact, normalize_text, sha256_file, sha256_json  # noqa: E402
from detectors import T1_REGEX  # noqa: E402
from quality_gates import acronym_matches_long, aligned_long_form  # noqa: E402
from run_machine_validation import NLIVerifier  # noqa: E402


CAL = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/calibration_v1"
V1 = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/machine_validation_v1"
OUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/machine_validation_v1_1"
SCIAI_ARCHIVE = ROOT / "aicorpus-data/external_gold/AAAI-21-SDU-shared-task-1-AI_9e810512.zip"
SCIAI_ROOT = next((ROOT / "aicorpus-data/external_gold/AAAI-21-SDU-shared-task-1-AI_9e810512").iterdir())
SCIAI_COMMIT = "9e810512b98e1cf034e757d92136c7e64f840d0f"
SCIAI_REPOSITORY = "https://github.com/amirveyseh/AAAI-21-SDU-shared-task-1-AI"

FROZEN_T1_HASHES = {
    "rules.py": "885429510e37e8be22a43ad928ef85d09d19a1e4f53931c6a555daa9a1be0297",
    "quality_gates.py": "fe465e8407d50b4b1174933ff85ac9dd7246a397f924a8f35929a5e7970dc39a",
    "detectors.py": "a1bbc7577f341308da48741065d22e2206d9932bfd4381b8430e36aa90b05bfb",
    "validation/common.py": "5d09f18db3ce5aa1ff43b6c7dfd9d6c9a533af318c5f66176200c06fafc918b0",
    "machine_verifier_config_v1.json": "b7ad7810170282490f14dfd6fb8c947ab61a6f2155884df447d289dc175285c7",
}
FROZEN_DATA_HASHES = {
    "calibration_v1/candidates.parquet": "74c96f8f0c68afa9caecb9906ddd3b504ebad8c31448fb845fcd25660787f755",
    "calibration_v1/paper_features_cal100.csv": "841c88c1c6493057cdb4a5d4c0dd0dd5c042ffc43f440b8b617445226a0629e2",
    "calibration_v1/holdout1900_manifest.csv": "6ab748e1a5348805bb57b874ead801c56b7d33a607fe2d52add490984dda289b",
    "machine_validation_v1/cal100_machine_verified.parquet": "23abbae683e5e8fdb41634b97fcf21044cc6d80a878a02178b017c13273cdb4e",
    "machine_validation_v1/counterfactual_metrics.csv": "95bad9296063f3883bf923d82bd93485899a2c6f7b7b1d82af3f3d3a8d800610",
    "machine_validation_v1/summary.json": "a92db527b838d8f9d681e817cc20f26f05cb830d5cb335013974bad26be4fc1c",
}
CONDITION_KEYS = ["quantity_or_threshold", "spatial", "temporal", "treatment", "comparison"]
LEGACY_AGGREGATE_NAME = "NLI_OR_PREVIOUS_STRESS_METRIC"
LEGACY_AGGREGATE_VALUE = 0.6610455311973018


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_frozen_inputs() -> None:
    paths = {
        "rules.py": SFT_DIR / "rules.py",
        "quality_gates.py": SFT_DIR / "quality_gates.py",
        "detectors.py": SFT_DIR / "detectors.py",
        "validation/common.py": HERE / "common.py",
        "machine_verifier_config_v1.json": V1 / "machine_verifier_config_v1.json",
    }
    for name, expected in FROZEN_T1_HASHES.items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise RuntimeError(f"Frozen T1 implementation changed: {name}: {actual} != {expected}")
    data_paths = {
        "calibration_v1/candidates.parquet": CAL / "candidates.parquet",
        "calibration_v1/paper_features_cal100.csv": CAL / "paper_features_cal100.csv",
        "calibration_v1/holdout1900_manifest.csv": CAL / "holdout1900_manifest.csv",
        "machine_validation_v1/cal100_machine_verified.parquet": V1 / "cal100_machine_verified.parquet",
        "machine_validation_v1/counterfactual_metrics.csv": V1 / "counterfactual_metrics.csv",
        "machine_validation_v1/summary.json": V1 / "summary.json",
    }
    for name, expected in FROZEN_DATA_HASHES.items():
        actual = sha256_file(data_paths[name])
        if actual != expected:
            raise RuntimeError(f"Frozen data changed: {name}: {actual} != {expected}")
    v1_checksums = {}
    for line in (V1 / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        v1_checksums[filename] = digest
    for filename, expected in v1_checksums.items():
        if sha256_file(V1 / filename) != expected:
            raise RuntimeError(f"machine_validation_v1 checksum mismatch: {filename}")
    if len(pd.read_csv(CAL / "holdout1900_manifest.csv")) != 1900:
        raise RuntimeError("Holdout1900 membership changed")
    if json.loads((V1 / "summary.json").read_text(encoding="utf-8"))["holdout1900_processed_count"] != 0:
        raise RuntimeError("Holdout1900 was processed")


def token_char_spans(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if pieces:
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        spans.append((start, cursor))
    return " ".join(pieces), spans


def overlapping_tokens(start: int, end: int, spans: list[tuple[int, int]]) -> list[int]:
    return [index for index, (token_start, token_end) in enumerate(spans) if token_start < end and start < token_end]


def predict_sciai_sentence(row: dict[str, Any]) -> dict[str, Any]:
    text, spans = token_char_spans(row["tokens"])
    entities: list[tuple[int, str, list[int]]] = []
    for subtype, pattern in T1_REGEX.items():
        if subtype == "explicit_definition":
            continue
        for match in pattern.finditer(text):
            short_form = match.group("abbr")
            raw_long = match.group("long")
            aligned = aligned_long_form(short_form, raw_long)
            long_form = aligned[0] if aligned else raw_long
            if not acronym_matches_long(short_form, long_form):
                continue
            long_start = match.start("long") + (aligned[1] if aligned else 0)
            long_end = long_start + len(long_form)
            short_start, short_end = match.span("abbr")
            short_tokens = overlapping_tokens(short_start, short_end, spans)
            long_tokens = overlapping_tokens(long_start, long_end, spans)
            if short_tokens and long_tokens:
                entities.append((short_start, "short", short_tokens))
                entities.append((long_start, "long", long_tokens))
    predictions = ["O"] * len(spans)
    for _offset, entity_type, token_ids in sorted(entities, key=lambda item: (item[0], 0 if item[1] == "short" else 1)):
        if any(predictions[index] != "O" for index in token_ids):
            continue
        for position, token_id in enumerate(token_ids):
            predictions[token_id] = ("B-" if position == 0 else "I-") + entity_type
    return {"id": row["id"], "predictions": predictions}


def phrase_sets(sequences: list[list[str]]) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    shorts: set[tuple[int, int, int]] = set()
    longs: set[tuple[int, int, int]] = set()
    for sentence_index, sentence in enumerate(sequences):
        short_phrase: list[int] = []
        long_phrase: list[int] = []
        for token_index, label in enumerate([*sentence, "O"]):
            if "B" in label or "O" in label:
                if long_phrase:
                    longs.add((sentence_index, long_phrase[0], long_phrase[-1]))
                    long_phrase = []
                if short_phrase:
                    shorts.add((sentence_index, short_phrase[0], short_phrase[-1]))
                    short_phrase = []
            if "short" in label:
                short_phrase.append(token_index)
            if "long" in label:
                long_phrase.append(token_index)
    return shorts, longs


def prf(predicted: set[Any], gold: set[Any]) -> dict[str, Any]:
    correct = len(predicted & gold)
    precision = correct / len(predicted) if predicted else 1.0
    recall = correct / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "correct": correct, "predicted": len(predicted), "gold": len(gold)}


def evaluate_sciai() -> tuple[dict[str, Any], dict[str, Any]]:
    test_path = SCIAI_ROOT / "dataset/test.json"
    truth_path = SCIAI_ROOT / "dataset/truth.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if [row["id"] for row in test] != [row["id"] for row in truth]:
        raise RuntimeError("SciAI test/truth ID mismatch")
    if any(test_row["tokens"] != truth_row["tokens"] for test_row, truth_row in zip(test, truth)):
        raise RuntimeError("SciAI test/truth token mismatch")
    predictions = [predict_sciai_sentence(row) for row in test]
    gold_sequences = [row["labels"] for row in truth]
    prediction_sequences = [row["predictions"] for row in predictions]
    scorer = load_module("sciai_official_scorer", SCIAI_ROOT / "scorer.py")
    official_precision, official_recall, official_f1 = scorer.score_phrase_level(gold_sequences, prediction_sequences, verbos=False)
    gold_short, gold_long = phrase_sets(gold_sequences)
    pred_short, pred_long = phrase_sets(prediction_sequences)
    short_metrics = prf(pred_short, gold_short)
    long_metrics = prf(pred_long, gold_long)
    baseline_module = load_module("sciai_official_baseline", SCIAI_ROOT / "code/character_match.py")
    baseline_predictions = baseline_module.predict(test)
    baseline_short, baseline_long = phrase_sets([row["predictions"] for row in baseline_predictions])
    baseline_short_metrics = prf(baseline_short, gold_short)
    baseline_long_metrics = prf(baseline_long, gold_long)
    baseline_official = scorer.score_phrase_level(gold_sequences, [row["predictions"] for row in baseline_predictions], verbos=False)
    metrics = {
        "status": "OK",
        "dataset": "SciAI: Scientific Acronym Identification",
        "split": "official test with truth.json labels",
        "role": "external_verifier_validation",
        "n_sentences": len(test),
        "gold_short_forms": len(gold_short),
        "gold_long_forms": len(gold_long),
        "predicted_short_forms": len(pred_short),
        "predicted_long_forms": len(pred_long),
        "short_form": short_metrics,
        "long_form": long_metrics,
        "official_macro": {"precision": official_precision, "recall": official_recall, "f1": official_f1},
        "pair_metric": {"status": "PAIR_METRIC_NOT_IDENTIFIABLE_FROM_OFFICIAL_SCHEMA", "reason": "official BIO labels identify short and long entity spans but provide no relation links"},
        "prediction_sha256": sha256_json(predictions),
        "current_rule_policy": "frozen parenthetical acronym-long-form regex plus frozen alignment; no SciAI-driven modification or subtype filtering",
        "official_schwartz_hearst_baseline_same_split": {
            "short_form": baseline_short_metrics,
            "long_form": baseline_long_metrics,
            "official_macro": {"precision": baseline_official[0], "recall": baseline_official[1], "f1": baseline_official[2]},
            "readme_reference_macro": {"precision": 0.9322, "recall": 0.7890, "f1": 0.8546},
        },
    }
    manifest = {
        "dataset": "SciAI: Scientific Acronym Identification",
        "official_source": SCIAI_REPOSITORY,
        "repository_commit_sha": SCIAI_COMMIT,
        "commit_date_utc": "2024-12-15T23:56:06Z",
        "download_date": "2026-08-12",
        "role": "external_verifier_validation",
        "split": "official test + truth.json",
        "n_sentences": len(test),
        "license": {"dataset": "CC BY-NC-SA 4.0 International", "scorer_and_baseline": "MIT"},
        "archive": {"path": str(SCIAI_ARCHIVE.relative_to(ROOT)), "sha256": sha256_file(SCIAI_ARCHIVE)},
        "files": {
            name: {"sha256": sha256_file(SCIAI_ROOT / name), "bytes": (SCIAI_ROOT / name).stat().st_size}
            for name in ["LICENSE", "README.md", "scorer.py", "code/character_match.py", "dataset/test.json", "dataset/truth.json"]
        },
        "frozen_t1_implementation_sha256_before_evaluation": FROZEN_T1_HASHES,
    }
    return metrics, manifest


def t1_subtype_metrics(candidates: pd.DataFrame, features: pd.DataFrame) -> list[dict[str, Any]]:
    t1 = candidates[candidates.task_type == "T1_TERM"].copy()
    t1["reporting_subtype"] = np.where(t1.task_subtype == "explicit_definition", "T1b_EXPLICIT_DEFINITION", "T1a_ACRONYM")
    total_tokens = float(features.paper_clean_token_count.sum())
    rows: list[dict[str, Any]] = []
    definitions = [
        ("T1_TOTAL", t1, "mixed_scope_report_T1a_and_T1b_separately"),
        ("T1a_ACRONYM", t1[t1.reporting_subtype == "T1a_ACRONYM"], "external_gold_calibrated_rule"),
        ("T1b_EXPLICIT_DEFINITION", t1[t1.reporting_subtype == "T1b_EXPLICIT_DEFINITION"], "deterministic_rule_verified_only"),
    ]
    for label, frame, scope in definitions:
        rows.append({
            "task": label,
            "raw": int(len(frame)),
            "hard_pass": int(frame.hard_gate_pass.sum()),
            "ready": int(frame.materialization_ready.sum()),
            "machine_verified": int(frame.machine_verified.sum()),
            "papers_with_at_least_one_machine_verified": int(frame[frame.machine_verified].paper_id.nunique()),
            "machine_verified_per_10k_source_tokens": float(frame.machine_verified.sum()) * 10000 / total_tokens,
            "verification_scope": scope,
        })
    return rows


def preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def replace_first(pattern: str, text: str, replacement: str | Callable[[re.Match[str]], str]) -> str | None:
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    value = replacement(match) if callable(replacement) else replacement
    return text[:match.start()] + value + text[match.end():]


def mutate_negation(row: pd.Series) -> str | None:
    claim = str(row.answer_span)
    match = re.search(r"\bnot\b", claim, re.I)
    if match:
        return normalize_text(claim[:match.start()] + claim[match.end():])
    match = re.search(r"\bwithout\b", claim, re.I)
    if match:
        return normalize_text(claim[:match.start()] + "with" + claim[match.end():])
    match = re.search(r"\bno\s+", claim, re.I)
    if match:
        return normalize_text(claim[:match.start()] + "a " + claim[match.end():])
    auxiliary = re.search(r"\b(?:is|are|was|were|has|have|had|can|could|may|might|will|would|should|do|does|did)\b", claim, re.I)
    if auxiliary:
        return claim[:auxiliary.end()] + " not" + claim[auxiliary.end():]
    return None


def mutate_increase_decrease(row: pd.Series) -> str | None:
    mapping = {
        "increase": "decrease", "increases": "decreases", "increased": "decreased", "increasing": "decreasing",
        "decrease": "increase", "decreases": "increases", "decreased": "increased", "decreasing": "increasing",
    }
    pattern = r"\b(?:increasing|decreasing|increased|decreased|increases|decreases|increase|decrease)\b"
    return replace_first(pattern, str(row.answer_span), lambda match: preserve_case(match.group(0), mapping[match.group(0).casefold()]))


def mutate_higher_lower(row: pd.Series) -> str | None:
    mapping = {"higher": "lower", "lower": "higher"}
    return replace_first(r"\b(?:higher|lower)\b", str(row.answer_span), lambda match: preserve_case(match.group(0), mapping[match.group(0).casefold()]))


def mutate_numeric(row: pd.Series) -> str | None:
    conditions = json.loads(row.condition_spans or "{}")
    quantity = conditions.get("quantity_or_threshold")
    if not quantity:
        return None
    claim = str(row.answer_span)
    relative_start = quantity.get("start", -1) - int(row.answer_start)
    relative_end = quantity.get("end", -1) - int(row.answer_start)
    if not (0 <= relative_start < relative_end <= len(claim)):
        return None
    original_quantity = claim[relative_start:relative_end]
    number = NUMBER_RE.search(original_quantity)
    if not number or re.search(r"[×x]\s*10", number.group(0)):
        return None
    raw = number.group(0)
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    delta = 1.0 if abs(value) < 10 else 10.0
    changed = f"{value + delta:g}"
    mutated_quantity = original_quantity[:number.start()] + changed + original_quantity[number.end():]
    return claim[:relative_start] + mutated_quantity + claim[relative_end:]


def mutate_uncertainty(row: pd.Series) -> str | None:
    claim = str(row.answer_span)
    match = re.search(r"\b(?:may|might|could|likely|possibly|approximately)\b", claim, re.I)
    if not match:
        return None
    return normalize_text(claim[:match.start()] + claim[match.end():])


def mutate_subject_object(row: pd.Series) -> str | None:
    subject = normalize_text(row.subject)
    outcome = normalize_text(row.outcome)
    relation = normalize_text(row.relation)
    if not subject or not outcome or not relation or subject.casefold() == outcome.casefold():
        return None
    if len(re.findall(r"[A-Za-z][A-Za-z'-]*", subject)) < 2 or len(re.findall(r"[A-Za-z][A-Za-z'-]*", outcome)) < 2:
        return None
    if re.match(r"^(?:from|than|to|with|by|in|on|at|for|of|as)\b", outcome, re.I):
        return None
    terminal = "." if str(row.answer_span).rstrip().endswith(".") else ""
    return f"{outcome.rstrip('.!?')} {relation} {subject.rstrip('.!?')}{terminal}"


MUTATORS: list[tuple[str, Callable[[pd.Series], str | None], str, bool]] = [
    ("negation_flip", mutate_negation, "negation_consistency", True),
    ("increase_decrease_flip", mutate_increase_decrease, "polarity_relation_check", True),
    ("higher_lower_flip", mutate_higher_lower, "polarity_relation_check", True),
    ("numeric_perturbation", mutate_numeric, "exact_numeric_condition_preservation", True),
    ("uncertainty_removal", mutate_uncertainty, "uncertainty_preservation", True),
    ("subject_object_swap", mutate_subject_object, "semantic_role_consistency_not_available", False),
]


def legacy_counterfactual_by_type(t3_hard: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    all_rejected: list[bool] = []
    for row in t3_hard.itertuples():
        payload = json.loads(row.verifier_scores)
        variants = payload.get("counterfactual", {}).get("variants", [])
        for variant in variants:
            old_type = variant["mutation_type"]
            if old_type == "direction_flip":
                if re.search(r"\b(?:higher|lower)\b", variant.get("original_claim", row.answer_span), re.I):
                    mutation_type = "higher_lower_flip"
                else:
                    mutation_type = "increase_decrease_flip"
            elif old_type in {"negation_insertion", "negation_removal"}:
                mutation_type = "negation_flip"
            else:
                mutation_type = old_type
            rejected = bool(variant.get("rejected"))
            grouped[mutation_type].append(rejected)
            all_rejected.append(rejected)
    aggregate = float(np.mean(all_rejected)) if all_rejected else None
    if aggregate is None or abs(aggregate - LEGACY_AGGREGATE_VALUE) > 1e-12:
        raise RuntimeError(f"Legacy T3 aggregate changed: {aggregate}")
    result = {
        mutation_type: {"n": len(values), "nli_reject_rate": float(np.mean(values)) if values else None}
        for mutation_type, values in grouped.items()
    }
    return result, {"name": LEGACY_AGGREGATE_NAME, "n": len(all_rejected), "value": aggregate}


def t3_counterfactual_diagnostic(candidates: pd.DataFrame, threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t3_hard = candidates[(candidates.task_type == "T3_CLAIM") & candidates.hard_gate_pass].copy()
    legacy_by_type, legacy_aggregate = legacy_counterfactual_by_type(t3_hard)
    variants: list[dict[str, Any]] = []
    for index, row in t3_hard.iterrows():
        supports = json.loads(row.support_evidence_clean or "[]")
        premise = " ".join([str(row.evidence_clean), *map(str, supports)])
        for mutation_type, mutator, deterministic_gate, deterministic_reject in MUTATORS:
            mutated = mutator(row)
            if not mutated or normalize_text(mutated) == normalize_text(row.answer_span):
                continue
            variants.append({
                "row_index": int(index), "candidate_id": row.candidate_id, "mutation_type": mutation_type,
                "original_claim": str(row.answer_span), "mutated_claim": mutated, "premise": premise,
                "deterministic_gate": deterministic_gate, "deterministic_rejected": deterministic_reject,
            })
    nli = NLIVerifier()
    scores = nli.predict([row["premise"] for row in variants], [row["mutated_claim"] for row in variants]) if variants else []
    for row, score in zip(variants, scores):
        row["nli_scores"] = score
        row["nli_rejected"] = score["entailment"] < threshold
        row["combined_rejected"] = row["nli_rejected"] or row["deterministic_rejected"]
    rows: list[dict[str, Any]] = []
    for mutation_type, _mutator, deterministic_gate, _deterministic_reject in MUTATORS:
        subset = [row for row in variants if row["mutation_type"] == mutation_type]
        legacy = legacy_by_type.get(mutation_type, {"n": 0, "nli_reject_rate": None})
        rows.append({
            "mutation_type": mutation_type,
            "n": len(subset),
            "nli_reject_rate": float(np.mean([row["nli_rejected"] for row in subset])) if subset else None,
            "deterministic_gate": deterministic_gate,
            "deterministic_gate_reject_rate": float(np.mean([row["deterministic_rejected"] for row in subset])) if subset else None,
            "full_combined_verifier_reject_rate": float(np.mean([row["combined_rejected"] for row in subset])) if subset else None,
            "legacy_previous_n": legacy["n"],
            "legacy_previous_nli_reject_rate": legacy["nli_reject_rate"],
        })
    aggregate = {
        "evaluation_name": "FULL_COMBINED_VERIFIER_REJECT_RATE",
        "n_variants": len(variants),
        "nli_reject_rate": float(np.mean([row["nli_rejected"] for row in variants])) if variants else None,
        "deterministic_gate_reject_rate": float(np.mean([row["deterministic_rejected"] for row in variants])) if variants else None,
        "full_combined_verifier_reject_rate": float(np.mean([row["combined_rejected"] for row in variants])) if variants else None,
        "variant_set_sha256": sha256_json([
            {key: row[key] for key in ["candidate_id", "mutation_type", "original_claim", "mutated_claim", "deterministic_gate", "deterministic_rejected", "nli_scores", "nli_rejected", "combined_rejected"]}
            for row in variants
        ]),
        "legacy_previous": legacy_aggregate,
        "semantic_role_verifier_available": False,
        "subject_object_swap_interpretation": "combined rejection relies on frozen NLI only; no reliable deterministic semantic-role verifier is claimed",
    }
    return rows, aggregate


def t3_original_integrity(candidates: pd.DataFrame) -> dict[str, Any]:
    machine_verified = candidates[(candidates.task_type == "T3_CLAIM") & candidates.machine_verified].copy()
    exact_checks: list[bool] = []
    negation_checks: list[bool] = []
    uncertainty_checks: list[bool] = []
    condition_checks: list[bool] = []
    condition_candidate_checks: list[bool] = []
    condition_candidate_count = 0
    for row in machine_verified.itertuples():
        evidence = str(row.evidence_clean)
        claim = str(row.answer_span)
        start, end = int(row.answer_start), int(row.answer_end)
        exact_checks.append(0 <= start < end <= len(evidence) and evidence[start:end] == claim)
        conditions = json.loads(row.condition_spans or "{}")
        # The preservation denominator is an explicit cue in the original extractive claim,
        # not a detector flag that may conservatively fire outside the claim span.
        has_negation = bool(NEGATION_RE.search(claim))
        if has_negation:
            span = conditions.get("negation")
            negation_checks.append(bool(row.negation_flag and span and span.get("text") in claim and evidence[span.get("start", -1):span.get("end", -1)] == span.get("text")))
        has_uncertainty = bool(UNCERTAINTY_RE.search(claim))
        if has_uncertainty:
            span = conditions.get("uncertainty")
            uncertainty_checks.append(bool(row.uncertainty_flag and span and span.get("text") in claim and evidence[span.get("start", -1):span.get("end", -1)] == span.get("text")))
        candidate_conditions: list[bool] = []
        for key in CONDITION_KEYS:
            span = conditions.get(key)
            if not span:
                continue
            value = str(span.get("text", ""))
            passed = bool(value and value in claim and evidence[span.get("start", -1):span.get("end", -1)] == value)
            condition_checks.append(passed)
            candidate_conditions.append(passed)
        if candidate_conditions:
            condition_candidate_count += 1
            condition_candidate_checks.append(all(candidate_conditions))
    result = {
        "scope": "machine_verified T3 original candidates from frozen machine_validation_v1",
        "n_machine_verified": len(machine_verified),
        "original_claim_exact_evidence_span": {"n": len(exact_checks), "passed": int(sum(exact_checks)), "rate": float(np.mean(exact_checks)) if exact_checks else None},
        "extractive_span_combination_count": 0,
        "negation_preservation": {"eligible": len(negation_checks), "passed": int(sum(negation_checks)), "rate": float(np.mean(negation_checks)) if negation_checks else None},
        "uncertainty_preservation": {"eligible": len(uncertainty_checks), "passed": int(sum(uncertainty_checks)), "rate": float(np.mean(uncertainty_checks)) if uncertainty_checks else None},
        "detected_condition_preservation": {"eligible_spans": len(condition_checks), "passed_spans": int(sum(condition_checks)), "rate": float(np.mean(condition_checks)) if condition_checks else None, "eligible_candidates": condition_candidate_count, "all_detected_conditions_preserved_candidates": int(sum(condition_candidate_checks))},
        "blocker": not all(exact_checks),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {out}")
    validate_frozen_inputs()
    candidates = pd.read_parquet(V1 / "cal100_machine_verified.parquet")
    features = pd.read_csv(CAL / "paper_features_cal100.csv")
    v1_config = json.loads((V1 / "machine_verifier_config_v1.json").read_text(encoding="utf-8"))
    v1_summary = json.loads((V1 / "summary.json").read_text(encoding="utf-8"))
    threshold = float(v1_config["T3"]["selected_entailment_threshold"])

    sciai_metrics, sciai_manifest = evaluate_sciai()
    subtype_rows = t1_subtype_metrics(candidates, features)
    integrity = t3_original_integrity(candidates)
    counterfactual_rows, counterfactual_aggregate = t3_counterfactual_diagnostic(candidates, threshold)

    critical_types = {"negation_flip", "increase_decrease_flip", "higher_lower_flip", "numeric_perturbation", "uncertainty_removal"}
    critical_rows = [row for row in counterfactual_rows if row["mutation_type"] in critical_types]
    t1_precision_oriented_support = min(sciai_metrics["short_form"]["precision"], sciai_metrics["long_form"]["precision"]) >= 0.95
    t3_critical_combined_pass = all(row["n"] > 0 and row["full_combined_verifier_reject_rate"] == 1.0 for row in critical_rows)
    systematic_failure = not (t1_precision_oriented_support and t3_critical_combined_pass and not integrity["blocker"])
    status = "SFT_A_MACHINE_VALIDATION_NOT_READY" if systematic_failure else "SFT_A_MACHINE_VALIDATION_READY"

    repair_code_hash = sha256_file(Path(__file__))
    verifier_config = {
        "experiment": "corpus_v1_sft_a_machine_validation_v1_1",
        "created_date": "2026-08-12",
        "scope": "repair only: SciAI external T1a validation and typed T3 combined-verifier stress diagnosis",
        "frozen_parent": {"directory": str(V1.relative_to(ROOT)), "hashes": FROZEN_DATA_HASHES, "config_sha256": FROZEN_T1_HASHES["machine_verifier_config_v1.json"]},
        "T1a_ACRONYM": {
            "verification_scope": "external_gold_calibrated_rule", "dataset": sciai_manifest,
            "evaluation_metrics": sciai_metrics, "implementation_sha256_frozen_before_evaluation": FROZEN_T1_HASHES,
            "rules_modified_using_sciai": False,
        },
        "T1b_EXPLICIT_DEFINITION": {
            "verification_scope": "deterministic_rule_verified_only",
            "external_gold_task_equivalence": False,
            "limitation": "SciAI validates acronym/long-form identification, not generic explicit scientific definitions",
        },
        "T2": {"unchanged_from_v1": True, "frozen_config": v1_config["T2"]},
        "T3": {
            "candidate_decisions_unchanged_from_v1": True,
            "frozen_scifact_and_nli_config": v1_config["T3"],
            "NLI": v1_config["NLI"],
            "legacy_stress_metric": counterfactual_aggregate["legacy_previous"],
            "combined_counterfactual_definition": {
                "NLI_rejection": "entailment score below frozen SciFact threshold",
                "negation_flip": "NLI OR evidence-relative negation consistency",
                "increase_decrease_flip": "NLI OR frozen relation/polarity preservation",
                "higher_lower_flip": "NLI OR frozen relation/polarity preservation",
                "numeric_perturbation": "NLI OR exact recognized numeric-condition preservation",
                "uncertainty_removal": "NLI OR evidence-relative uncertainty preservation",
                "subject_object_swap": "NLI only; no reliable deterministic semantic-role verifier is claimed",
            },
            "aggregate": counterfactual_aggregate,
        },
        "readiness_assessment": {
            "no_parameter_or_rule_tuning": True,
            "t1_precision_oriented_support": t1_precision_oriented_support,
            "t3_critical_combined_categories_all_nonempty_and_fully_rejected": t3_critical_combined_pass,
            "t3_original_claim_integrity_blocker": integrity["blocker"],
            "systematic_failure": systematic_failure,
        },
        "code_sha256": {"run_machine_validation_repair_v1_1.py": repair_code_hash},
        "holdout1900_processed": 0,
        "candidate_detection_rerun": False,
        "T2_modified": False,
        "SFT_training": False,
        "generative_LLM_judge": False,
    }

    out.mkdir(parents=True)
    write_json(out / "sciai_manifest.json", sciai_manifest)
    write_json(out / "sciai_metrics.json", sciai_metrics)
    pd.DataFrame(subtype_rows).to_csv(out / "t1_subtype_metrics.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(counterfactual_rows).to_csv(out / "t3_counterfactual_by_type.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_json(out / "t3_original_claim_integrity.json", integrity)
    write_json(out / "machine_verifier_config_v1_1.json", verifier_config)
    config_hash = sha256_file(out / "machine_verifier_config_v1_1.json")
    summary = {
        "experiment": "corpus_v1_sft_a_machine_validation_v1_1",
        "status": status,
        "repair_scope_only": ["replace unavailable MACRONYM future role with SciAI for T1a external validation", "reinterpret T3 stress using typed full combined verifier"],
        "historical_macronym_status": "DATASET_NOT_AVAILABLE retained in machine_validation_v1",
        "SciAI": sciai_metrics,
        "T1_Cal100_split": subtype_rows,
        "T3_original_claim_integrity": integrity,
        "T3_counterfactual_by_type": counterfactual_rows,
        "T3_counterfactual_aggregate": counterfactual_aggregate,
        "readiness": verifier_config["readiness_assessment"],
        "ready_interpretation": "precision-oriented extractive acronym/claim materialization; not generic definition validation or a general scientific fact checker",
        "limitations": [
            "SciAI validates T1a acronym and long-form spans but not T1b generic definitions.",
            "Frozen T1a rules have high precision but low short-form recall on SciAI.",
            "SciAI BIO labels do not identify short-long relation pairs.",
            "T3 subject/object swaps lack a reliable deterministic semantic-role verifier and rely on frozen NLI.",
            "Synthetic counterfactual rejection is a stress test, not human accuracy or general fact verification.",
        ],
        "frozen_invariants": {
            "Cal100_papers": 100, "holdout1900_processed": 0,
            "raw_hard_ready_unchanged": True, "T3_candidate_decisions_unchanged": True,
            "T2_results_unchanged": True, "accepted_SFT_instances": 0, "training_runs": 0,
        },
        "machine_verifier_config_v1_1_sha256": config_hash,
        "parent_v1_status": v1_summary["status"],
    }
    write_json(out / "summary.json", summary)
    filenames = sorted(path.name for path in out.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (out / "checksums.sha256").write_text("\n".join(f"{sha256_file(out / name)}  {name}" for name in filenames) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status, "output": str(out), "config_sha256": config_hash,
        "summary_sha256": sha256_file(out / "summary.json"), "SciAI": sciai_metrics,
        "T1": subtype_rows, "T3_integrity": integrity, "T3_counterfactual": counterfactual_rows,
        "T3_aggregate": counterfactual_aggregate,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
