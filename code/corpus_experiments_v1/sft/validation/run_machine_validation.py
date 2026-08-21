from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    NEGATION_RE,
    NUMBER_RE,
    UNCERTAINTY_RE,
    binary_metrics,
    distribution,
    exact_or_normalized_in,
    independent_acronym_alignment,
    is_strict_numeric_answer,
    json_compact,
    normalize_match,
    normalize_text,
    numeric_signature,
    select_f05_threshold,
    sha256_file,
    sha256_json,
    stable_id,
)


ROOT = Path(__file__).resolve().parents[4]
CAL = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/calibration_v1"
OUT = ROOT / "aicorpus-derived/experiments/corpus_v1/sft_a/machine_validation_v1"
QASPER = ROOT / "aicorpus-data/public_benchmarks/qasper-test-and-evaluator-v0.3.tgz"
SCITAT = ROOT / "aicorpus-data/external_gold/SciTaT-main/dataset/scitat_test.json"
SCIFACT = ROOT / "aicorpus-data/external_gold/scifact_official_latest_data.tar.gz"
MODEL = ROOT / "aicorpus-data/external_models/DeBERTa-v3-large-mnli-fever-anli-ling-wanli_b3546ea"
MODEL_REPO = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MODEL_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
EXPECTED_INPUT_HASHES = {
    "audit_sample.csv": "80570c9623e206e30bd9b6d05e579f1396e514f2a9f66292ceceb3d835d67aef",
    "cal100_manifest.csv": "08a7b637c95aad7f6a8a10dbf30ebf03310da76ce12cd0f647b5efb8b4765eaf",
    "candidates.parquet": "74c96f8f0c68afa9caecb9906ddd3b504ebad8c31448fb845fcd25660787f755",
    "holdout1900_manifest.csv": "6ab748e1a5348805bb57b874ead801c56b7d33a607fe2d52add490984dda289b",
    "paper_features_cal100.csv": "841c88c1c6493057cdb4a5d4c0dd0dd5c042ffc43f440b8b617445226a0629e2",
    "paper_metrics.csv": "d06b53ee6299fd382cda4e6576de0069cf141cebd230534c7349e5ae655c8f45",
    "sft_a_rules_config.json": "33ec2eefbdc8c9d2ca9adec4a71a893ae2d5d2746658625bde53faf85e27dc91",
    "summary.json": "6769545e90dccb56d9f722cc88af34493cc750bffbaf4ede8a2a2e596608bd25",
}
MODEL_REQUIRED_FILES = [
    "README.md", "added_tokens.json", "config.json", "model.safetensors",
    "special_tokens_map.json", "spm.model", "tokenizer.json", "tokenizer_config.json",
]
EXPLICIT_DEFINITION_CUES = ("refers to", "is defined as", "denotes", "means")
T3_MIN_HELDOUT_PRECISION = 0.70
T3_MIN_HELDOUT_F05 = 0.70


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for name, expected in EXPECTED_INPUT_HASHES.items():
        actual = sha256_file(CAL / name)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {name}: {actual} != {expected}")
    candidates = pd.read_parquet(CAL / "candidates.parquet")
    features = pd.read_csv(CAL / "paper_features_cal100.csv")
    old_summary = json.loads((CAL / "summary.json").read_text(encoding="utf-8"))
    cal_manifest = pd.read_csv(CAL / "cal100_manifest.csv")
    if len(candidates) != 18420 or len(cal_manifest) != 100 or cal_manifest["paper_id"].nunique() != 100:
        raise RuntimeError("Cal100 candidate schema/count mismatch")
    if old_summary["selection"]["holdout_processed_count"] != 0:
        raise RuntimeError("Holdout1900 is no longer unprocessed")
    return candidates, features, old_summary


class NLIVerifier:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, local_files_only=True, torch_dtype=dtype
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.device = next(self.model.parameters()).device
        labels = {int(k): str(v).casefold() for k, v in self.model.config.id2label.items()}
        self.indices = {label: index for index, label in labels.items()}
        if set(self.indices) != {"entailment", "neutral", "contradiction"}:
            raise RuntimeError(f"Unexpected NLI label mapping: {labels}")

    def predict(self, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        output: list[dict[str, float]] = []
        for start in range(0, len(premises), batch_size):
            encoded = self.tokenizer(
                premises[start:start + batch_size], hypotheses[start:start + batch_size],
                padding=True, truncation="longest_first", max_length=512, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                scores = torch.softmax(self.model(**encoded).logits.float(), dim=-1).cpu().numpy()
            for row in scores:
                output.append({label: float(row[index]) for label, index in self.indices.items()})
        return output


def qasper_numeric_metrics() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with tarfile.open(QASPER, "r:*") as archive:
        member = next(name for name in archive.getnames() if name.endswith("qasper-test-v0.3.json"))
        data = json.load(archive.extractfile(member))
    rows: list[dict[str, Any]] = []
    for paper_id, paper in data.items():
        for qa in paper["qas"]:
            for annotation in qa["answers"]:
                answer = annotation["answer"]
                evidence = [normalize_text(x) for x in answer.get("evidence") or []]
                for span_index, span in enumerate(answer.get("extractive_spans") or []):
                    span = normalize_text(span)
                    if not is_strict_numeric_answer(span) or not evidence:
                        continue
                    matches = [exact_or_normalized_in(span, item) for item in evidence]
                    if not any(normalized for _, normalized in matches):
                        continue
                    signature = numeric_signature(span)
                    rows.append({
                        "instance_id": stable_id([paper_id, qa["question_id"], annotation["annotation_id"], span_index], "qnum"),
                        "paper_id": paper_id,
                        "question_id": qa["question_id"],
                        "annotation_id": annotation["annotation_id"],
                        "answer": span,
                        "answer_signature": signature,
                        "gold_numeric_answer_recovered": signature is not None,
                        "gold_answer_span_recovered": any(exact for exact, _ in matches),
                        "unit_preserved": bool(signature is not None and any(normalized for _, normalized in matches)),
                        "gold_evidence_supported": any(normalized for _, normalized in matches),
                    })
    metrics = {
        "status": "OK",
        "dataset": "QASPER",
        "version": "v0.3",
        "split": "test",
        "role": "external_verifier_calibration",
        "usage_note": "reused_public_gold_calibration; never a blind SFT evaluation",
        "subset": "QASPER_NUMERIC_EXTRACTIVE",
        "selection_rule": "human extractive answer; strict numeric/range/percent/number+unit/statistical schema; normalized match in gold evidence",
        "eligible_gold_count": len(rows),
        "subset_sha256": sha256_json(rows),
        "gold_numeric_answer_recovered": float(np.mean([r["gold_numeric_answer_recovered"] for r in rows])) if rows else None,
        "gold_answer_span_recovered": float(np.mean([r["gold_answer_span_recovered"] for r in rows])) if rows else None,
        "unit_preserved": float(np.mean([r["unit_preserved"] for r in rows])) if rows else None,
        "gold_evidence_supported": float(np.mean([r["gold_evidence_supported"] for r in rows])) if rows else None,
        "precision_not_reported_reason": "QASPER gold does not exhaustively label every valid numeric fact in each paragraph",
    }
    return metrics, rows


def flatten_scitat_source(row: dict[str, Any]) -> str:
    parts: list[str] = []
    paragraph = row.get("paragraph")
    if isinstance(paragraph, dict):
        parts.extend([str(paragraph.get("text", ""))])
    elif paragraph:
        parts.append(str(paragraph))
    for table in row.get("tables") or []:
        parts.append(str(table.get("caption", "")))
        for table_row in table.get("table") or []:
            parts.append(" | ".join(str(cell) for cell in table_row))
    return "\n".join(parts)


def scitat_metrics() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(SCITAT.read_text(encoding="utf-8"))
    allowed = {"Table Look up", "Span Look up", "Paragraph Look up"}
    rows: list[dict[str, Any]] = []
    for item in data:
        if item.get("question_type") not in allowed:
            continue
        source = flatten_scitat_source(item)
        answers = item.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        numeric_answers = [normalize_text(answer) for answer in answers if is_strict_numeric_answer(answer)]
        if len(numeric_answers) != 1:
            continue
        answer = numeric_answers[0]
        exact, normalized = exact_or_normalized_in(answer, source)
        if not normalized:
            continue
        signature = numeric_signature(answer)
        rows.append({
            "instance_id": str(item["id"]), "question_type": item["question_type"], "answer_type": item.get("answer_type"), "answer": answer,
            "answer_recovered": normalized, "exact_span_recovered": exact,
            "value_unit_normalized": signature is not None,
        })
    metrics = {
        "status": "OK", "dataset": "SciTaT", "version": "official repository snapshot",
        "split": "test", "role": "diagnostic_only", "usage_note": "secondary_external_diagnostic",
        "selection_rule": "official test; lookup-only question type (therefore no calculation/multi-step task); exactly one strict numeric final answer; normalized answer present in supplied table/text",
        "eligible_gold_count": len(rows), "subset_sha256": sha256_json(rows),
        "answer_recoverability": float(np.mean([r["answer_recovered"] for r in rows])) if rows else None,
        "exact_span_recoverability": float(np.mean([r["exact_span_recovered"] for r in rows])) if rows else None,
        "unit_value_normalization_accuracy": float(np.mean([r["value_unit_normalized"] for r in rows])) if rows else None,
        "precision_not_reported_reason": "SciTaT is a secondary diagnostic and is not task-equivalent exhaustive T2 gold",
    }
    return metrics, rows


def load_scifact() -> tuple[dict[int, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    with tarfile.open(SCIFACT, "r:*") as archive:
        names = archive.getnames()
        def lines(suffix: str) -> list[dict[str, Any]]:
            name = next(name for name in names if name.endswith(suffix))
            return [json.loads(line) for line in archive.extractfile(name).read().decode("utf-8").splitlines()]
        corpus_rows = lines("corpus.jsonl")
        claims = {split: lines(f"claims_{split}.jsonl") for split in ("train", "dev", "test")}
        logical_hashes = {}
        for split, values in claims.items():
            logical_hashes[split] = sha256_json(values)
        logical_hashes["corpus"] = sha256_json(corpus_rows)
    corpus = {int(row["doc_id"]): row for row in corpus_rows}
    return corpus, claims, logical_hashes


def scifact_instances(corpus: dict[int, dict[str, Any]], rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for row in rows:
        for doc_id, evidence_sets in (row.get("evidence") or {}).items():
            doc = corpus[int(doc_id)]
            for rationale_index, evidence in enumerate(evidence_sets):
                sentence_ids = evidence["sentences"]
                premise = " ".join(doc["abstract"][int(index)] for index in sentence_ids)
                instances.append({
                    "instance_id": stable_id([split, row["id"], doc_id, rationale_index, sentence_ids], "sf"),
                    "claim_id": row["id"], "doc_id": int(doc_id), "sentence_ids": sentence_ids,
                    "premise": premise, "hypothesis": row["claim"], "label": evidence["label"],
                })
    return instances


def evaluate_scifact(nli: NLIVerifier) -> tuple[dict[str, Any], float, dict[str, str]]:
    corpus, claims, logical_hashes = load_scifact()
    train = scifact_instances(corpus, claims["train"], "train")
    dev = scifact_instances(corpus, claims["dev"], "dev")
    train_scores = nli.predict([x["premise"] for x in train], [x["hypothesis"] for x in train])
    train_y = [int(x["label"] == "SUPPORT") for x in train]
    threshold, train_metrics = select_f05_threshold(train_y, [x["entailment"] for x in train_scores])
    dev_scores = nli.predict([x["premise"] for x in dev], [x["hypothesis"] for x in dev])
    dev_y = [int(x["label"] == "SUPPORT") for x in dev]
    dev_pred = [int(x["entailment"] >= threshold) for x in dev_scores]
    dev_metrics = binary_metrics(dev_y, dev_pred, beta=0.5)
    argmax_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for gold, score in zip((x["label"] for x in dev), dev_scores):
        argmax_confusion[gold][max(score, key=score.get)] += 1
    result = {
        "status": "OK",
        "dataset": "SciFact", "version": "official latest release archive downloaded 2026-08-11",
        "release_structure": {"train_claims": len(claims["train"]), "dev_claims": len(claims["dev"]), "test_claims": len(claims["test"]), "test_labels_available": False},
        "calibration_split": {"split": "train", "role": "external_verifier_calibration", "n_gold_rationales": len(train), "metrics": train_metrics},
        "heldout_reporting_split": {"split": "dev", "role": "external_verifier_validation", "n_gold_rationales": len(dev), "metrics": dev_metrics},
        "official_test_not_used_reason": "official release test claims contain no evidence labels",
        "selected_entailment_threshold": threshold,
        "threshold_selection": "maximize SUPPORT-vs-CONTRADICT F0.5 on train; ties: higher precision, then higher threshold",
        "argmax_heldout_confusion": {gold: dict(counts) for gold, counts in argmax_confusion.items()},
        "logical_split_sha256": logical_hashes,
        "heldout_gate": {
            "minimum_precision": T3_MIN_HELDOUT_PRECISION, "minimum_f0.5": T3_MIN_HELDOUT_F05,
            "passed": dev_metrics["precision"] >= T3_MIN_HELDOUT_PRECISION and dev_metrics["f0.5"] >= T3_MIN_HELDOUT_F05,
        },
    }
    return result, threshold, logical_hashes


def valid_candidate_span(row: pd.Series) -> bool:
    text = str(row["evidence_clean"])
    try:
        start, end = int(row["answer_start"]), int(row["answer_end"])
    except (TypeError, ValueError):
        return False
    return 0 <= start < end <= len(text) and text[start:end] == row["answer_span"]


def t1_counterfactuals(hard: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    acronym = hard[hard["task_subtype"] != "explicit_definition"].sort_values("candidate_id")
    definitions = acronym[["candidate_id", "definition"]].to_dict("records")
    by_id: dict[str, dict[str, Any]] = {}
    rejected = generated = 0
    for index, (_, row) in enumerate(acronym.iterrows()):
        variant = None
        for offset in range(1, len(definitions) + 1):
            other = definitions[(index + offset) % len(definitions)]
            if normalize_match(other["definition"]) != normalize_match(row["definition"]):
                variant = other["definition"]
                break
        if variant is None:
            by_id[row["candidate_id"]] = {"generated": False, "pass": None, "variants": []}
            continue
        accept = independent_acronym_alignment(row["term"], variant)
        generated += 1
        rejected += int(not accept)
        by_id[row["candidate_id"]] = {
            "generated": True, "pass": not accept,
            "variants": [{"mutation_type": "term_A_definition_B", "mutated_definition": variant, "accepted": accept}],
        }
    positives = [independent_acronym_alignment(row.term, row.definition) for row in acronym.itertuples()]
    metrics = {
        "test_type": "synthetic consistency test", "n_positive": len(positives),
        "positive_accept_rate": float(np.mean(positives)) if positives else None,
        "n_counterfactual": generated, "counterfactual_reject_rate": rejected / generated if generated else None,
    }
    return by_id, metrics


def numeric_same_unit_ambiguity(row: pd.Series) -> bool:
    anchor = str(row["anchor_text"])
    target = numeric_signature(str(row["answer_span"]))
    if target is None:
        return True
    signatures = []
    pattern = re.compile(r"(?:[pPrRtTFz]\s*(?:=|<|>|≤|≥)\s*)?" + NUMBER_RE.pattern + r"\s*(?:%|percent(?:age)?|[A-Za-zμ°][A-Za-z0-9μ°/%²^.-]*)?", re.I)
    for match in pattern.finditer(anchor):
        sig = numeric_signature(match.group(0))
        if sig is not None and sig["unit"] == target["unit"]:
            signatures.append(sig["normalized"])
    return len(set(signatures)) > 1


def t2_counterfactual(answer: str) -> tuple[str, str] | None:
    text = str(answer)
    if re.search(r"[pP]\s*<", text):
        return "operator_flip", re.sub(r"([pP]\s*)<", r"\1>", text, count=1)
    if re.search(r"[pP]\s*>", text):
        return "operator_flip", re.sub(r"([pP]\s*)>", r"\1<", text, count=1)
    range_match = re.search(r"(" + NUMBER_RE.pattern + r")\s*(–|—|-|to)\s*(" + NUMBER_RE.pattern + r")", text, re.I)
    if range_match:
        return "range_endpoint_swap", text[:range_match.start()] + range_match.group(3) + range_match.group(2) + range_match.group(1) + text[range_match.end():]
    number = NUMBER_RE.search(text)
    if number:
        raw = number.group(0)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            return None
        delta = 10.0 if "%" in text or "percent" in text.casefold() else (1.0 if abs(value) < 10 else 10.0)
        mutated = f"{value + delta:g}"
        return "number_perturbation", text[:number.start()] + mutated + text[number.end():]
    return None


def t3_counterfactual(row: pd.Series) -> tuple[str, str] | None:
    claim = str(row["answer_span"])
    replacements = [(r"\bincreased\b", "decreased"), (r"\bdecreased\b", "increased"), (r"\bhigher\b", "lower"), (r"\blower\b", "higher")]
    for pattern, replacement in replacements:
        if re.search(pattern, claim, re.I):
            return "direction_flip", re.sub(pattern, replacement, claim, count=1, flags=re.I)
    uncertain = UNCERTAINTY_RE.search(claim)
    if uncertain:
        return "uncertainty_removal", normalize_text(claim[:uncertain.start()] + claim[uncertain.end():])
    negation = NEGATION_RE.search(claim)
    if negation:
        return "negation_removal", normalize_text(claim[:negation.start()] + claim[negation.end():])
    relation = str(row.get("relation") or "")
    if relation and re.search(rf"\b{re.escape(relation)}\b", claim, re.I):
        return "negation_insertion", re.sub(rf"\b{re.escape(relation)}\b", "not " + relation, claim, count=1, flags=re.I)
    return None


def validate_t3_deterministic(row: pd.Series) -> tuple[bool, str | None, dict[str, Any]]:
    claim = str(row["answer_span"])
    conditions = json.loads(row["condition_spans"] or "{}")
    negation_present = bool(NEGATION_RE.search(claim))
    uncertainty_present = bool(UNCERTAINTY_RE.search(claim))
    if negation_present and not row["negation_flag"]:
        return False, "negation_flag_not_preserved", {"negation_present": True}
    if negation_present and not conditions.get("negation"):
        return False, "negation_condition_not_preserved", {"negation_present": True}
    if uncertainty_present and not row["uncertainty_flag"]:
        return False, "uncertainty_flag_not_preserved", {"uncertainty_present": True}
    if uncertainty_present and not conditions.get("uncertainty"):
        return False, "uncertainty_condition_not_preserved", {"uncertainty_present": True}
    required = ["quantity_or_threshold", "spatial", "temporal", "treatment", "comparison"]
    missing = []
    invalid_offsets = []
    evidence = str(row["evidence_clean"])
    for key in required:
        span = conditions.get(key)
        if not span:
            continue
        text = str(span.get("text", ""))
        if text not in claim:
            missing.append(key)
        start, end = span.get("start"), span.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or evidence[start:end] != text:
            invalid_offsets.append(key)
    if invalid_offsets:
        return False, "condition_span_offset_invalid", {"invalid": invalid_offsets}
    if missing:
        return False, "condition_not_preserved", {"missing": missing}
    return True, None, {"negation_present": negation_present, "uncertainty_present": uncertainty_present, "condition_keys_checked": required}


def build_manifest(scifact_hashes: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "MACRONYM", "version": None, "split": "official scientific English subset/splits",
            "source": "author/official release not available locally or from an identifiable author release",
            "role": "external_verifier_calibration", "sha256": None, "status": "DATASET_NOT_AVAILABLE",
        },
        {
            "dataset": "QASPER", "version": "v0.3", "split": "test", "source": str(QASPER.relative_to(ROOT)),
            "role": "external_verifier_calibration", "usage_note": "reused_public_gold_calibration",
            "sha256": sha256_file(QASPER), "status": "OK",
        },
        {
            "dataset": "SciTaT", "version": "official repository snapshot", "split": "test", "source": str(SCITAT.relative_to(ROOT)),
            "role": "diagnostic_only", "usage_note": "secondary_external_diagnostic",
            "sha256": sha256_file(SCITAT), "status": "OK",
        },
        {
            "dataset": "SciFact", "version": "official latest release archive downloaded 2026-08-11", "split": "train",
            "source": str(SCIFACT.relative_to(ROOT)), "role": "external_verifier_calibration",
            "sha256": scifact_hashes["train"], "archive_sha256": sha256_file(SCIFACT), "status": "OK",
        },
        {
            "dataset": "SciFact", "version": "official latest release archive downloaded 2026-08-11", "split": "dev",
            "source": str(SCIFACT.relative_to(ROOT)), "role": "external_verifier_validation",
            "sha256": scifact_hashes["dev"], "archive_sha256": sha256_file(SCIFACT), "status": "OK",
        },
        {
            "dataset": "SciFact", "version": "official latest release archive downloaded 2026-08-11", "split": "test",
            "source": str(SCIFACT.relative_to(ROOT)), "role": "diagnostic_only",
            "sha256": scifact_hashes["test"], "archive_sha256": sha256_file(SCIFACT), "status": "LABELS_NOT_AVAILABLE_NOT_USED",
        },
    ]


def model_manifest() -> dict[str, Any]:
    files = {name: {"sha256": sha256_file(MODEL / name), "bytes": (MODEL / name).stat().st_size} for name in MODEL_REQUIRED_FILES}
    aggregate = sha256_json({name: value["sha256"] for name, value in files.items()})
    return {
        "repo_id": MODEL_REPO, "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
        "local_path": str(MODEL.relative_to(ROOT)), "files": files, "aggregate_sha256": aggregate,
    }


def public_metrics_rows(t1: dict[str, Any], qasper: dict[str, Any], scitat: dict[str, Any], scifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(dataset: str, split: str, role: str, n: Any, metric: str, value: Any, status: str) -> None:
        rows.append({"dataset": dataset, "split": split, "role": role, "n": n, "metric": metric, "value": value, "status": status})
    for metric in ("short_form_precision", "short_form_recall", "short_form_f1", "long_form_precision", "long_form_recall", "long_form_f1", "pair_precision", "pair_recall", "pair_f1"):
        add("MACRONYM", "official scientific English subset", "external_verifier_calibration", None, metric, None, t1["status"])
    for metric in ("gold_numeric_answer_recovered", "gold_answer_span_recovered", "unit_preserved", "gold_evidence_supported"):
        add("QASPER", "test", "external_verifier_calibration", qasper["eligible_gold_count"], metric, qasper[metric], qasper["status"])
    for metric in ("answer_recoverability", "exact_span_recoverability", "unit_value_normalization_accuracy"):
        add("SciTaT", "test", "diagnostic_only", scitat["eligible_gold_count"], metric, scitat[metric], scitat["status"])
    held = scifact["heldout_reporting_split"]
    for metric in ("precision", "recall", "f0.5", "f1"):
        add("SciFact", "dev", "external_verifier_validation", held["n_gold_rationales"], metric, held["metrics"][metric], scifact["status"])
    add("SciFact", "train", "external_verifier_calibration", scifact["calibration_split"]["n_gold_rationales"], "selected_entailment_threshold", scifact["selected_entailment_threshold"], scifact["status"])
    return rows


def verify_cal100(candidates: pd.DataFrame, features: pd.DataFrame, nli: NLIVerifier, threshold: float) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output = candidates.copy()
    appended = {
        "verification_level": [], "machine_verified": [], "machine_reject_reason": [],
        "verifier_scores": [], "counterfactual_pass": [], "verification_scope": [],
    }
    t1_hard = candidates[(candidates.task_type == "T1_TERM") & candidates.hard_gate_pass]
    t1_cf_by_id, t1_cf_metrics = t1_counterfactuals(t1_hard)

    t3_indices = candidates.index[(candidates.task_type == "T3_CLAIM") & candidates.hard_gate_pass].tolist()
    t3_premises = []
    t3_hypotheses = []
    for index in t3_indices:
        row = candidates.loc[index]
        supports = json.loads(row["support_evidence_clean"] or "[]")
        t3_premises.append(" ".join([str(row["evidence_clean"]), *map(str, supports)]))
        t3_hypotheses.append(str(row["answer_span"]))
    t3_scores = nli.predict(t3_premises, t3_hypotheses)
    score_by_index = dict(zip(t3_indices, t3_scores))
    t3_cf_rows: list[tuple[int, str, str, str]] = []
    for index in t3_indices:
        mutation = t3_counterfactual(candidates.loc[index])
        if mutation:
            mtype, claim = mutation
            supports = json.loads(candidates.loc[index, "support_evidence_clean"] or "[]")
            premise = " ".join([str(candidates.loc[index, "evidence_clean"]), *map(str, supports)])
            t3_cf_rows.append((index, mtype, premise, claim))
    cf_scores = nli.predict([x[2] for x in t3_cf_rows], [x[3] for x in t3_cf_rows]) if t3_cf_rows else []
    t3_cf_by_index: dict[int, dict[str, Any]] = {}
    for (index, mtype, _premise, claim), score in zip(t3_cf_rows, cf_scores):
        rejected = score["entailment"] < threshold
        t3_cf_by_index[index] = {"generated": True, "pass": rejected, "variants": [{"mutation_type": mtype, "original_claim": str(candidates.loc[index, "answer_span"]), "mutated_claim": claim, "scores": score, "rejected": rejected}]}

    for index, row in candidates.iterrows():
        task = row["task_type"]
        verified = False
        reason: str | None = None
        level = "HARD_GATE_ONLY"
        scope = "not_evaluated_failed_hard_gate"
        scores: dict[str, Any] = {}
        cf_pass: bool | None = None
        if not row["hard_gate_pass"]:
            reason = "original_hard_gate_failed"
            level = "REJECTED_MACHINE_VERIFICATION"
        elif task == "T1_TERM":
            exact_term = str(row["term"]) in str(row["evidence_clean"])
            exact_definition = valid_candidate_span(row) and str(row["definition"]) == str(row["answer_span"])
            if row["task_subtype"] == "explicit_definition":
                cue = next((cue for cue in EXPLICIT_DEFINITION_CUES if cue in str(row["anchor_text"]).casefold()), None)
                verified = exact_term and exact_definition and cue is not None
                reason = None if verified else "explicit_definition_relation_invalid"
                scope = "rule_verified_definition"
                level = "RULE_VERIFIED" if verified else "REJECTED_MACHINE_VERIFICATION"
                scores = {"exact_term": exact_term, "exact_definition": exact_definition, "explicit_cue": cue}
            else:
                aligned = independent_acronym_alignment(str(row["term"]), str(row["definition"]))
                cf = t1_cf_by_id.get(row["candidate_id"], {"generated": False, "pass": None, "variants": []})
                cf_pass = cf["pass"]
                verified = exact_term and exact_definition and aligned and cf_pass is not False
                reason = None if verified else ("acronym_counterfactual_not_rejected" if cf_pass is False else "independent_acronym_alignment_failed")
                scope = "rule_verified_acronym_no_external_gold"
                level = "RULE_VERIFIED" if verified else "REJECTED_MACHINE_VERIFICATION"
                scores = {"exact_term": exact_term, "exact_definition": exact_definition, "independent_alignment": aligned, "counterfactual": cf}
        elif task == "T2_NUMERIC":
            exact_span = valid_candidate_span(row)
            signature = numeric_signature(str(row["answer_span"]))
            unit_ok = signature is not None and (not row["unit"] or normalize_match(row["unit"]) in normalize_match(row["answer_span"]))
            raw = str(row["evidence_raw"])
            raw_occurrences = raw.count(str(row["answer_span"]))
            roundtrip_exact = raw_occurrences == 1
            roundtrip_normalized = normalize_match(str(row["answer_span"])) and normalize_match(raw).count(normalize_match(str(row["answer_span"]))) == 1
            context_ok = len(re.findall(r"[A-Za-z][A-Za-z'-]*", str(row["anchor_text"]))) >= 4
            ambiguity = numeric_same_unit_ambiguity(row)
            mutation = t2_counterfactual(str(row["answer_span"]))
            cf_rejected = None
            cf_detail = None
            if mutation:
                mtype, mutated = mutation
                _exact, norm = exact_or_normalized_in(mutated, raw)
                cf_rejected = not norm
                cf_detail = {"mutation_type": mtype, "mutated_answer": mutated, "rejected": cf_rejected}
            cf_pass = cf_rejected
            gates = {
                "exact_span": exact_span, "normalization": signature is not None, "unit_consistent": unit_ok,
                "semantic_context": context_ok, "roundtrip_exact": roundtrip_exact,
                "roundtrip_normalized": bool(roundtrip_normalized), "answer_uniqueness_check": not ambiguity,
            }
            failed = next((name for name, passed in gates.items() if not passed), None)
            if ambiguity:
                reason = "ambiguous_numeric_binding"
            elif failed:
                reason = f"t2_{failed}_failed"
            else:
                verified = True
            level = "EXTERNAL_GOLD_CALIBRATED_RULE" if verified else "REJECTED_MACHINE_VERIFICATION"
            scope = "qasper_numeric_calibrated_rule_with_scitat_diagnostic"
            scores = {**gates, "signature": signature, "raw_exact_occurrences": raw_occurrences, "counterfactual": cf_detail}
        elif task == "T3_CLAIM":
            score = score_by_index[index]
            deterministic_ok, deterministic_reason, deterministic = validate_t3_deterministic(row)
            mutation = t3_cf_by_index.get(index, {"generated": False, "pass": None, "variants": []})
            cf_pass = mutation["pass"]
            if score["entailment"] < threshold:
                reason = "nli_entailment_below_frozen_threshold"
            elif not deterministic_ok:
                reason = deterministic_reason
            else:
                verified = True
            level = "MACHINE_VERIFIED" if verified else "REJECTED_MACHINE_VERIFICATION"
            scope = "scifact_calibrated_nli_plus_deterministic_conditions"
            scores = {"nli": score, "threshold": threshold, "deterministic": deterministic, "counterfactual": mutation}
        appended["verification_level"].append(level)
        appended["machine_verified"].append(bool(verified))
        appended["machine_reject_reason"].append(reason)
        appended["verifier_scores"].append(json_compact(scores))
        appended["counterfactual_pass"].append(cf_pass)
        appended["verification_scope"].append(scope)
    for column, values in appended.items():
        output[column] = values

    metrics_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    feature_tokens = dict(zip(features.paper_id, features.paper_clean_token_count))
    for task in ["T1_TERM", "T2_NUMERIC", "T3_CLAIM"]:
        part = output[output.task_type == task]
        paper_counts = part[part.machine_verified].groupby("paper_id").size().reindex(features.paper_id, fill_value=0)
        per10k = [float(count) * 10000 / float(feature_tokens[paper]) for paper, count in paper_counts.items()]
        task_summary = {
            "raw": int(len(part)), "hard_pass": int(part.hard_gate_pass.sum()),
            "materialization_ready": int(part.materialization_ready.sum()), "machine_verified": int(part.machine_verified.sum()),
            "papers_with_at_least_one_machine_verified": int((paper_counts > 0).sum()),
            "machine_verified_per_10k_source_tokens": float(part.machine_verified.sum()) * 10000 / float(features.paper_clean_token_count.sum()),
            "yield_count_per_paper": {**distribution(paper_counts.astype(float).tolist()), "zero_yield_papers": int((paper_counts == 0).sum())},
            "yield_per_10k_tokens_per_paper": {**distribution(per10k), "zero_yield_papers": int(sum(value == 0 for value in per10k))},
        }
        summary[task] = task_summary
        for stage in ("raw", "hard_pass", "materialization_ready", "machine_verified", "papers_with_at_least_one_machine_verified", "machine_verified_per_10k_source_tokens"):
            metrics_rows.append({"task": task, "metric": stage, "value": task_summary[stage]})
        for distribution_name in ("yield_count_per_paper", "yield_per_10k_tokens_per_paper"):
            for metric, value in task_summary[distribution_name].items():
                metrics_rows.append({"task": task, "metric": f"{distribution_name}.{metric}", "value": value})

    t2_hard = output[(output.task_type == "T2_NUMERIC") & output.hard_gate_pass]
    t2_cf = t2_hard.counterfactual_pass.dropna().astype(bool)
    t3_hard = output[(output.task_type == "T3_CLAIM") & output.hard_gate_pass]
    t3_cf = t3_hard.counterfactual_pass.dropna().astype(bool)
    counterfactual_rows = [
        {"task": "T1_TERM", "test_type": "synthetic consistency test", "n": t1_cf_metrics["n_counterfactual"], "metric": "counterfactual_reject_rate", "value": t1_cf_metrics["counterfactual_reject_rate"]},
        {"task": "T1_TERM", "test_type": "positive consistency check", "n": t1_cf_metrics["n_positive"], "metric": "positive_accept_rate", "value": t1_cf_metrics["positive_accept_rate"]},
        {"task": "T2_NUMERIC", "test_type": "synthetic consistency test", "n": int(len(t2_cf)), "metric": "counterfactual_reject_rate", "value": float(t2_cf.mean()) if len(t2_cf) else None},
        {"task": "T3_CLAIM", "test_type": "synthetic stress test", "n": int(len(t3_cf)), "metric": "counterfactual_sensitivity", "value": float(t3_cf.mean()) if len(t3_cf) else None},
    ]
    return output, metrics_rows, counterfactual_rows, summary


def top_rejections(output: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for task in ["T1_TERM", "T2_NUMERIC", "T3_CLAIM"]:
        counts = output[(output.task_type == task) & output.hard_gate_pass & ~output.machine_verified].machine_reject_reason.value_counts().head(5)
        result[task] = [{"reason": str(reason), "count": int(count)} for reason, count in counts.items()]
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {out}")
    candidates, features, old_summary = validate_frozen_inputs()
    required = [QASPER, SCITAT, SCIFACT, *(MODEL / name for name in MODEL_REQUIRED_FILES)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Required frozen assets missing: {missing}")

    qasper_metrics, _qasper_rows = qasper_numeric_metrics()
    scitat_result, _scitat_rows = scitat_metrics()
    nli = NLIVerifier()
    scifact_result, threshold, scifact_hashes = evaluate_scifact(nli)
    manifest = build_manifest(scifact_hashes)
    model_info = model_manifest()
    t1_metrics = {
        "status": "DATASET_NOT_AVAILABLE", "dataset": "MACRONYM",
        "scope": "scientific English official subset/splits",
        "short_form": {"precision": None, "recall": None, "f1": None},
        "long_form": {"precision": None, "recall": None, "f1": None},
        "paired_relation": {"precision": None, "recall": None, "f1": None},
        "exact_span_match": None, "normalized_pair_match": None,
        "impact": "T1 external-gold calibration unavailable; acronym verification remains independent deterministic rule verification only",
    }

    code_hashes = {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    verifier_config = {
        "experiment": "corpus_v1_sft_a_machine_validation_v1", "created_date": "2026-08-11",
        "scope": "machine-verifiable materializable candidates only; not final SFT instances",
        "frozen_parent": {"directory": str(CAL.relative_to(ROOT)), "input_sha256": EXPECTED_INPUT_HASHES},
        "external_datasets": manifest,
        "T1": {
            "acronym": ["hard_gate_pass", "exact_short_span", "exact_long_span", "independent_backward_character_alignment", "synthetic_mismatch_consistency"],
            "generic_definition": ["hard_gate_pass", "exact_definition_span", "explicit cue in: refers to/is defined as/denotes/means"],
            "external_gold_status": "DATASET_NOT_AVAILABLE",
        },
        "T2": {
            "requirements": ["hard_gate_pass", "exact_answer_span", "numeric_normalization", "unit_preservation", "semantic_local_context", "unique_raw_roundtrip", "answer_uniqueness_check"],
            "answer_uniqueness_rule": "reject when the anchor contains more than one distinct strict numeric expression with the same normalized unit/operator class",
            "roundtrip": "stored answer must occur exactly once, and normalized answer exactly once, in evidence_raw",
        },
        "T3": {
            "requirements": ["hard_gate_pass", "entailment_at_or_above_frozen_threshold", "negation_consistency", "uncertainty_consistency", "recognized_condition_span_preservation"],
            "selected_entailment_threshold": threshold,
            "selection_split": "SciFact train", "heldout_reporting_split": "SciFact dev",
            "heldout_gate_minimums": {"precision": T3_MIN_HELDOUT_PRECISION, "f0.5": T3_MIN_HELDOUT_F05},
        },
        "normalization": {"unicode": "NFKC", "whitespace": "collapse", "numeric": "comma removal; minus/dash and percent lexical normalization"},
        "counterfactual_rules": {
            "T1": ["term_A_definition_B"], "T2": ["operator_flip", "range_endpoint_swap", "number_perturbation"],
            "T3": ["direction_flip", "uncertainty_removal", "negation_removal", "negation_insertion"],
        },
        "NLI": {
            **model_info, "training": False, "inference_only": True, "dtype": "float16" if torch.cuda.is_available() else "float32",
            "device": "cuda" if torch.cuda.is_available() else "cpu", "max_length": 512, "truncation": "longest_first",
            "premise_hypothesis_order": "tokenizer(premise, hypothesis)", "batch_size": 16,
            "label_mapping": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        },
        "code_sha256": code_hashes,
        "holdout1900_processed": 0, "generative_llm_judge": False, "SFT_training": False,
    }

    # The public-gold-selected threshold and all rules are frozen before Cal100 verification begins.
    out.mkdir(parents=True)
    write_json(out / "public_gold_manifest.json", manifest)
    write_json(out / "t1_macronym_metrics.json", t1_metrics)
    write_json(out / "t2_qasper_numeric_metrics.json", qasper_metrics)
    write_json(out / "t2_scitat_metrics.json", scitat_result)
    write_json(out / "t3_scifact_metrics.json", scifact_result)
    write_csv(out / "public_gold_metrics.csv", public_metrics_rows(t1_metrics, qasper_metrics, scitat_result, scifact_result))
    write_json(out / "machine_verifier_config_v1.json", verifier_config)
    config_sha = sha256_file(out / "machine_verifier_config_v1.json")

    verified, metric_rows, counterfactual_rows, cal_summary = verify_cal100(candidates, features, nli, threshold)
    verified.to_parquet(out / "cal100_machine_verified.parquet", index=False)
    write_csv(out / "cal100_machine_metrics.csv", metric_rows)
    write_csv(out / "counterfactual_metrics.csv", counterfactual_rows)

    provenance_total = int(verified.machine_verified.sum())
    provenance_valid = int(verified[verified.machine_verified].provenance.notna().sum())
    gates = {
        "cal100_100_of_100": len(features) == 100 and features.paper_id.nunique() == 100,
        "holdout_processed_zero": old_summary["selection"]["holdout_processed_count"] == 0,
        "candidate_provenance_100_percent": provenance_total > 0 and provenance_valid == provenance_total,
        "machine_verified_candidate_ids_unique": not verified[verified.machine_verified].candidate_id.duplicated().any(),
        "original_hard_gate_results_unchanged": verified.hard_gate_pass.equals(candidates.hard_gate_pass) and verified.materialization_ready.equals(candidates.materialization_ready),
        "public_gold_split_roles_explicit": all(item["role"] in {"external_verifier_calibration", "external_verifier_validation", "diagnostic_only"} for item in manifest),
        "nli_threshold_selected_only_on_calibration_split": scifact_result["calibration_split"]["split"] == "train",
        "heldout_split_not_used_for_threshold": scifact_result["heldout_reporting_split"]["split"] == "dev",
        "machine_verifier_config_hashed": len(config_sha) == 64,
        "scifact_heldout_gate_passed": scifact_result["heldout_gate"]["passed"],
        "external_gold_all_required_available": t1_metrics["status"] == "OK" and qasper_metrics["status"] == "OK" and scitat_result["status"] == "OK" and scifact_result["status"] == "OK",
    }
    status = "SFT_A_MACHINE_VALIDATION_READY" if all(gates.values()) else "SFT_A_MACHINE_VALIDATION_NOT_READY"
    summary = {
        "experiment": "corpus_v1_sft_a_machine_validation_v1", "status": status,
        "status_reason": "MACRONYM official author release unavailable; T1 public-gold calibration could not run" if t1_metrics["status"] != "OK" else None,
        "scope": "public-gold component calibration plus Cal100 machine verification; no human validation and no final SFT instances",
        "public_gold": {"T1_MACRONYM": t1_metrics, "T2_QASPER": qasper_metrics, "T2_SCITAT": scitat_result, "T3_SCIFACT": scifact_result},
        "cal100": cal_summary, "top_machine_rejection_reasons": top_rejections(verified),
        "counterfactual": {row["task"] + ":" + row["metric"]: row for row in counterfactual_rows},
        "ready_gates": gates, "machine_verifier_config_sha256": config_sha,
        "frozen_parent_sha256_reverified": EXPECTED_INPUT_HASHES,
        "holdout1900_processed_count": 0, "accepted_sft_instances": 0, "training_runs": 0,
        "limitations": [
            "MACRONYM official release was unavailable, so T1 acronym external precision/recall/F1 are unknown.",
            "MACRONYM would not validate generic-definition patterns even if available; those remain rule_verified_definition.",
            "QASPER and SciTaT measure eligible gold recoverability, not Cal100 T2 precision.",
            "SciFact validates the NLI component on biomedical claims, not every geoscience Cal100 claim.",
            "Synthetic counterfactual tests are consistency/stress tests, not human accuracy estimates.",
        ],
    }
    write_json(out / "summary.json", summary)
    names = sorted(path.name for path in out.iterdir() if path.is_file() and path.name != "checksums.sha256")
    checksum_lines = [f"{sha256_file(out / name)}  {name}" for name in names]
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(out), "config_sha256": config_sha, "summary_sha256": sha256_file(out / "summary.json"), "cal100": cal_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
