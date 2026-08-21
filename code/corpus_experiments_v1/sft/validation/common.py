from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)(?:\s*[×x]\s*10\s*[−-]?\s*\d+)?")
STAT_RE = re.compile(r"^\s*(?:[pPrRtTFz]|χ2|χ²|R\s*\^?2)\s*(?:=|<|>|≤|≥)\s*" + NUMBER_RE.pattern + r"\s*$")
RANGE_RE = re.compile(r"^\s*" + NUMBER_RE.pattern + r"\s*(?:–|—|-|to)\s*" + NUMBER_RE.pattern + r"(?:\s*[^\d\W][\w/%°²μ.-]{0,20})?\s*$", re.I)
PERCENT_RE = re.compile(r"^\s*(?:about|approximately|roughly|nearly|up to|at least|less than|more than)?\s*" + NUMBER_RE.pattern + r"\s*(?:%|percent(?:age)?)\s*$", re.I)
NUMERIC_WITH_UNIT_RE = re.compile(
    r"^\s*(?:about|approximately|roughly|nearly|up to|at least|less than|more than)?\s*"
    + NUMBER_RE.pattern
    + r"\s*(?:%|percent(?:age)?|[A-Za-zμ°][A-Za-z0-9μ°/%²^.-]*(?:\s+[A-Za-z][A-Za-z0-9/%²^.-]*){0,3})?\s*$",
    re.I,
)
UNCERTAINTY_RE = re.compile(r"\b(?:may|might|could|suggest\w*|likely|possibly|approximately)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:not|no|without)\b", re.I)


def normalize_text(value: Any) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", "" if value is None else str(value))).strip()


def normalize_match(value: Any) -> str:
    text = normalize_text(value).casefold().replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace(",", "").replace("\\", "").replace("percent", "%").replace("percentage", "%")
    return re.sub(r"\s+", "", text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_id(parts: Iterable[Any], prefix: str) -> str:
    payload = "|".join("" if value is None else str(value) for value in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def exact_or_normalized_in(needle: str, haystack: str) -> tuple[bool, bool]:
    exact = needle in haystack
    normalized = bool(normalize_match(needle)) and normalize_match(needle) in normalize_match(haystack)
    return exact, normalized


def is_strict_numeric_answer(answer: str) -> bool:
    text = normalize_text(answer).strip(".;,()[]")
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return False
    if STAT_RE.fullmatch(text) or RANGE_RE.fullmatch(text) or PERCENT_RE.fullmatch(text):
        return True
    return len(numbers) == 1 and bool(NUMERIC_WITH_UNIT_RE.fullmatch(text))


def numeric_signature(answer: str) -> dict[str, Any] | None:
    text = normalize_text(answer).strip(".;,()[]")
    if not is_strict_numeric_answer(text):
        return None
    values = [normalize_match(x) for x in NUMBER_RE.findall(text)]
    operator = None
    match = re.search(r"(?:=|<|>|≤|≥|±|\+/-|\bto\b|–|—|-)", text, re.I)
    if match:
        operator = match.group(0).casefold().replace("–", "-").replace("—", "-")
    remainder = NUMBER_RE.sub(" ", text)
    remainder = re.sub(r"\b(?:about|approximately|roughly|nearly|up|to|at|least|less|than|more|p|r|t|f|z|ci|confidence|interval)\b", " ", remainder, flags=re.I)
    remainder = re.sub(r"[=<>≤≥±+\-/()[\],.;:]", " ", remainder)
    unit = normalize_match(remainder) or ("%" if "%" in normalize_match(text) else None)
    if "percent" in text.casefold() or "percentage" in text.casefold():
        unit = "%"
    return {"values": values, "unit": unit, "operator": operator, "normalized": normalize_match(text)}


def independent_acronym_alignment(short_form: str, long_form: str) -> bool:
    """Independent backward character alignment; does not import the detector implementation."""
    short = re.sub(r"[^A-Za-z0-9]", "", normalize_text(short_form)).upper()
    long = normalize_text(long_form)
    if not (2 <= len(short) <= 10) or not long:
        return False
    cursor = len(long) - 1
    for index in range(len(short) - 1, -1, -1):
        target = short[index].casefold()
        found = -1
        while cursor >= 0:
            if long[cursor].casefold() == target:
                if index != 0 or cursor == 0 or not long[cursor - 1].isalnum():
                    found = cursor
                    break
            cursor -= 1
        if found < 0:
            return False
        cursor = found - 1
    return True


def binary_metrics(y_true: list[int], y_pred: list[int], beta: float = 0.5) -> dict[str, Any]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        f"f{beta:g}": fbeta,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def select_f05_threshold(y_true: list[int], scores: list[float]) -> tuple[float, dict[str, Any]]:
    candidates = sorted({0.0, 1.0, *[float(x) for x in scores]})
    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in candidates:
        pred = [int(score >= threshold) for score in scores]
        metrics = binary_metrics(y_true, pred, beta=0.5)
        ranked.append((metrics["f0.5"], metrics["precision"], threshold, metrics))
    best = max(ranked, key=lambda item: (item[0], item[1], item[2]))
    return best[2], best[3]


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "q1": None, "q3": None, "iqr": None, "p90": None}
    array = np.asarray(values, dtype=float)
    q1, median, q3, p90 = np.percentile(array, [25, 50, 75, 90])
    return {
        "mean": float(array.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p90": float(p90),
    }


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
