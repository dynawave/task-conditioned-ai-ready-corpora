from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable

from rules import NEGATION_CUES, UNCERTAINTY_CUES, classify_section


SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
URL_DOI_RE = re.compile(r"(?:https?://|www\.|doi\s*:|10\.\d{4,9}/)", re.I)
CITATION_ONLY_RE = re.compile(r"^\s*(?:\[[\d,; –—-]+\]|\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?\))\s*[.]?\s*$")
REFERENCE_LIKE_RE = re.compile(r"(?:\bdoi\b|https?://|\bet\s+al\.?,?\s*\(?\d{4}\)?|^\s*\[\d+\])", re.I)


def normalize_text(text: str | None) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def normalize_key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold()).strip()


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text):
        prefix = text[max(start, match.start() - 16):match.start()].casefold()
        if any(prefix.endswith(token) for token in ("fig.", "figs.", "eq.", "eqs.", "ref.", "refs.", "cf.", "al.", "e.g.", "i.e.", "vs.")):
            continue
        end = match.start()
        sentence = text[start:end].strip()
        if sentence:
            actual_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((actual_start, actual_start + len(sentence), sentence))
        start = match.end()
    sentence = text[start:].strip()
    if sentence:
        actual_start = start + len(text[start:]) - len(text[start:].lstrip())
        spans.append((actual_start, actual_start + len(sentence), sentence))
    return spans


def containing_sentence(text: str, span_start: int, span_end: int) -> tuple[int, int, str]:
    for start, end, sentence in sentence_spans(text):
        if start <= span_start and span_end <= end:
            return start, end, sentence
    return 0, len(text), text


def complete_provenance(paragraph: dict[str, Any]) -> bool:
    provenance = paragraph.get("provenance") or {}
    bbox = paragraph.get("bbox")
    source_bbox = provenance.get("source_bbox")
    return all(
        (
            paragraph.get("paragraph_id"),
            paragraph.get("page") is not None,
            isinstance(bbox, list) and len(bbox) == 4,
            provenance.get("source_file"),
            provenance.get("source_page") is not None,
            isinstance(source_bbox, list) and len(source_bbox) == 4,
            provenance.get("parser_source"),
            provenance.get("source_index") is not None,
        )
    )


def body_section_gate(major_section: str | None, section_path: list[str] | None) -> tuple[bool, str]:
    label = classify_section(major_section, section_path)
    if label == "excluded":
        return False, "excluded_structural_section"
    return True, label


def is_reference_like(text: str) -> bool:
    return bool(REFERENCE_LIKE_RE.search(text))


def semantic_context(text: str) -> bool:
    words = WORD_RE.findall(text)
    return len(text) >= 18 and len(words) >= 4 and not CITATION_ONLY_RE.fullmatch(text)


def explicit_flags(text: str) -> tuple[bool, bool]:
    lowered = f" {text.casefold()} "
    negation = any(cue in lowered for cue in NEGATION_CUES)
    uncertainty = any(re.search(rf"\b{re.escape(cue)}\w*\b", lowered) for cue in UNCERTAINTY_CUES)
    return negation, uncertainty


def acronym_matches_long(abbreviation: str, long_form: str) -> bool:
    abbr = re.sub(r"[^A-Z0-9]", "", abbreviation.upper())
    words = [re.sub(r"[^A-Za-z0-9]", "", word) for word in long_form.split()]
    words = [word for word in words if word]
    if not (2 <= len(abbr) <= 10) or not words:
        return False
    initials = "".join(word[0].upper() for word in words if word.casefold() not in {"of", "the", "and", "for", "in", "to", "on", "with"})
    if initials == abbr or initials.endswith(abbr):
        return True
    source = long_form.upper()
    cursor = len(source) - 1
    for abbreviation_index in range(len(abbr) - 1, -1, -1):
        char = abbr[abbreviation_index]
        found = source.rfind(char, 0, cursor + 1)
        if abbreviation_index == 0:
            while found > 0 and source[found - 1].isalnum():
                found = source.rfind(char, 0, found)
        if found < 0:
            return False
        cursor = found - 1
    return True


def aligned_long_form(abbreviation: str, raw_long_form: str) -> tuple[str, int] | None:
    """Return the shortest aligned word suffix and its offset in raw_long_form."""
    words = list(re.finditer(r"[A-Za-z][A-Za-z0-9'/-]*", raw_long_form))
    if not words:
        return None
    max_words = min(len(words), max(len(re.sub(r"[^A-Z0-9]", "", abbreviation)) + 5, 2 * len(abbreviation)))
    earliest = max(0, len(words) - max_words)
    for index in range(len(words) - 1, earliest - 1, -1):
        suffix_start = words[index].start()
        suffix = raw_long_form[suffix_start:].strip()
        if acronym_matches_long(abbreviation, suffix):
            return suffix, suffix_start
    return None


def stable_id(parts: Iterable[Any], prefix: str = "cand") -> str:
    payload = "|".join("" if value is None else str(value) for value in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def valid_span(evidence: str, start: int | None, end: int | None, answer: str | None) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(evidence)
        and evidence[start:end] == answer
    )


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
