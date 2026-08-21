from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from quality_gates import (
    URL_DOI_RE,
    aligned_long_form,
    acronym_matches_long,
    body_section_gate,
    complete_provenance,
    containing_sentence,
    explicit_flags,
    json_compact,
    normalize_key,
    normalize_text,
    semantic_context,
    sentence_spans,
    stable_id,
    valid_span,
)
from rules import RELATION_CUES, T1_PATTERNS, T2_PATTERNS, UNIT_PATTERN, classify_section


T1_REGEX = {name: re.compile(pattern) for name, pattern in T1_PATTERNS.items()}
T2_REGEX = {name: re.compile(pattern, re.I if name in {"percentage", "statistical_ci"} else 0) for name, pattern in T2_PATTERNS.items()}
UNIT_RE = re.compile(UNIT_PATTERN)
RELATION_RE = re.compile(
    r"\b(?P<cue>"
    + "|".join(
        [
            r"significant(?:ly)?",
            *[re.escape(cue) + (r"(?:d|s|ing)?" if cue in {"increase", "decrease", "improve", "reduce", "affect"} else "") for cue in sorted(RELATION_CUES, key=len, reverse=True) if cue != "significant"],
        ]
    )
    + r")\b",
    re.I,
)
STRUCTURAL_NUMERIC_RE = re.compile(r"(?:fig(?:ure)?|table|eq(?:uation)?|section|page|pp?|vol(?:ume)?|no)\.?\s*$", re.I)
STAT_CONTEXT_RE = re.compile(r"\b(?:significant|difference|effect|association|correlation|compared|higher|lower|increase|decrease|model|regression|test)\w*\b", re.I)
METHOD_RESULT_RE = re.compile(r"\b(?:performance|accuracy|precision|recall|f1|outperform|compared|higher|lower|resulted|yield|error|improved|reduced)\w*\b", re.I)
INVALID_EXPLICIT_TERM_RE = re.compile(
    r"\b(?:provides?|acquired|evaluated|trained|computed|present|allows?|refined|this|that|which|we|it)\b",
    re.I,
)


def _base_candidate(
    document: dict[str, Any],
    paragraph: dict[str, Any],
    section: dict[str, Any] | None,
    task_type: str,
    subtype: str,
    evidence: str,
    answer: str,
    answer_start: int,
    answer_end: int,
    anchor: tuple[int, int, str],
    identity_suffix: str,
) -> dict[str, Any]:
    section = section or {}
    negation, uncertainty = explicit_flags(anchor[2])
    candidate_id = stable_id(
        [task_type, document.get("source_id"), paragraph.get("paragraph_id"), subtype, answer_start, answer_end, identity_suffix]
    )
    return {
        "candidate_id": candidate_id,
        "task_type": task_type,
        "task_subtype": subtype,
        "paper_id": document.get("paper_id"),
        "source_id": document.get("source_id"),
        "section_id": paragraph.get("section_id"),
        "section_path": json_compact(section.get("section_path")) if section.get("section_path") is not None else None,
        "major_section": section.get("major_section"),
        "section_class": classify_section(section.get("major_section"), section.get("section_path")),
        "paragraph_id": paragraph.get("paragraph_id"),
        "evidence_raw": paragraph.get("raw_text"),
        "evidence_clean": evidence,
        "page": paragraph.get("page"),
        "bbox": json_compact(paragraph.get("bbox")) if paragraph.get("bbox") is not None else None,
        "answer_span": answer,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "answer_normalized": normalize_text(answer),
        "anchor_text": anchor[2],
        "anchor_start": anchor[0],
        "anchor_end": anchor[1],
        "condition_spans": json_compact({
            "population_or_material": None,
            "treatment": None,
            "comparison": None,
            "method": None,
            "spatial": None,
            "temporal": None,
            "quantity_or_threshold": None,
            "polarity": None,
            "negation": None,
            "uncertainty": None,
        }),
        "negation_flag": negation,
        "uncertainty_flag": uncertainty,
        "provenance": json_compact(paragraph.get("provenance")) if paragraph.get("provenance") is not None else None,
        "source_path": document.get("source_path"),
        "raw_opportunity": True,
        "hard_gate_pass": False,
        "materialization_ready": False,
        "stage": "RAW_OPPORTUNITY",
        "auto_reject_reason": None,
        "auto_flags": json_compact([]),
        "prototype_only": True,
        "term": None,
        "definition": None,
        "unit": None,
        "relation": None,
        "subject": None,
        "outcome": None,
        "support_paragraph_ids": json_compact([]),
        "support_evidence_clean": json_compact([]),
        "duplicate_key": None,
    }


def _finish(candidate: dict[str, Any], reject_reason: str | None, flags: list[str], ready: bool) -> dict[str, Any]:
    candidate["auto_reject_reason"] = reject_reason
    candidate["hard_gate_pass"] = reject_reason is None
    candidate["materialization_ready"] = bool(reject_reason is None and ready)
    candidate["stage"] = (
        "MATERIALIZATION_READY" if candidate["materialization_ready"]
        else "HARD_GATE_PASS" if candidate["hard_gate_pass"]
        else "RAW_OPPORTUNITY"
    )
    candidate["auto_flags"] = json_compact(sorted(set(flags)))
    return candidate


def detect_t1(document: dict[str, Any], section_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paragraph in document.get("paragraphs") or []:
        evidence = paragraph.get("clean_text") or ""
        section = section_by_id.get(paragraph.get("section_id"))
        for pattern_name, pattern in T1_REGEX.items():
            for match in pattern.finditer(evidence):
                if pattern_name == "explicit_definition":
                    term = normalize_text(match.group("term"))
                    definition = normalize_text(match.group("definition"))
                    answer_start, answer_end = match.span("definition")
                    cue = match.group("cue").casefold()
                    if term.casefold().startswith("where "):
                        term = term.split()[-1]
                    coordinated = re.search(r"\s+and\s+[A-Za-z0-9_-]{1,12}\s+(?:is|denotes|means|refers\s+to)\b", definition, re.I)
                    if coordinated:
                        definition = definition[:coordinated.start()].rstrip()
                        answer_end = answer_start + len(definition)
                else:
                    term = normalize_text(match.group("abbr"))
                    raw_definition = match.group("long")
                    aligned = aligned_long_form(term, raw_definition)
                    definition = normalize_text(aligned[0] if aligned else raw_definition)
                    answer_start = match.start("long") + (aligned[1] if aligned else 0)
                    answer_end = match.end("long")
                answer = evidence[answer_start:answer_end]
                anchor = containing_sentence(evidence, answer_start, answer_end)
                candidate = _base_candidate(
                    document, paragraph, section, "T1_TERM", pattern_name, evidence, answer,
                    answer_start, answer_end, anchor, f"{term}|{definition}",
                )
                candidate["term"] = term
                candidate["definition"] = definition
                duplicate_key = f"{normalize_key(term)}|{normalize_key(definition)}"
                candidate["duplicate_key"] = duplicate_key
                flags = [f"detector:{pattern_name}"]
                if paragraph.get("suspected_word_split"):
                    flags.append("suspected_word_split")
                reject: str | None = None
                section_ok, _ = body_section_gate(section.get("major_section") if section else None, section.get("section_path") if section else None)
                if not section_ok:
                    reject = "excluded_structural_section"
                elif not complete_provenance(paragraph):
                    reject = "incomplete_provenance"
                elif not valid_span(evidence, answer_start, answer_end, answer):
                    reject = "answer_span_not_traceable"
                elif not term or not definition or len(re.findall(r"[A-Za-z]", definition)) < 3:
                    reject = "empty_or_nonlexical_definition"
                elif term.isdigit() or URL_DOI_RE.search(term):
                    reject = "invalid_term_url_doi_or_number"
                elif pattern_name == "explicit_definition" and (
                    INVALID_EXPLICIT_TERM_RE.search(term)
                    or term.casefold().split()[-1] in {"a", "an", "the", "by", "of", "and"}
                    or (cue == "means" and definition.casefold().split()[0] in {"of", "to", "and"})
                ):
                    reject = "nondefinitional_cue_usage"
                elif pattern_name == "explicit_definition" and answer_end < len(evidence) and evidence[answer_end:answer_end + 1].isalpha():
                    reject = "truncated_definition_span"
                elif pattern_name != "explicit_definition" and not acronym_matches_long(term, definition):
                    reject = "acronym_long_form_alignment_failed"
                elif duplicate_key in seen:
                    reject = "duplicate_term_definition_within_paper"
                if reject is None:
                    seen.add(duplicate_key)
                candidates.append(_finish(candidate, reject, flags, ready=reject is None))
    return candidates


def _numeric_unit(answer: str, subtype: str) -> str | None:
    if subtype == "percentage":
        return "%"
    if subtype == "statistical_value" or subtype == "other_numeric":
        return None
    matches = list(UNIT_RE.finditer(answer))
    return matches[-1].group(0) if matches else None


def _numeric_subtype(pattern_name: str) -> str:
    if pattern_name in {"statistical_p", "statistical_ci", "statistical_test"}:
        return "statistical_value"
    if pattern_name == "year_like":
        return "other_numeric"
    return pattern_name


def detect_t2(document: dict[str, Any], section_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    priority = ["mean_dispersion", "range", "percentage", "statistical_ci", "statistical_p", "statistical_test", "measurement", "year_like"]
    for paragraph in document.get("paragraphs") or []:
        evidence = paragraph.get("clean_text") or ""
        section = section_by_id.get(paragraph.get("section_id"))
        occupied: list[tuple[int, int]] = []
        for pattern_name in priority:
            for match in T2_REGEX[pattern_name].finditer(evidence):
                start, end = match.span("value")
                if any(start < prior_end and prior_start < end for prior_start, prior_end in occupied):
                    continue
                occupied.append((start, end))
                answer = evidence[start:end]
                anchor = containing_sentence(evidence, start, end)
                subtype = _numeric_subtype(pattern_name)
                candidate = _base_candidate(
                    document, paragraph, section, "T2_NUMERIC", subtype, evidence, answer,
                    start, end, anchor, f"{pattern_name}|{answer}",
                )
                unit = _numeric_unit(answer, subtype)
                candidate["unit"] = unit
                candidate["answer_normalized"] = normalize_text(answer).replace(",", "").replace("−", "-")
                duplicate_key = f"{subtype}|{normalize_key(answer)}|{normalize_key(anchor[2])}"
                candidate["duplicate_key"] = duplicate_key
                flags = [f"detector:{pattern_name}"]
                if paragraph.get("suspected_word_split"):
                    flags.append("suspected_word_split")
                reject: str | None = None
                section_ok, _ = body_section_gate(section.get("major_section") if section else None, section.get("section_path") if section else None)
                context_prefix = evidence[max(0, start - 24):start]
                if pattern_name == "year_like" or re.fullmatch(r"(?:18|19|20|21)\d{2}s", answer):
                    flags.append("year_like")
                    reject = "year_like_1800_2100"
                elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)[A-Za-z]", answer):
                    reject = "ambiguous_compact_single_letter_unit"
                elif not section_ok:
                    reject = "excluded_structural_section"
                elif not complete_provenance(paragraph):
                    reject = "incomplete_provenance"
                elif not valid_span(evidence, start, end, answer):
                    reject = "answer_span_not_traceable"
                elif URL_DOI_RE.search(anchor[2]) and URL_DOI_RE.search(evidence[max(0, start - 40):min(len(evidence), end + 40)]):
                    reject = "doi_or_url_numeric"
                elif STRUCTURAL_NUMERIC_RE.search(context_prefix) or re.search(r"\[\s*$", context_prefix):
                    reject = "bibliographic_or_structural_numeric"
                elif not semantic_context(anchor[2]):
                    reject = "insufficient_semantic_context"
                elif subtype not in {"statistical_value", "other_numeric", "percentage"} and not unit:
                    reject = "unit_not_retained"
                elif duplicate_key in seen:
                    reject = "duplicate_numeric_fact_within_paper"
                if reject is None:
                    seen.add(duplicate_key)
                ready = reject is None
                if reject is None and subtype == "statistical_value" and not STAT_CONTEXT_RE.search(anchor[2]):
                    ready = False
                    flags.append("materialization_requires_unambiguous_statistical_relation")
                candidates.append(_finish(candidate, reject, flags, ready=ready))
    return candidates


def _span_value(sentence: str, pattern: str, anchor_start: int) -> dict[str, Any] | None:
    match = re.search(pattern, sentence, re.I)
    if not match:
        return None
    return {"text": match.group(0), "start": anchor_start + match.start(), "end": anchor_start + match.end()}


def _claim_conditions(
    sentence: str, anchor_start: int, subject: str, subject_offset: int,
    outcome: str, outcome_offset: int, relation_match: re.Match[str],
) -> dict[str, Any]:
    quantity = _span_value(sentence, rf"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(?:{UNIT_PATTERN}|%|percent)", anchor_start)
    comparison = _span_value(sentence, r"\b(?:compared (?:with|to)|relative to|versus|vs\.?|than)\s+[^,;.]{1,90}", anchor_start)
    treatment = _span_value(sentence, r"\b(?:treated with|exposed to|subjected to|under)\s+[^,;.]{1,80}", anchor_start)
    method = _span_value(sentence, r"\b(?:using|measured by|estimated (?:with|using)|based on)\s+[^,;.]{1,90}", anchor_start)
    spatial = _span_value(sentence, r"\b(?:at Site\s+[A-Za-z0-9-]+|in the\s+[A-Za-z][A-Za-z -]{2,50}\s+(?:region|area|basin|margin|ocean|sea|soil|sediment))", anchor_start)
    temporal = _span_value(sentence, r"\b(?:during|after|before|over the past|throughout)\s+[^,;.]{1,80}", anchor_start)
    negation = _span_value(sentence, r"\b(?:not|no|neither|without|did not|was not)\b", anchor_start)
    uncertainty = _span_value(sentence, r"\b(?:may|might|could|suggest\w*|likely|possibly|potentially|appear\w*)\b", anchor_start)
    cue = relation_match.group("cue")
    cue_lower = cue.casefold()
    if any(x in cue_lower for x in ("increase", "higher", "improv", "promot")):
        polarity = "positive"
    elif any(x in cue_lower for x in ("decrease", "lower", "reduc", "inhibit")):
        polarity = "negative"
    else:
        polarity = "neutral_or_unspecified"
    return {
        "population_or_material": {"text": subject, "start": anchor_start + subject_offset, "end": anchor_start + subject_offset + len(subject)} if subject else None,
        "treatment": treatment,
        "comparison": comparison,
        "method": method,
        "spatial": spatial,
        "temporal": temporal,
        "quantity_or_threshold": quantity,
        "polarity": {"value": polarity, "cue": cue, "start": anchor_start + relation_match.start(), "end": anchor_start + relation_match.end()},
        "negation": negation,
        "uncertainty": uncertainty,
        "outcome": {"text": outcome, "start": anchor_start + outcome_offset, "end": anchor_start + outcome_offset + len(outcome)} if outcome else None,
    }


def detect_t3(
    document: dict[str, Any],
    section_by_id: dict[str, dict[str, Any]],
    adjacent_by_paragraph: dict[str, tuple[list[str], list[str]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paragraph in document.get("paragraphs") or []:
        evidence = paragraph.get("clean_text") or ""
        section = section_by_id.get(paragraph.get("section_id"))
        section_class = classify_section(section.get("major_section") if section else None, section.get("section_path") if section else None)
        for anchor_start, anchor_end, sentence in sentence_spans(evidence):
            relation_match = RELATION_RE.search(sentence)
            if relation_match is None:
                continue
            answer_start, answer_end = anchor_start, anchor_end
            answer = evidence[answer_start:answer_end]
            raw_prefix = sentence[:relation_match.start()]
            raw_suffix = sentence[relation_match.end():]
            prefix = raw_prefix.strip(" ,;:-")
            suffix = raw_suffix.strip(" ,;:-")
            prefix_offset = len(raw_prefix) - len(raw_prefix.lstrip(" ,;:-"))
            suffix_offset = relation_match.end() + len(raw_suffix) - len(raw_suffix.lstrip(" ,;:-"))
            subject = prefix
            outcome = suffix
            candidate = _base_candidate(
                document, paragraph, section, "T3_CLAIM", "extractive_relation_claim", evidence,
                answer, answer_start, answer_end, (anchor_start, anchor_end, sentence),
                f"{relation_match.group('cue')}|{answer}",
            )
            candidate["answer_normalized"] = normalize_text(answer)
            candidate["relation"] = relation_match.group("cue")
            candidate["subject"] = subject or None
            candidate["outcome"] = outcome or None
            candidate["condition_spans"] = json_compact(
                _claim_conditions(sentence, anchor_start, subject, prefix_offset, outcome, suffix_offset, relation_match)
            )
            support_ids, support_texts = adjacent_by_paragraph.get(paragraph.get("paragraph_id"), ([], []))
            candidate["support_paragraph_ids"] = json_compact(support_ids[:2])
            candidate["support_evidence_clean"] = json_compact(support_texts[:2])
            duplicate_key = normalize_key(answer)
            candidate["duplicate_key"] = duplicate_key
            flags = ["detector:explicit_relation_cue", f"section:{section_class}"]
            if paragraph.get("suspected_word_split"):
                flags.append("suspected_word_split")
            reject: str | None = None
            if section_class == "excluded":
                reject = "excluded_structural_section"
            elif section_class in {"introduction", "other", "abstract"}:
                reject = "background_or_ineligible_section"
            elif section_class == "methods" and not METHOD_RESULT_RE.search(sentence):
                reject = "methods_without_performance_or_comparison_result"
            elif not complete_provenance(paragraph):
                reject = "incomplete_provenance"
            elif not valid_span(evidence, answer_start, answer_end, answer):
                reject = "claim_span_not_traceable"
            elif not re.search(r"[.!?][\"']?$", sentence):
                reject = "incomplete_statement_no_terminal_punctuation"
            elif not semantic_context(sentence) or len(sentence) > 500:
                reject = "incomplete_or_nonsemantic_statement"
            elif len(re.findall(r"[A-Za-z][A-Za-z'-]*", subject)) < 2:
                reject = "unidentified_claim_subject"
            elif len(re.findall(r"[A-Za-z][A-Za-z'-]*", outcome)) < 2:
                reject = "unidentified_claim_outcome"
            elif duplicate_key in seen:
                reject = "duplicate_claim_within_paper"
            if reject is None:
                seen.add(duplicate_key)
            ready = reject is None and len(sentence) <= 350
            if reject is None and not ready:
                flags.append("materialization_deferred_long_claim")
            candidates.append(_finish(candidate, reject, flags, ready=ready))
    return candidates


def detect_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    sections = document.get("sections") or []
    section_by_id = {section.get("section_id"): section for section in sections}
    paragraphs = document.get("paragraphs") or []
    grouped: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for paragraph in paragraphs:
        grouped[paragraph.get("section_id")].append(paragraph)
    adjacent: dict[str, tuple[list[str], list[str]]] = {}
    for section_paragraphs in grouped.values():
        for index, paragraph in enumerate(section_paragraphs):
            neighbors: list[dict[str, Any]] = []
            if index > 0:
                neighbors.append(section_paragraphs[index - 1])
            if index + 1 < len(section_paragraphs):
                neighbors.append(section_paragraphs[index + 1])
            adjacent[paragraph.get("paragraph_id")] = (
                [item.get("paragraph_id") for item in neighbors],
                [item.get("clean_text") or "" for item in neighbors],
            )
    candidates = detect_t1(document, section_by_id)
    candidates.extend(detect_t2(document, section_by_id))
    candidates.extend(detect_t3(document, section_by_id, adjacent))
    return candidates
