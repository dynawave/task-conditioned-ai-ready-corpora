from __future__ import annotations

import re
from typing import Any


RULES_VERSION = "sft_a_rules_v1"
UNIT_TABLE_VERSION = "scientific_units_v1"
SECTION_MAPPING_VERSION = "canonical_heading_rules_v1"
SELECTION_SEED = "20260811"
AUDIT_SEED = "20260811"

ACRONYM_PATTERN = r"[A-Z][A-Z0-9-]{1,9}"
LONG_FORM_PATTERN = r"[A-Za-z][A-Za-z0-9'/-]*(?:\s+(?:of|the|and|for|in|to|on|with|[A-Za-z][A-Za-z0-9'/-]*)){1,11}"
EXPLICIT_DEFINITION_CUES = ("refers to", "is defined as", "denotes", "means")

RELATION_CUES = (
    "increased", "increase", "decreased", "decrease", "higher", "lower",
    "associated", "correlated", "resulted", "led", "promoted", "inhibited",
    "improved", "reduced", "affected", "related", "effect", "significant",
)

NEGATION_CUES = ("not", "no ", "neither", "without", "didn't", "did not", "wasn't", "was not")
UNCERTAINTY_CUES = ("may", "might", "could", "suggest", "likely", "possibly", "potentially", "appears")

UNITS = (
    "%", "°C", "°F", "K", "km", "m", "cm", "mm", "μm", "um", "nm",
    "km2", "km²", "m2", "m²", "cm2", "cm²", "mm2", "mm²",
    "kg", "g", "mg", "μg", "ug", "ng", "pg",
    "L", "mL", "ml", "μL", "uL", "mol", "mmol", "μmol", "umol",
    "M", "mM", "μM", "uM", "nM", "pM", "mg/L", "g/L", "ng/mL",
    "Pa", "kPa", "MPa", "bar", "mbar", "Hz", "kHz", "MHz", "GHz",
    "s", "ms", "min", "h", "hr", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "yr", "yrs", "ka", "kyr", "Ma", "Myr",
    "m/s", "km/h", "mm/yr", "mm/y", "cm/yr", "m/yr", "kg/m2", "kg/m²",
    "ng/g", "μg/g", "ug/g", "mg/g", "g/kg", "mg/kg", "μg/kg", "ug/kg",
    "ppm", "ppb", "ppt", "‰", "Sv", "W", "kW", "MW", "J", "kJ", "MJ",
    "N", "mN", "dB", "cells/mL", "CFU/mL", "bp", "kb", "Mb", "Gb",
)

EXCLUDED_SECTION_TERMS = (
    "reference", "bibliograph", "acknowledg", "funding", "author contribution",
    "credit authorship", "competing interest", "conflict of interest", "data availability",
    "supplementary", "appendix", "article info", "orcid", "citation", "declaration",
    "ethics statement", "publisher", "copyright", "keywords",
)

SECTION_PATTERNS = {
    "results": ("result", "finding"),
    "discussion": ("discussion",),
    "conclusion": ("conclusion", "summary"),
    "methods": ("method", "material", "experimental", "methodology", "statistical analysis"),
    "abstract": ("abstract",),
    "introduction": ("introduction", "background"),
}

NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)(?:\s*[×x]\s*10\s*[−-]?\s*\d+)?"
UNIT_PATTERN = "(?:" + "|".join(sorted((re.escape(x) for x in UNITS), key=len, reverse=True)) + ")"

T1_PATTERNS = {
    "long_form_abbreviation": rf"(?P<long>{LONG_FORM_PATTERN})\s*\(\s*(?P<abbr>{ACRONYM_PATTERN})\s*\)",
    "abbreviation_long_form": rf"(?P<abbr>{ACRONYM_PATTERN})\s*\(\s*(?P<long>{LONG_FORM_PATTERN})\s*\)",
    "explicit_definition": rf"(?P<term>[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){{0,4}})\s+(?P<cue>{'|'.join(re.escape(x) for x in EXPLICIT_DEFINITION_CUES)})\s+(?P<definition>[^.;!?]{{3,180}})",
}

T2_PATTERNS = {
    "mean_dispersion": rf"(?<![\w.])(?P<value>{NUMBER}\s*(?:±|\+/-)\s*{NUMBER}\s*{UNIT_PATTERN})(?!\w)",
    "range": rf"(?<![\w.])(?P<value>{NUMBER}\s*(?:–|—|-|to)\s*{NUMBER}\s*{UNIT_PATTERN})(?!\w)",
    "percentage": rf"(?<![\w.])(?P<value>{NUMBER}\s*(?:%|percent(?:age)?))(?!\w)",
    "statistical_p": rf"(?<!\w)(?P<value>[pP]\s*(?:=|<|>|≤|≥)\s*{NUMBER})(?!\w)",
    "statistical_ci": rf"(?<!\w)(?P<value>{NUMBER}\s*%\s*(?:CI|confidence interval)\s*[:=]?\s*[\[(]?\s*{NUMBER}\s*[,–—-]\s*{NUMBER}\s*[\])]?)(?!\w)",
    "statistical_test": rf"(?<!\w)(?P<value>(?:R\s*\^?2|r|F|t|χ2|χ²|z)\s*(?:=|<|>|≤|≥)\s*{NUMBER})(?!\w)",
    "measurement": rf"(?<![\w.])(?P<value>{NUMBER}\s*{UNIT_PATTERN})(?!\w)",
    "year_like": r"(?<![\w.])(?P<value>(?:18\d{2}|19\d{2}|20\d{2}|2100))(?![\w.])",
}


def classify_section(major_section: str | None, section_path: list[str] | None = None) -> str:
    text = " ".join([major_section or "", *(section_path or [])]).casefold()
    if any(term in text for term in EXCLUDED_SECTION_TERMS):
        return "excluded"
    has_results = any(term in text for term in SECTION_PATTERNS["results"])
    has_discussion = any(term in text for term in SECTION_PATTERNS["discussion"])
    if has_results and has_discussion:
        return "results_discussion"
    if has_results:
        return "results"
    if has_discussion:
        return "discussion"
    for label in ("conclusion", "methods", "abstract", "introduction"):
        if any(term in text for term in SECTION_PATTERNS[label]):
            return label
    return "other"


def rules_payload(code_sha256: str) -> dict[str, Any]:
    return {
        "experiment": "corpus_v1_sft_a_calibration_v1",
        "rules_version": RULES_VERSION,
        "prototype_only": True,
        "development_calibration_only": True,
        "llm_generation_or_scoring": False,
        "selection_seed": SELECTION_SEED,
        "audit_seed": AUDIT_SEED,
        "stages": ["RAW_OPPORTUNITY", "HARD_GATE_PASS", "MATERIALIZATION_READY"],
        "accepted_sft_instance_stage_run": False,
        "tasks": {
            "T1_TERM": {
                "patterns": T1_PATTERNS,
                "hard_gates": [
                    "term_and_definition_exact_in_body_evidence", "nonempty_definition",
                    "term_not_url_doi_or_pure_number", "complete_paragraph_provenance",
                    "same_paper_normalized_term_definition_dedup",
                    "acronym_2_to_10_uppercase_digit_hyphen_and_alignment_for_parenthetical_patterns",
                ],
            },
            "T2_NUMERIC": {
                "patterns": T2_PATTERNS,
                "unit_table_version": UNIT_TABLE_VERSION,
                "units": list(UNITS),
                "subtypes": ["measurement", "percentage", "range", "mean_dispersion", "statistical_value", "other_numeric"],
                "hard_gates": [
                    "exact_answer_span", "unit_retained_when_applicable", "semantic_sentence_context",
                    "complete_paragraph_provenance", "bibliographic_doi_page_section_citation_rejection",
                    "year_like_1800_to_2100_flag_not_measurement", "same_paper_duplicate_fact_dedup",
                ],
            },
            "T3_CLAIM": {
                "relation_cues": list(RELATION_CUES),
                "preferred_sections": ["results", "results_discussion", "discussion", "conclusion"],
                "secondary_section": "abstract_when_span_level_provenance_exists",
                "methods_policy": "performance_or_comparison_results_only",
                "max_same_section_adjacent_support_paragraphs": 2,
                "hard_gates": [
                    "complete_statement", "explicit_relation_or_result", "identifiable_subject_and_outcome",
                    "complete_paragraph_provenance", "preserve_explicit_negation_uncertainty_and_conditions",
                    "reject_background_citation_only_heading", "same_paper_normalized_claim_dedup",
                ],
            },
        },
        "section_mapping_version": SECTION_MAPPING_VERSION,
        "section_patterns": SECTION_PATTERNS,
        "excluded_section_terms": list(EXCLUDED_SECTION_TERMS),
        "negation_cues": list(NEGATION_CUES),
        "uncertainty_cues": list(UNCERTAINTY_CUES),
        "dedup_scope": "within_paper_and_task_after_normalization",
        "code_sha256": code_sha256,
    }
