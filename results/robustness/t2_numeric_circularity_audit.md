# T2 numeric eligibility and circularity audit

## QASPER

The frozen code iterates human extractive spans, retains spans satisfying the strict numeric/range/percent/number+unit/statistical predicate, requires nonempty human evidence, and requires normalized answer containment in at least one human-annotated evidence string. The final N=126 is reproduced exactly. Exact character containment is not an eligibility gate, but normalized containment is. An explicit unit is not required: number-only, range, percent, and statistical forms can qualify.

## SciTaT

The frozen code starts from official test items, retains lookup-only question types, requires exactly one strict numeric final answer, and requires normalized answer containment in the supplied table/text. The final N=5 is reproduced exactly. The small N is principally the result of a deliberately narrow, non-calculation lookup-only and recoverable-answer gate; it is not evidence that SciTaT itself contains only five numeric examples.

## Eligibility-by-metric overlap matrix

| eligibility_rule | evaluation_metric | overlap | reason |
|---|---|---|---|
| strict_numeric_schema | numeric_recovery | DIRECT_OVERLAP | numeric_signature re-applies the same strict numeric predicate |
| strict_numeric_schema | exact_span_match | NO_OVERLAP | exact character containment is not required by numeric syntax |
| strict_numeric_schema | unit_preservation | PARTIAL_OVERLAP | numeric schema permits number-only, range, percent, unit, and statistical forms; unit is not mandatory |
| strict_numeric_schema | human_annotated_evidence_support | NO_OVERLAP | numeric syntax does not establish evidence support |
| nonempty_human_annotated_evidence | numeric_recovery | NO_OVERLAP | evidence presence does not determine numeric parsing |
| nonempty_human_annotated_evidence | exact_span_match | PARTIAL_OVERLAP | an evidence container is required but exact containment is not |
| nonempty_human_annotated_evidence | unit_preservation | PARTIAL_OVERLAP | unit/value checking is evaluated only inside required evidence |
| nonempty_human_annotated_evidence | human_annotated_evidence_support | DIRECT_OVERLAP | the metric uses the same required human evidence field |
| normalized_answer_present_in_gold_evidence | numeric_recovery | NO_OVERLAP | normalized containment does not determine numeric parsing |
| normalized_answer_present_in_gold_evidence | exact_span_match | PARTIAL_OVERLAP | eligibility accepts normalized containment; exact containment is a stricter diagnostic |
| normalized_answer_present_in_gold_evidence | unit_preservation | DIRECT_OVERLAP | reported unit_preserved is numeric signature plus the same normalized evidence match |
| normalized_answer_present_in_gold_evidence | human_annotated_evidence_support | DIRECT_OVERLAP | reported evidence support is the same normalized evidence-match gate |

## Interpretation

QASPER's four 1.0 values are best described as a **deterministic recoverability/consistency audit**, not external validator performance or independent accuracy. Numeric recovery, evidence support, and the reported unit-preservation computation directly reuse eligibility predicates; exact-span recovery is a stricter diagnostic than the normalized-containment gate but remains conditioned on that gate. SciTaT is likewise a narrow secondary recoverability diagnostic. The v86 files were unavailable, so their exact wording is `NOT RECOVERABLE`; any wording calling these four values independent external accuracy should be downgraded.
