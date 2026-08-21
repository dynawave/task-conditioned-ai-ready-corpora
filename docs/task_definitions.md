# Task and view definitions

## RAG representations

- **B1 Fixed**: fixed-length document units.
- **B2 Atomic**: section-aware atomic retrieval units.
- **B3 SEP**: atomic units with section-path context.

CES@5 is complete evidence-set recovery within the top five units. BudgetCES@B follows the ranked list and adds a whole retrieval unit only when the cumulative token count remains within budget B; units are never truncated. BudgetRecall uses the same prefix.

## SFT tasks

- **T1a_ACRONYM**: expand a scientific abbreviation/short form to its long form.
- **T2_NUMERIC**: return a normalized numerical fact, retaining required value, unit, and operator information.
- **T3_CLAIM**: return a conditional scientific claim while preserving evidence support, factual correctness, conditions/qualifiers, and answer uniqueness.

Primary evaluation is paper-disjoint normalized exact match (NEM) under TEMPLATE_0. TEMPLATE_1/2 are secondary instruction-template robustness checks. Task detectors, rules, and quality gates are released as code; copyrighted prompt/evidence/target rows are not.
