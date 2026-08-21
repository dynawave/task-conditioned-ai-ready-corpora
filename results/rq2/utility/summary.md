# Source Suitability Utility Validation V1

## OOF integrity

- T1a_ACRONYM: `OOF_SUCCESS`
- T1b_EXPLICIT_DEFINITION: `OOF_SUCCESS`
- T2_NUMERIC: `OOF_SUCCESS`
- T3_CLAIM: `OOF_SUCCESS`

## Primary 20% budget results

| Task | Budget | Random capture | Task-specific capture | Gain | Relative lift | 95% effect CI |
|---|---|---:|---:|---:|---:|---:|
| T1a_ACRONYM | paper | 0.2000 | 0.2741 | +0.0740 | 1.370 | [+0.0500, +0.0958] |
| T1a_ACRONYM | token | 0.1998 | 0.2403 | +0.0405 | 1.203 | [+0.0180, +0.0615] |
| T2_NUMERIC | paper | 0.2000 | 0.3327 | +0.1327 | 1.663 | [+0.1088, +0.1571] |
| T2_NUMERIC | token | 0.1994 | 0.2178 | +0.0184 | 1.092 | [-0.0014, +0.0368] |
| T3_CLAIM | paper | 0.2007 | 0.3470 | +0.1463 | 1.729 | [+0.1134, +0.1805] |
| T3_CLAIM | token | 0.2004 | 0.4165 | +0.2161 | 2.078 | [+0.1796, +0.2493] |

## Judgments

- T1a_ACRONYM: `SUPPORTED`
- T2_NUMERIC: `DIRECTIONALLY_SUPPORTED`
- T3_CLAIM: `SUPPORTED`
- Overall: `SOURCE_SELECTION_UTILITY_SUPPORTED`

## Cross-task ranking

- T1a_ACRONYM vs T2_NUMERIC: Spearman=0.4278; Top20 Jaccard=0.1639.
- T1a_ACRONYM vs T3_CLAIM: Spearman=0.7203; Top20 Jaccard=0.3919.
- T2_NUMERIC vs T3_CLAIM: Spearman=0.4204; Top20 Jaccard=0.2044.

## 100-paper illustration

- T1a_ACRONYM: task-specific=1026.0, random mean=716.6, additional=+309.4.
- T2_NUMERIC: task-specific=4580.0, random mean=2191.5, additional=+2388.5.
- T3_CLAIM: task-specific=1299.0, random mean=618.8, additional=+680.2.

## Interpretation

Recommended placement: **A. Main Results**.

Strict wording: “Under fixed processing budgets, cross-fitted task-specific source rankings showed predictive selection utility for machine-verified corpus yield; the magnitude and stability were task dependent.”

These are predictive selection results, not causal effects or measures of scientific paper quality. ORACLE is an upper-bound diagnostic only. T1b remains supplementary regardless of its result.
