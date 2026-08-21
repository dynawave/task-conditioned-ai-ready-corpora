# Data dictionary

## Common identifiers

- `paper_id`: release-only SHA-256-derived pseudonym; no title, DOI, or source text is provided.
- `instance_id`, `candidate_id`: pre-existing opaque deterministic identifiers.
- `audit_id`, `sample_id`: frozen T3 semantic-audit membership identifiers.
- `task`: `T1a_ACRONYM`, `T1b_EXPLICIT_DEFINITION`, `T2_NUMERIC`, or `T3_CLAIM` as applicable.

## RQ2 feature/OOF data

- `source_tokens`: source-document token count used for the offset and token budget.
- `machine_verified_count`, `actual_mv_count`: task-specific machine-verified opportunity yield; not human annotation accuracy.
- `oof_predicted_count`, `oof_predicted_rate_per_10k`: five-fold out-of-fold NB2 predictions.
- `fold`, `fold_train_n`, `fold_test_n`, `model_converged`, `alpha`: frozen cross-fitting metadata.
- Structural predictors are numeric document-engineering counts/densities; no document text is included.

## RQ3 membership/results

- `scale`: nested training-set size.
- `seed`: training seed (42, 314159, or 271828).
- `NEM`: normalized exact-match score on frozen paper-disjoint evaluation instances.
- `point_difference`: large-scale minus small-scale NEM unless a result file explicitly names another direction.
- `one_sided_ucb95`: 95% upper confidence bound for the reference-minus-candidate difference used by the practical-equivalence rule.
- `delta_nem`: fixed equivalence tolerance, primary value 0.02.

## MJSA labels

- Four dimensions: `evidence_support`, `factual_correctness`, `condition_completeness`, `answer_uniqueness`.
- `overall_judgment`: Accept, Borderline, or Reject.
- Strict majority Accept: at least two Accept judgments and zero Reject judgments.
- `complete_agreement`: all three judges assign the same overall category.
- `Short_Reason`, `Evidence_Context`, and `Conditional_Claim` are intentionally omitted.
