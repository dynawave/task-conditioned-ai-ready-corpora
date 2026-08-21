# Quality gates

## Document engineering

The canonical document schema separates body paragraphs from titles, front matter, references, headers/footers, page numbers, and uncertain elements; preserves page/bounding-box provenance; and records figure/table/formula object coverage. The final S2K run is `READY_FOR_RAG` with 2,000 successful papers.

## RAG

- QASPER v0.3 test; 1,292 formally evaluable questions.
- Gold mapping gate: at least 95%; observed 5,420/5,648 (95.96%).
- B1/B2/B3 definitions, embedding revision, tokenizer, paper-scoped retrieval, and token counting are frozen.
- Budget evaluation requires `budget_prefix_incomplete = 0` at 512/1024/2048.
- Top-k regression must exactly match the preceding frozen Top-5 results.
- Paired bootstrap uses 10,000 replicates; multiple comparisons use Holm correction.

## Source suitability

- Formal effective sample: 1,898 papers.
- NB2 with log source-token offset, five-fold cross-fitting, fold-local standardization, and frozen OOF selection.
- Paper- and token-budget selection uses whole papers and performance-independent stopping rules.
- Random baseline seed 20260817; 10,000 repeats in the final robustness analysis.

## SFT scaling

- Train/eval papers are disjoint; nested membership is fixed and audited by hashes.
- At most three instances per paper in the extension pool; no resampling based on model results.
- Fresh base and fresh LoRA for every task/scale/seed run.
- Assistant-only labels, finite loss, zero target truncation, complete predictions, deterministic evaluation IDs, and RNG/epoch-order audits.
- Primary NEM evaluation uses TEMPLATE_0; practical equivalence uses one-sided UCB95 and `delta_NEM = 0.02`; sensitivity deltas are 0.01/0.03/0.05.

Machine-verifier and task-specific rule implementations are in `code/corpus_experiments_v1/sft/`; frozen settings are in `configs/sft/`.
