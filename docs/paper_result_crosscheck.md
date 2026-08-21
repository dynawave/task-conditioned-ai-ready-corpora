# Paper-result cross-check

Status: **PASS** against the frozen files in this release.

| Area | Frozen value | Release source |
|---|---:|---|
| Processable papers | 30,883 | `results/corpus/asset_readiness_summary_31k_v2.json` |
| S2K papers | 2,000 | `results/corpus/s2k_summary.json` |
| S2K body paragraphs | 114,511 | same |
| S2K clean tokens | 21,013,210 | same |
| RAG formal questions | 1,292 | `results/rq1/rag/summary.json` |
| Mapped evidence | 5,420 / 5,648 (95.96%) | same |
| Fixed CES@5 | 0.688854 | same |
| Atomic BudgetCES@1024 | 0.650155 | `results/rq1/rag/budget_final_v2/budget_metrics.csv` |
| RQ2 20% paper budget, T1a/T2/T3 | 0.274083 / 0.332739 / 0.346993 | `results/rq2/final_robustness/rq2_common_vs_specific.csv` |
| RQ2 20% token budget, T1a/T2/T3 | 0.240275 / 0.217828 / 0.416460 | same |
| Qwen2.5-3B n* | T1a 1,000; T2 3,000; T3 not reached at 2,286 | `results/rq3/full_grid/n_star_summary.csv`, `scale_extension/n_star_summary.csv` |
| Qwen2.5-7B key-interval ΔNEM | +0.010000 / +0.001111 / +0.036405 | `results/rq3/qwen7b/qwen7b_key_interval_bootstrap.csv` |
| Phi key-interval ΔNEM | +0.001111 / +0.003333 / −0.005688 | `results/rq3/phi/bootstrap.csv` |
| MJSA strict majority Accept | 135/150 (90.0%) | `results/mjsa/mjsa_summary.json` |
| MJSA complete agreement | 109/150 (72.7%) | same |
| MJSA raw pairwise agreement | 81.6% | same |
| MJSA Fleiss κ | 0.228 | same |

Run `python code/common/validate_release.py` to repeat these checks. The check is read-only and does not regenerate formal results.
