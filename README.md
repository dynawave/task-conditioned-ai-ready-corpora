# From Scientific Literature to AI-Ready Corpora

**From Scientific Literature to AI-Ready Corpora: A Task-Conditioned Framework and Empirical Evidence from Geoscience**

Authors: see published/submitted manuscript

Release version: **v1.0.0**

This release is a whitelist-built, copyright-aware companion to the paper. It supports unified scientific-document engineering, RAG evidence representation, source-paper prioritization, SFT scaling, practical-equivalence analysis, robustness checks, and figure reproduction. Frozen numerical outputs are copied without recomputing or changing the formal experiments.

## Contents

- `code/`: document engineering, RAG, source-prioritization, SFT, statistics, and plotting code.
- `configs/`: sanitized frozen configurations; private local paths are replaced by `<LOCAL_PATH_OMITTED>`.
- `manifests/`: source provenance, anonymous corpus/split membership, and nested SFT membership.
- `data/derived/`: anonymous numeric RQ2 feature and OOF-prediction tables.
- `results/`: frozen aggregate/per-seed metrics and copyright-safe MJSA labels.
- `figures/final/`: final author-generated Figures 1–5 and Supplementary Figure S1.
- `docs/`: schemas, task definitions, quality gates, result cross-check, and reproduction map.

## Not included

This repository does **not** redistribute publisher PDFs, publisher XML/HTML, paper full text, abstracts, evidence paragraphs, raw third-party benchmarks, model weights/adapters, embeddings, checkpoints, API credentials, private caches, smoke/debug artifacts, training stdout/stderr logs, manuscripts, submissions, or reviewer spreadsheets. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md), and [docs/MJSA_RELEASE_POLICY.md](docs/MJSA_RELEASE_POLICY.md).

## Reproduction map

| Paper item | Script | Frozen config | Canonical frozen input/output |
|---|---|---|---|
| Figure 2 / RQ1 | `code/corpus_experiments_v1/plot_ipm_submission_figures.py` | `configs/rag/` | `results/rq1/rag/` |
| Figure 3 / RQ2 | same plotting script | `configs/rq2/`, `configs/robustness/` | `results/rq2/final_robustness/`, `results/rq2/reviewer2/` |
| Figure 4 / RQ3 scaling | same plotting script | `configs/qwen3b/` | `results/rq3/full_grid/`, `results/rq3/scale_extension/` |
| Figure 5 / model replication | same plotting script | `configs/qwen7b/`, `configs/phi/` | `results/rq3/qwen7b/`, `results/rq3/phi/` |
| Supplementary Figure S1 | same plotting script | `configs/rq2/source_suitability_nb2_config.json` | `results/rq2/source_suitability/nb_results.csv` |
| Table 3 | `code/corpus_experiments_v1/rag/run_rag_experiment.py` and budget scripts | `configs/rag/` | `results/rq1/rag/metrics.csv`, `budget_final_v2/budget_metrics.csv` |
| Table 4 | `code/corpus_experiments_v1/sft/run_source_suitability_v1.py` | `configs/rq2/source_suitability_nb2_config.json` | `results/rq2/source_suitability/nb_results.csv` |
| Table 5 | `code/corpus_experiments_v1/sft/run_ipm_final_robustness.py` | `configs/robustness/ipm_final_robustness_config.json` | `results/rq2/final_robustness/rq2_common_vs_specific.csv` |
| Table 6 | C2/C3 scripts | `configs/qwen3b/` | `results/rq3/full_grid/practical_equivalence.csv`, `results/rq3/scale_extension/practical_equivalence.csv` |
| Table 7 | Qwen7B/Phi scripts | `configs/qwen7b/`, `configs/phi/` | `results/rq3/qwen7b/cross_model_scale_comparison.csv`, `results/rq3/phi/cross_model_comparison.csv` |
| Supplementary Table S10 | `run_sft_c4_compute_matched.py` | `configs/qwen3b/compute_matched_config.json` | `results/rq3/compute_matched/` |
| Supplementary Table S13A | RAG bootstrap code | `configs/rag/` | `results/rq1/rag/budget_final_v2/budget_bootstrap.csv` |
| Supplementary Table S14 | RQ2 robustness code | `configs/robustness/` | `results/rq2/final_robustness/rq2_common_vs_specific.csv` |
| Supplementary Table S16 | RQ3 robustness code | `configs/robustness/` | `results/rq3/robustness/rq3_delta_sensitivity.csv` |
| Supplementary Table S17 | reviewer-check script | frozen seeds in outputs | `results/rq1/robustness/` |
| Supplementary Table S18 | reviewer-check script | frozen seeds in outputs | `results/rq2/reviewer2/rq2_length_only_baseline.csv` |
| Supplementary Table S19 | machine-validation code | `configs/sft/` | `results/robustness/t2_numeric_eligibility_funnel.csv` |
| Supplementary Table S20 | `code/statistics/mjsa_statistics.py` | audit seed in `results/mjsa/mjsa_summary.json` | `results/mjsa/` |

## Quick validation

From the repository root:

```bash
python code/common/validate_release.py
python code/statistics/mjsa_statistics.py --bootstrap-seed 20260819
python code/corpus_experiments_v1/plot_ipm_submission_figures.py --output figures/regenerated
```

The MJSA command uses `20260819` as a clearly identified **release reproduction seed**. The original historical MJSA bootstrap RNG seed was not retained; the audit sampling seed itself is formally frozen as `20260819`.

## Licensing

- Code and project-authored configuration/scripts: [MIT License](LICENSE).
- Project-generated derived research data: [CC BY 4.0](LICENSE-DATA.md).
- Third-party resources: their original licenses and terms.

The exact mixed-content scope is defined in [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

## Reproducibility boundaries

Two historical metadata boundaries are disclosed neutrally: the historical MJSA bootstrap RNG seed was not recorded, and the exact historical `microsoft/Phi-4-mini-instruct` snapshot revision could not be recovered. Available frozen outputs and configurations remain included. See [docs/REPRODUCIBILITY_LIMITATIONS.md](docs/REPRODUCIBILITY_LIMITATIONS.md).

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for Level A and Level B instructions.
