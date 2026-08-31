# Reproducibility

## Level A — frozen results, statistics, and figures (no GPU)

Create a Python environment and run from the repository root. For CPU-only Level A, install NumPy, pandas, SciPy, statsmodels, and Matplotlib; the pinned CUDA PyTorch build in `requirements-release.txt` is needed only for Level B and may require the official PyTorch CUDA 12.8 package index.

```bash
python code/common/validate_release.py
python code/statistics/mjsa_statistics.py --bootstrap-seed 20260819
python code/corpus_experiments_v1/plot_ipm_submission_figures.py --output figures/regenerated
```

The plotting script verifies SHA-256 checksums for its 14 formal inputs before creating Figure 2–5 and Supplementary Figure S1. Figure 1 is an editable Draw.io conceptual framework and is reproduced from `figures/final/Figure1_framework_final.drawio`, not from experimental data.

The validation script checks the headline corpus, RQ1, RQ2, RQ3, model-replication, and MJSA values and rejects forbidden binary extensions or private-path/secret patterns. It does not retrain a model or overwrite frozen results.

For v1.0.1 it also directly checks the frozen values in:

- `results/rq3/no_sft/no_sft_control.json` (Sections 5.6/6.4, Supplementary S9.1, Table S12A);
- `results/rq1/sensitivity/adaptive_chunking_feasibility.json` (Supplementary S15.1);
- `results/rq1/sensitivity/structure_title_augmented_sensitivity.json` (Supplementary S15.2).

No-SFT is an independent control condition, not an `n=0` SFT learning-curve point. The RQ1 additions are development-only feasibility/sensitivity checks and do not replace the frozen B1/B2/B3 comparison.

## Level B — model experiments (requires external data and GPU)

Level B requires legally obtained source documents, official third-party resources, and the official base models. The public package provides configurations, scripts, anonymous memberships, and hashes but not copyrighted text or weights.

Set the repository `code/` directory on `PYTHONPATH` before invoking package-style experiment modules (for example, `PYTHONPATH=code` on POSIX or `$env:PYTHONPATH="code"` in PowerShell). Release scripts resolve project assets relative to the repository root; model locations can be supplied through the environment variables listed below.

### Qwen2.5-3B-Instruct main experiment

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- FP16, no quantization; LoRA `r=16`, `alpha=32`, dropout `0.05`, targets `q_proj/k_proj/v_proj/o_proj`.
- 3 epochs, AdamW, learning rate `2e-4`, linear warmup ratio `0.03`.
- Micro-batch 2, gradient accumulation 4, effective batch 8.
- Maximum sequence length 1280; assistant-only loss.
- Training seeds 42, 314159, 271828.

### Qwen2.5-7B-Instruct key-interval replication

- Model revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Same scientific recipe, scales, optimizer, precision, LoRA settings, effective batch, and seeds as the specified 3B comparison.
- Necessary memory implementation: micro-batch 1, gradient accumulation 8, gradient checkpointing enabled; effective batch remains 8; SDPA attention.
- This is a prespecified key-interval replication, not a new full learning curve or a new estimate of `n*`.

### Phi-4-mini-instruct directional replication

- Model: `microsoft/Phi-4-mini-instruct`; the exact revision was not recoverable from the frozen local snapshot and is explicitly recorded as such.
- FP16, no quantization; LoRA `r=16`, `alpha=32`, dropout `0.05`, architecture-mapped targets `qkv_proj/o_proj`.
- 3 epochs, AdamW, learning rate `2e-4`, warmup ratio `0.03`, effective batch 8, maximum sequence length 1280, assistant-only loss.
- Seeds 42, 314159, 271828.

All released training runners use `local_files_only=True`; they do not automatically resolve or download model files from Hugging Face. Obtain the specified official model revision, store that exact snapshot locally, and provide its path through `QWEN3B_MODEL_PATH`, `QWEN7B_MODEL_PATH`, or `PHI4MINI_MODEL_PATH` as applicable. See the corresponding sanitized protocol files for the retained model revisions and parameters.

### Editorial-revision analysis reruns

The actual runners are released as `run_sft_c0_no_sft_control.py`, `run_sft_c0_no_sft_vs_n100_bootstrap.py`, `run_adaptive_chunking_feasibility.py`, and `run_hierarchical_structure_feasibility.py`. Full reruns require the legally obtained, hash-matching frozen QASPER/RQ3 evaluation assets described above; text-bearing predictions and benchmark content are not redistributed. The Adaptive runner additionally requires the official upstream repository at commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d` and assesses only its deterministic recursive candidates, not the complete adaptive selector.
