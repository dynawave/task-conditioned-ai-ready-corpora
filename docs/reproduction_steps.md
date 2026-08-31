# Reproduction steps

## 1. Verify the release

```bash
python code/common/validate_release.py
```

Expected terminal lines:

```text
PAPER_RESULT_CROSSCHECK: PASS
EDITORIAL_REVISION_CROSSCHECK: PASS
RELEASE_SAFETY_CHECK: PASS
RELEASE_MANIFEST_CHECK: PASS
```

## 2. Reproduce MJSA agreement

```bash
python code/statistics/mjsa_statistics.py --bootstrap-seed 20260819
```

The seed is a public-release reproduction seed. The source workbooks record the sample seed (20260819) but not the historical bootstrap RNG seed.

## 3. Recreate Figures 2–5 and Supplementary Figure S1

```bash
python code/corpus_experiments_v1/plot_ipm_submission_figures.py --output figures/regenerated
```

The output directory must not already exist. The script verifies 14 frozen input checksums. Compare regenerated SVG/PDF/PNG values and labels with `figures/final/`. Minor binary hash differences may occur across Matplotlib/font backends; the data checks and plotted values must match.

## 4. Figure 1

Open `figures/final/Figure1_framework_final.drawio` in Draw.io and export to SVG/PDF/PNG. Figure 1 is conceptual and has no empirical input table.

## 5. Full experimental reruns

See [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md). Full document and model reruns require externally obtained licensed data and models and are not necessary to verify the released paper numbers.

The same Level A validation command directly verifies the v1.0.1 No-SFT, Adaptive-feasibility, and structure/title-sensitivity aggregate values. Their full scripts are retained for auditability, but rerunning them requires the corresponding legally obtained frozen inputs and is not needed for result verification.
