# Reproducibility boundaries

## Historical MJSA bootstrap random seed

The T3 Multi-Judge Semantic Audit used 10,000 bootstrap resamples for the reported confidence intervals. The random seed used for this historical bootstrap run was not recorded. The frozen bootstrap outputs used in the manuscript are included in this release. The released implementation supports deterministic reruns with a user-specified random seed, but a rerun is not expected to reproduce the historical confidence-interval endpoints bit-for-bit.

No historical seed has been inferred or substituted, and the manuscript's frozen results have not been recomputed or overwritten.

## Phi model snapshot revision

The cross-family directional check used `microsoft/Phi-4-mini-instruct`. The exact Hugging Face snapshot/commit revision used in the historical runs was not recorded and could not be recovered from the available local artifacts. The model identifier, available training configuration, frozen per-seed metrics, and aggregate evaluation outputs are provided. Consequently, exact from-scratch reproduction of the historical Phi runs at the model-file level cannot be guaranteed.

No commit SHA has been inferred, and no currently downloadable model snapshot is presented as the historical snapshot.
