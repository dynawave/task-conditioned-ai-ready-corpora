# Released schemas

The release contains no source-document text. All identifiers are pseudonymous or release-hashed.

- `manifests/corpus/s2k_paper_metrics_anonymous.csv`: one S2K paper per row; structural counts, token counts, provenance coverage, and quality indicators.
- `data/derived/rq2_source_features_anonymous.csv`: one paper-task row; numeric source features, machine-verified task yields, offsets, and standardized predictors.
- `data/derived/rq2_oof_predictions_anonymous.csv`: one paper-task row; frozen five-fold OOF count/rate predictions and fold metadata.
- `manifests/splits/*_anonymous.csv`: paper-disjoint training/evaluation source membership; original source identifiers are omitted.
- `manifests/rq3/nested_pool_manifest.csv`: task, scale, and opaque SFT instance ID.
- `manifests/rq3/extension_pool_manifest_anonymous.csv`: task/scale membership with opaque instance/candidate IDs and release-hashed paper IDs.
- `results/mjsa/mjsa_judgments_anonymous.csv`: 450 structured judgments; no evidence, claim, or free-text reason.
- `results/mjsa/mjsa_consensus_anonymous.csv`: 150 audit memberships and deterministic consensus flags.

Detailed column descriptions are in [data_dictionary.md](data_dictionary.md).
