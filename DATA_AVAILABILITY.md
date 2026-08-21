# Data availability

## Included

- Analysis code and experiment configurations.
- Anonymized S2K corpus metrics, Train1500/Eval400 split membership, and nested SFT membership information.
- Project-generated derived metrics, numeric RQ2 features, and OOF prediction scores.
- Frozen aggregate, per-seed, bootstrap, and robustness results supporting the paper.
- Frozen figure inputs, author-generated final figures, and figure-generation code.
- Copyright-safe MJSA labels, consensus indicators, and aggregate agreement metrics.
- Reproducibility documentation and SHA-256 provenance manifests.

## Excluded

- The 30,883-paper source collection, copyrighted source PDFs/full text, paragraphs, abstracts, titles, DOI metadata, publisher XML/HTML, figures, and tables.
- Raw QASPER, SciAI, SciFact, SciTaT, or other third-party datasets.
- Model weights, LoRA adapters, embeddings, checkpoints, and model/tokenizer caches.
- Private credentials, API tokens, private caches, and logs.
- Copyright-sensitive `Evidence_Context`, `Conditional_Claim`, reviewer free text, and generated training prompts/targets that may reproduce source wording.
- Debug and smoke artifacts.

## Obtaining dependencies

Third-party public datasets must be obtained from their original providers. Official sources and their roles in this study are listed in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). Full document-engineering and model-training replication additionally requires a legally obtained source corpus; this release does not provide or grant access to that corpus.
