# Third-party data and models

No raw third-party dataset or model weight is redistributed in this release.

## QASPER

- Official source: <https://allenai.org/data/qasper>
- Version/split used: v0.3, test.
- Role: formal public-gold RAG evidence-retrieval evaluation; also reused public-gold calibration for a restricted SFT numeric-verifier diagnostic.
- Redistribution: raw questions, paper text, evidence strings, and annotations are not redistributed. Only anonymous membership/counts and aggregate results are included.
- License: Please consult the original resource license and terms of use.

## SciAI — Scientific Acronym Identification

- Official source used: <https://github.com/amirveyseh/AAAI-21-SDU-shared-task-1-AI>
- Frozen source revision: `9e810512b98e1cf034e757d92136c7e64f840d0f`.
- Role: external verifier validation for T1a acronym/long-form identification; not an SFT training or scaling evaluation set.
- Local frozen record: dataset CC BY-NC-SA 4.0; scorer/baseline MIT. Users must verify the upstream terms before reuse.
- Redistribution: not redistributed.

## SciFact

- Official source: <https://github.com/allenai/scifact>
- Splits used: train for entailment-threshold calibration; dev for held-out verifier validation; public test had no gold evidence labels and was not used for scored reporting.
- Role: validation of the NLI component, not primary geoscience SFT evaluation.
- Redistribution: not redistributed.
- License: Please consult the original resource license and terms of use.

## SciTaT

- Official source: <https://github.com/zhxlia/SciTaT>
- Split used: official test, restricted lookup-only diagnostic subset.
- Role: secondary numeric answer/evidence recoverability diagnostic; not primary SFT evaluation.
- Redistribution: not redistributed.
- License: Please consult the original resource license and terms of use.

## Base and encoder models

Model weights are obtained separately from their official repositories and are not redistributed:

- `BAAI/bge-base-en-v1.5`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/Phi-4-mini-instruct`
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

The exact revisions available in frozen records are retained in the sanitized configuration files. Consult each upstream model card and license.

## Adaptive Chunking feasibility method

- Publication: *Adaptive Chunking: Optimizing Chunking-Method Selection for RAG*, LREC 2026; preprint `arXiv:2603.25333v1`.
- Official source: <https://github.com/ekimetrics/adaptive-chunking>.
- Frozen source revision used for the feasibility assessment: `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d` (package version `0.1.0`).
- Scope: only the official deterministic `our_recurs_600` and `our_recurs_1100` candidate generators were assessed; the complete multi-candidate adaptive selector was not run.
- Redistribution: no upstream source code or third-party paper text is redistributed. Obtain the implementation from its official repository and follow its license and installation instructions.
