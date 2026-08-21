# Recorded environments

Versions below come from frozen experiment configurations, summaries, or formal run records. Unrecorded versions are not inferred.

## RAG formal run

- Python 3.10.20
- PyTorch 2.7.1+cu128
- Transformers 4.51.3
- NumPy 2.2.6
- PyArrow 25.0.0
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 25,650,855,936 bytes reported
- Precision: FP16

## RQ2 and frozen-result robustness analysis

- Python 3.10.20
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.3
- statsmodels 0.14.6
- ReportLab 4.4.9

## Qwen2.5-7B formal runs

- OS: Windows-10-10.0.26200-SP0
- Python 3.10.20
- PyTorch 2.7.1+cu128
- CUDA 12.8
- cuDNN 90701
- Transformers 4.51.3
- PEFT 0.15.2
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 25,650,855,936 bytes reported

The Qwen2.5-3B and Phi formal run artifacts record the same core Python/PyTorch/Transformers/PEFT stack used by the training runner. Exact frozen versions of Matplotlib, scikit-learn, sentence-transformers, TRL, Accelerate, and bitsandbytes were not recorded in the retained formal artifacts; these are therefore not assigned invented versions. The formal training protocols used no 4-bit/8-bit quantization and did not require bitsandbytes.
