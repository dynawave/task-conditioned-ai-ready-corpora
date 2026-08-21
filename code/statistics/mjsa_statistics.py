#!/usr/bin/env python3
"""Reproduce copyright-safe MJSA agreement and consensus statistics.

The release deliberately omits evidence, claims, and free-text reviewer reasons.
The optional bootstrap uses a release-specified RNG seed; the original frozen
MJSA bootstrap seed was not recorded in the retained artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fleiss(labels: pd.DataFrame, field: str) -> tuple[float, float]:
    categories = sorted(labels[field].dropna().unique())
    rows = []
    for _, group in labels.groupby("audit_id", sort=False):
        rows.append([int((group[field] == category).sum()) for category in categories])
    counts = np.asarray(rows, dtype=float)
    n_raters = counts.sum(axis=1)
    observed_per_item = ((counts * counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    observed = float(observed_per_item.mean())
    proportions = counts.sum(axis=0) / counts.sum()
    expected = float((proportions * proportions).sum())
    return observed, float((observed - expected) / (1.0 - expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("results/mjsa/mjsa_judgments_anonymous.csv"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, required=True,
                        help="Release reproduction seed; the formal historical bootstrap seed was not recorded.")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    if labels.shape[0] != 450 or labels.audit_id.nunique() != 150:
        raise RuntimeError("Expected 450 judgments over 150 audit samples")
    consensus = []
    for audit_id, group in labels.groupby("audit_id", sort=False):
        counts = group.overall_judgment.value_counts().to_dict()
        consensus.append({
            "audit_id": audit_id,
            "strict": counts.get("Accept", 0) >= 2 and counts.get("Reject", 0) == 0,
            "relaxed": counts.get("Accept", 0) >= 2,
            "majority_reject": counts.get("Reject", 0) >= 2,
            "conflict": counts.get("Accept", 0) > 0 and counts.get("Reject", 0) > 0,
            "complete": group.overall_judgment.nunique() == 1,
        })
    consensus = pd.DataFrame(consensus)
    pairwise, kappa = fleiss(labels, "overall_judgment")
    dimensions = {}
    for field in ["evidence_support", "factual_correctness", "condition_completeness", "answer_uniqueness"]:
        dimensions[field] = dict(zip(["pairwise_agreement", "fleiss_kappa"], fleiss(labels, field)))

    rng = np.random.default_rng(args.bootstrap_seed)
    values = consensus.strict.to_numpy(dtype=float)
    draws = values[rng.integers(0, len(values), size=(args.bootstrap_replicates, len(values)))].mean(axis=1)
    payload = {
        "samples": int(len(consensus)),
        "judgments": int(len(labels)),
        "strict_majority_accept": int(consensus.strict.sum()),
        "strict_majority_accept_rate": float(consensus.strict.mean()),
        "strict_rate_bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "relaxed_majority_accept": int(consensus.relaxed.sum()),
        "majority_reject": int(consensus.majority_reject.sum()),
        "accept_reject_conflict": int(consensus.conflict.sum()),
        "complete_agreement": int(consensus.complete.sum()),
        "overall_raw_pairwise_agreement": pairwise,
        "overall_fleiss_kappa": kappa,
        "dimension_agreement": dimensions,
        "bootstrap_replicates": args.bootstrap_replicates,
        "release_bootstrap_seed": args.bootstrap_seed,
        "formal_historical_bootstrap_seed": "NOT_RECORDED",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
