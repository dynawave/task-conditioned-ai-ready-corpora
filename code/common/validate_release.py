#!/usr/bin/env python3
"""Read-only numeric and package-safety validation for the public release."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def close(actual: float, expected: float, tol: float = 5e-7) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"numeric drift: {actual} != {expected}")


def main() -> None:
    asset = load_json("results/corpus/asset_readiness_summary_31k_v2.json")
    s2k = load_json("results/corpus/s2k_summary.json")
    assert asset["corpus"]["gate0_technical_eligible"] == 30_883
    assert s2k["canonical"]["papers"] == 2_000
    assert s2k["canonical"]["body_paragraph_count"] == 114_511
    assert s2k["cpt_ready"]["body_clean_tokens"] == 21_013_210

    rag = load_json("results/rq1/rag/summary.json")
    gold = rag["public_gold_mapping"]
    assert (gold["formal_question_count"], gold["mapped_gold_evidence_strings"], gold["total_gold_evidence_strings"]) == (1292, 5420, 5648)
    b1 = next(row for row in rag["metrics"] if row["view"] == "B1")
    close(b1["ces@5"], 0.6888544891640866)
    budget = pd.read_csv(ROOT / "results/rq1/rag/budget_final_v2/budget_metrics.csv")
    close(budget.loc[budget.view.eq("B2"), "BudgetCES@1024"].iloc[0], 0.6501547987616099)

    rq2 = pd.read_csv(ROOT / "results/rq2/final_robustness/rq2_common_vs_specific.csv")
    expected = {
        ("T1a_ACRONYM", "paper_count"): 0.27408282276518786,
        ("T2_NUMERIC", "paper_count"): 0.33273922782340615,
        ("T3_CLAIM", "paper_count"): 0.3469928993070408,
        ("T1a_ACRONYM", "source_token"): 0.2402745995423341,
        ("T2_NUMERIC", "source_token"): 0.21782798236951756,
        ("T3_CLAIM", "source_token"): 0.4164599195825135,
    }
    for (task, budget_type), value in expected.items():
        row = rq2[(rq2.task == task) & (rq2.budget_type == budget_type) & (rq2.strategy == "task_specific")].iloc[0]
        close(row.capture_fraction, value)

    c2 = pd.read_csv(ROOT / "results/rq3/full_grid/n_star_summary.csv")
    c3 = pd.read_csv(ROOT / "results/rq3/scale_extension/n_star_summary.csv")
    assert int(c2.loc[c2.task.eq("T1a_ACRONYM"), "n_star"].iloc[0]) == 1000
    assert int(c3.loc[c3.task.eq("T2_NUMERIC"), "n_star"].iloc[0]) == 3000
    assert c3.loc[c3.task.eq("T3_CLAIM"), "n_star_status"].iloc[0] == "REFERENCE_ONLY_NOT_IDENTIFIED"

    q7 = pd.read_csv(ROOT / "results/rq3/qwen7b/qwen7b_key_interval_bootstrap.csv")
    q7_expected = {"T1a_ACRONYM": 0.0100, "T2_NUMERIC": 0.0011111111111111, "T3_CLAIM": 0.03640500568828212}
    for task, value in q7_expected.items(): close(q7.loc[q7.task.eq(task), "point_difference"].iloc[0], value)
    phi = pd.read_csv(ROOT / "results/rq3/phi/bootstrap.csv")
    phi_expected = {"T1a_ACRONYM": 0.0011111111111111, "T2_NUMERIC": 0.0033333333333333, "T3_CLAIM": -0.0056882821387941}
    for task, value in phi_expected.items(): close(phi.loc[phi.task.eq(task), "point_difference"].iloc[0], value)

    mjsa = load_json("results/mjsa/mjsa_summary.json")
    assert (mjsa["strict_majority_accept"], mjsa["complete_agreement"]) == (135, 109)
    close(mjsa["overall_raw_pairwise_agreement"], 0.8155555555555556)
    close(mjsa["fleiss_kappa"], 0.22763555151163994)

    forbidden_extensions = {".docx", ".xlsx", ".xls", ".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".key", ".pem"}
    bad = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden_extensions]
    if bad: raise AssertionError(f"forbidden extensions: {bad}")
    text_ext = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".csv", ".txt", ".cff", ".gitignore", ".sha256"}
    patterns = [re.compile(x, re.I) for x in [
        r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:CAS|Users|Documents|Data)", "C:" + r"\\Users\\", "OPENAI" + "_API_KEY",
        "HF" + "_TOKEN", "WANDB" + "_API_KEY", "Author" + r"ization:\\s*" + "Bear" + "er",
    ]]
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_ext or ".audit_work" in path.parts: continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(p.search(text) for p in patterns): hits.append(path)
    if hits: raise AssertionError(f"private path or secret patterns: {hits}")
    print("PAPER_RESULT_CROSSCHECK: PASS")
    print("RELEASE_SAFETY_CHECK: PASS")


if __name__ == "__main__":
    main()
