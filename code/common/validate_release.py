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

    no_sft = load_json("results/rq3/no_sft/no_sft_control.json")
    no_sft_expected = {
        "T1a_ACRONYM": (300, 0.0, 0.86, 0.86, 0.8366666666666666, 0.8833333333333333),
        "T2_NUMERIC": (300, 0.0, 0.8433333333333333, 0.8433333333333333, 0.8188888888888889, 0.8677777777777779),
        "T3_CLAIM": (293, 0.0, 0.2946530147895336, 0.2946530147895336, 0.2605233219567691, 0.32992036405005687),
    }
    assert no_sft["bootstrap"]["replicates"] == 10_000
    for row in no_sft["results"]:
        eval_n, base, n100, delta, low, high = no_sft_expected[row["task"]]
        assert row["eval_n"] == eval_n
        close(sum(row["n100_sft_seed_NEM"]) / len(row["n100_sft_seed_NEM"]), n100)
        for key, value in (
            ("no_sft_NEM", base), ("n100_sft_mean_NEM", n100),
            ("delta_N100_SFT_minus_No_SFT", delta), ("ci95_low", low), ("ci95_high", high),
        ):
            close(row[key], value)

    adaptive = load_json("results/rq1/sensitivity/adaptive_chunking_feasibility.json")
    assert adaptive["selection"]["papers_tested"] == adaptive["selection"]["papers_successful"] == 10
    assert adaptive["selection"]["papers_failed"] == 0
    assert adaptive["evidence_mapping"] == {
        "gold_evidence_items": 157, "fully_mapped": 132, "multi_chunk": 0, "unmapped": 25,
    }
    adaptive_candidates = {row["candidate"]: row for row in adaptive["candidate_diagnostics"]}
    assert (adaptive_candidates["our_recurs_600"]["chunks"], adaptive_candidates["our_recurs_600"]["chunks_over_BGE_512_token_limit"]) == (125, 85)
    assert (adaptive_candidates["our_recurs_1100"]["chunks"], adaptive_candidates["our_recurs_1100"]["chunks_over_BGE_512_token_limit"]) == (73, 64)

    structure = load_json("results/rq1/sensitivity/structure_title_augmented_sensitivity.json")
    assert (structure["sampling"]["papers"], structure["sampling"]["formal_questions"], structure["sampling"]["seed"]) == (20, 65, 20260823)
    assert structure["coverage"]["fully_covered_questions"] == 63
    close(structure["coverage"]["fully_covered_rate"], 63 / 65)
    close(structure["coverage"]["mean_best_gold_recall_ceiling"], 0.9846153846153847)
    assert (structure["representations"]["paragraph_units"], structure["representations"]["structure_title_augmented_units"]) == (827, 827)
    tokens = structure["token_statistics"]
    assert (tokens["paragraph_total_tokens"], tokens["structure_title_augmented_total_tokens"]) == (89669, 102428)
    close(tokens["paragraph_mean_tokens"], 108.42684401451028)
    close(tokens["structure_title_augmented_mean_tokens"], 123.85489721886336)
    close(tokens["token_increase_rate"], 0.14228997758422635)
    assert tokens["paragraph_units_over_512"] == tokens["structure_title_augmented_units_over_512"] == 0
    retrieval = structure["retrieval_diagnostics"]
    close(retrieval["paragraph_CES_at_5"], 0.6)
    close(retrieval["structure_title_augmented_CES_at_5"], 0.6)
    close(retrieval["paragraph_BudgetCES_at_1024"], 0.7230769230769231)
    close(retrieval["structure_title_augmented_BudgetCES_at_1024"], 0.6307692307692307)

    forbidden_extensions = {".docx", ".xlsx", ".xls", ".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".key", ".pem"}
    bad = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden_extensions]
    if bad: raise AssertionError(f"forbidden extensions: {bad}")
    text_ext = {"", ".py", ".json", ".yaml", ".yml", ".toml", ".md", ".csv", ".txt", ".cff", ".gitignore", ".sha256"}
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

    manifest_path = ROOT / "MANIFEST_SHA256.txt"
    entries = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries:
            raise AssertionError(f"duplicate manifest entry: {relative}")
        entries[relative] = digest
    intended = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path.name != manifest_path.name
        and "__pycache__" not in path.relative_to(ROOT).parts
    )
    if set(entries) != set(intended):
        raise AssertionError("release manifest coverage mismatch")
    for relative in intended:
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != entries[relative]:
            raise AssertionError(f"release manifest hash mismatch: {relative}")
    print("PAPER_RESULT_CROSSCHECK: PASS")
    print("EDITORIAL_REVISION_CROSSCHECK: PASS")
    print("RELEASE_SAFETY_CHECK: PASS")
    print("RELEASE_MANIFEST_CHECK: PASS")


if __name__ == "__main__":
    main()
