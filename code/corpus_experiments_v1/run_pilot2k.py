from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from audit_cpt_ready import audit_parquet
from build_canonical import aggregate_metrics, build_documents
from utils import (
    parse_simple_yaml_model,
    require_absent,
    sha256_file,
    write_checksums,
    write_csv,
    write_json,
)


EXPECTED_E2_PREP_SHA256 = "3b97c102d3d76143cbb75c20a7188f6e6b85273807eb69ca4f00eb2853560172"


def load_manifest(database: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    con = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    pool = dict(con.execute("select * from source_pools where pool_id='S2K'").fetchone())
    members = [
        dict(row) for row in con.execute(
            "select * from source_pool_members where pool_id='S2K' order by ordinal"
        )
    ]
    metadata = {row["key"]: row["value"] for row in con.execute("select * from run_metadata")}
    integrity = con.execute("pragma integrity_check").fetchone()[0]
    con.close()
    metadata["integrity_check"] = integrity
    return pool, members, metadata


def distributions(members: list[dict[str, Any]]) -> dict[str, Any]:
    def dist(key: str) -> dict[str, int]:
        values = Counter("unavailable" if row.get(key) is None else str(row[key]) for row in members)
        return dict(sorted(values.items()))
    return {
        "topic_cluster": dist("topic_cluster"),
        "paragraph_count_bucket": dist("paragraph_count_bucket"),
        "structure_bucket": dist("structure_bucket"),
        "year": dist("year"),
        "document_type": dist("document_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output = args.output or project_root / "aicorpus-derived/experiments/corpus_v1/pilot2k"
    smoke_summary_path = project_root / "aicorpus-derived/experiments/corpus_v1/smoke10/run_04/summary.json"
    if not smoke_summary_path.is_file():
        raise RuntimeError("pilot2k requires a completed smoke10 summary")
    smoke = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    if smoke.get("gate_status") != "PASS":
        raise RuntimeError(f"pilot2k blocked by smoke status: {smoke.get('gate_status')}")

    database = project_root / "aicorpus-derived/e2_prep_v25/e2_prep_v25.sqlite"
    database_sha256 = sha256_file(database)
    if database_sha256 != EXPECTED_E2_PREP_SHA256:
        raise RuntimeError(f"Frozen E2 prep database SHA256 changed: {database_sha256}")
    pool, members, metadata = load_manifest(database)
    paper_ids = [row["paper_id"] for row in members]
    member_hash = hashlib.sha256("\n".join(paper_ids).encode("utf-8")).hexdigest()
    if len(paper_ids) != 2000 or member_hash != pool["paper_id_sha256"]:
        raise RuntimeError(
            f"Frozen S2K mismatch: count={len(paper_ids)}, observed_hash={member_hash}, stored_hash={pool['paper_id_sha256']}"
        )
    if metadata["integrity_check"] != "ok":
        raise RuntimeError(f"Manifest SQLite integrity_check={metadata['integrity_check']}")

    data_root = project_root / "aicorpus-data/mineru_core_31k"
    missing = [
        paper_id for paper_id in paper_ids
        if not (data_root / paper_id / "paragraph_article.json").is_file()
        or not (data_root / paper_id / "paragraph_article.source.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"S2K raw MinerU core assets missing for {len(missing)} papers: {missing[:10]}")

    formal_config_path = project_root / "aicorpus-derived/e2_formal_v25/formal_e2_config.yaml"
    tokenizer_meta = parse_simple_yaml_model(formal_config_path)
    tokenizer_snapshot = Path(tokenizer_meta["snapshot"])
    if not tokenizer_snapshot.is_dir():
        raise RuntimeError(f"Frozen tokenizer snapshot is unavailable: {tokenizer_snapshot}")

    output.mkdir(parents=True, exist_ok=True)
    for filename in ("canonical_documents.parquet", "paper_metrics.csv", "summary.json", "config.json"):
        require_absent(output / filename)
    strata = distributions(members)
    config = {
        "experiment": "corpus_v1_pilot2k",
        "manifest": "aicorpus-derived/e2_prep_v25/e2_prep_v25.sqlite::source_pool_members[pool_id=S2K]",
        "manifest_database_sha256": database_sha256,
        "manifest_integrity_check": metadata["integrity_check"],
        "pool_id": pool["pool_id"],
        "paper_count": pool["paper_count"],
        "seed": pool["seed"],
        "selection_rule": pool["selection_rule"],
        "paper_id_sha256": member_hash,
        "raw_source_root": "aicorpus-data/mineru_core_31k",
        "raw_source_files": ["paragraph_article.source.json", "paragraph_article.json"],
        "content_list_process_availability": 0,
        "parser_source": "paragraph_article_source",
        "source_adapter_validation": "smoke10 content_list_process/source location parity PASS",
        "page_numbering": "zero_based_mineru_page_idx",
        "tokenizer": tokenizer_meta,
        "clean_text_rules": ["Unicode NFKC", "remove Unicode Cc/Cf controls except tab/newline/carriage return before whitespace collapse", "collapse whitespace"],
        "suspected_word_split_action": "flag only; never merge",
        "paragraph_fragment_similarity": "normalized exact cross-document match, minimum 200 characters",
        "paper_near_duplicate_policy": "reuse frozen Gate0 signals; do not create a new semantic threshold",
    }
    write_json(output / "config.json", config)

    member_by_id = {row["paper_id"]: row for row in members}
    specs = [
        {"paper_dir": data_root / paper_id, "source_id": paper_id, "version_id": None}
        for paper_id in paper_ids
    ]
    metrics, failures = build_documents(
        specs, project_root, output / "canonical_documents.parquet", force_parser="paragraph_article_source"
    )
    canonical = aggregate_metrics(metrics)
    cpt, token_metrics = audit_parquet(
        output / "canonical_documents.parquet",
        tokenizer_snapshot,
        tokenizer_meta,
        project_root,
        database,
        strata,
    )
    for row in metrics:
        row.update(token_metrics.get(row["paper_id"], {}))
        member = member_by_id[row["paper_id"]]
        row.update({
            "topic_cluster": member["topic_cluster"],
            "paragraph_count_bucket": member["paragraph_count_bucket"],
            "structure_bucket": member["structure_bucket"],
            "year": member["year"],
            "document_type": member["document_type"],
            "version_group": member["version_group"],
        })
    fields = list(metrics[0]) if metrics else ["paper_id"]
    write_csv(output / "paper_metrics.csv", metrics, fields)
    if failures:
        write_csv(output / "failures.csv", failures, ["paper_id", "exception_type", "message"])

    blockers: list[str] = []
    if failures or len(metrics) != 2000:
        blockers.append(f"Canonical succeeded for {len(metrics)}/2000 papers; failures={len(failures)}")
    for name in ("paragraph_page_coverage", "paragraph_bbox_coverage", "provenance_coverage"):
        if canonical.get(name) != 1.0:
            blockers.append(f"{name}={canonical.get(name)!r}")
    if canonical.get("formula_latex_coverage") != 1.0:
        blockers.append(f"formula_latex_coverage={canonical.get('formula_latex_coverage')!r}")
    if cpt["paper_duplicate_context"]["frozen_gate0_semantic_threshold"] is None:
        limitations = [
            "Frozen Gate0 defines semantic nearest-neighbour signals but no threshold; no semantic near-duplicate count is asserted.",
            "S2K core assets omit content_list_process.json; the paragraph_article.source adapter is used after smoke parity validation.",
            "Year and document type are unavailable in the frozen manifest and remain unavailable rather than inferred.",
            f"Table HTML coverage is {canonical.get('table_html_coverage')}; missing source HTML remains null.",
        ]
    else:
        limitations = [
            "S2K core assets omit content_list_process.json; the paragraph_article.source adapter is used after smoke parity validation.",
            "Year and document type are unavailable in the frozen manifest and remain unavailable rather than inferred.",
            f"Table HTML coverage is {canonical.get('table_html_coverage')}; missing source HTML remains null.",
        ]
    readiness = "READY_FOR_RAG" if not blockers else "NOT_READY_FOR_RAG"
    summary = {
        "experiment": "corpus_v1_pilot2k",
        "manifest": {
            "path": config["manifest"],
            "paper_count": len(paper_ids),
            "seed": pool["seed"],
            "paper_id_sha256": member_hash,
            "database_sha256": database_sha256,
        },
        "canonical_success_count": len(metrics),
        "canonical_failure_count": len(failures),
        "canonical_success_rate": len(metrics) / len(paper_ids),
        "canonical": canonical,
        "cpt_ready": cpt,
        "blockers": blockers[:5],
        "limitations": limitations,
        "readiness": readiness,
    }
    write_json(output / "summary.json", summary)
    names = ["canonical_documents.parquet", "paper_metrics.csv", "summary.json", "config.json"]
    if failures:
        names.append("failures.csv")
    checksums = write_checksums(output, names)
    print(json.dumps({
        "readiness": readiness,
        "canonical_success_count": len(metrics),
        "body_paragraph_count": canonical["body_paragraph_count"],
        "body_clean_tokens": cpt["body_clean_tokens"],
        "blockers": blockers[:5],
        "checksums": checksums,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
