from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from canonical_schema import arrow_schema
from parse_mineru import parse_document


def document_metrics(document: dict[str, Any]) -> dict[str, Any]:
    paragraphs = document["paragraphs"]
    objects = document["objects"]
    sections = document["sections"]
    figures = [row for row in objects if row["object_type"] == "figure"]
    tables = [row for row in objects if row["object_type"] == "table"]
    formulas = [row for row in objects if row["object_type"] == "formula"]
    elements = paragraphs + objects

    def covered(rows: list[dict[str, Any]], field: str) -> int:
        return sum(row.get(field) is not None for row in rows)

    def provenance_covered(row: dict[str, Any]) -> bool:
        provenance = row.get("provenance") or {}
        return bool(
            provenance.get("source_file")
            and provenance.get("parser_source")
            and provenance.get("source_page") is not None
            and provenance.get("source_bbox") is not None
        )

    assigned = sum(row.get("section_id") is not None for row in paragraphs)
    return {
        "paper_id": document["paper_id"],
        "source_id": document["source_id"],
        "parser_source": document["parser_source"],
        "title_available": int(bool(document.get("title"))),
        "abstract_available": int(bool(document.get("abstract"))),
        "section_count": len(sections),
        "body_paragraph_count": len(paragraphs),
        "body_raw_character_count": sum(len(row["raw_text"]) for row in paragraphs),
        "body_clean_character_count": sum(len(row["clean_text"]) for row in paragraphs),
        "paragraphs_with_section": assigned,
        "paragraph_section_coverage": assigned / len(paragraphs) if paragraphs else None,
        "suspected_word_split_paragraphs": sum(bool(row["suspected_word_split"]) for row in paragraphs),
        "figure_count": len(figures),
        "table_count": len(tables),
        "formula_count": len(formulas),
        "other_object_count": len(objects) - len(figures) - len(tables) - len(formulas),
        "figure_caption_count": sum(bool(row.get("caption")) for row in figures),
        "table_caption_count": sum(bool(row.get("caption")) for row in tables),
        "table_html_count": sum(bool(row.get("table_html")) for row in tables),
        "formula_latex_count": sum(bool(row.get("formula_latex")) for row in formulas),
        "paragraph_page_count": covered(paragraphs, "page"),
        "paragraph_bbox_count": covered(paragraphs, "bbox"),
        "object_page_count": covered(objects, "page"),
        "object_bbox_count": covered(objects, "bbox"),
        "provenance_count": sum(provenance_covered(row) for row in elements),
        "canonical_element_count": len(elements),
        "uncertain_element_count": len(document["uncertain_elements"]),
        "excluded_counts_json": document["excluded_counts_json"],
    }


def build_documents(
    specs: Iterable[dict[str, Any]],
    project_root: Path,
    output_path: Path,
    force_parser: str | None = None,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    schema = arrow_schema()
    writer = pq.ParquetWriter(partial, schema, compression="zstd", use_dictionary=True)
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    batch: list[dict[str, Any]] = []
    try:
        for spec in specs:
            paper_dir = Path(spec["paper_dir"])
            try:
                document = parse_document(
                    paper_dir,
                    project_root=project_root,
                    source_id=spec.get("source_id"),
                    version_id=spec.get("version_id"),
                    force_parser=force_parser,
                )
                metrics.append(document_metrics(document))
                batch.append(document)
                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            except Exception as exc:  # retain the true failure and continue the formal set
                failures.append({
                    "paper_id": paper_dir.name,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                })
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        writer.close()
    os.replace(partial, output_path)
    return metrics, failures


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    excluded = Counter()
    for row in rows:
        for key in (
            "section_count", "body_paragraph_count", "body_raw_character_count", "body_clean_character_count",
            "paragraphs_with_section", "suspected_word_split_paragraphs", "figure_count", "table_count",
            "formula_count", "other_object_count", "figure_caption_count", "table_caption_count",
            "table_html_count", "formula_latex_count", "paragraph_page_count", "paragraph_bbox_count",
            "object_page_count", "object_bbox_count", "provenance_count", "canonical_element_count",
            "uncertain_element_count",
        ):
            totals[key] += int(row.get(key) or 0)
        excluded.update(json.loads(row["excluded_counts_json"]))

    def ratio(numerator: str, denominator: str) -> float | None:
        return totals[numerator] / totals[denominator] if totals[denominator] else None

    object_count = totals["figure_count"] + totals["table_count"] + totals["formula_count"] + totals["other_object_count"]
    return {
        "papers": len(rows),
        **dict(totals),
        "paragraph_section_coverage": ratio("paragraphs_with_section", "body_paragraph_count"),
        "paragraph_page_coverage": ratio("paragraph_page_count", "body_paragraph_count"),
        "paragraph_bbox_coverage": ratio("paragraph_bbox_count", "body_paragraph_count"),
        "object_page_coverage": totals["object_page_count"] / object_count if object_count else None,
        "object_bbox_coverage": totals["object_bbox_count"] / object_count if object_count else None,
        "provenance_coverage": ratio("provenance_count", "canonical_element_count"),
        "figure_caption_coverage": ratio("figure_caption_count", "figure_count"),
        "table_caption_coverage": ratio("table_caption_count", "table_count"),
        "table_html_coverage": ratio("table_html_count", "table_count"),
        "formula_latex_coverage": ratio("formula_latex_count", "formula_count"),
        "excluded_counts": dict(sorted(excluded.items())),
    }
