from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "canonical-scientific-document-v1.0.0"
PARSER_VERSION = "mineru-canonical-parser-v1.0.0"


PAPER_FIELDS = ("paper_id", "source_id", "version_id", "title", "abstract", "source_path")
SECTION_FIELDS = ("section_id", "section_path", "major_section")
PARAGRAPH_FIELDS = ("paragraph_id", "raw_text", "clean_text", "page", "bbox", "section_id")
OBJECT_FIELDS = (
    "object_id", "object_type", "object_subtype", "page", "bbox", "caption", "table_html", "formula_latex"
)
PROVENANCE_FIELDS = ("source_file", "source_page", "source_bbox", "parser_source")


def validate_document(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in PAPER_FIELDS:
        if key not in document:
            errors.append(f"missing paper field: {key}")
    if not document.get("paper_id") or not document.get("source_id"):
        errors.append("paper_id/source_id must be non-empty")

    for collection, fields, id_field in (
        ("sections", SECTION_FIELDS, "section_id"),
        ("paragraphs", PARAGRAPH_FIELDS, "paragraph_id"),
        ("objects", OBJECT_FIELDS, "object_id"),
    ):
        rows = document.get(collection)
        if not isinstance(rows, list):
            errors.append(f"{collection} must be a list")
            continue
        ids: list[str] = []
        for index, row in enumerate(rows):
            for key in fields:
                if key not in row:
                    errors.append(f"{collection}[{index}] missing {key}")
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                errors.append(f"{collection}[{index}] missing provenance")
            else:
                for key in PROVENANCE_FIELDS:
                    if key not in provenance:
                        errors.append(f"{collection}[{index}] provenance missing {key}")
            if row.get(id_field):
                ids.append(row[id_field])
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate {id_field}")
    return errors


def arrow_schema():
    import pyarrow as pa

    bbox = pa.list_(pa.float64())
    provenance = pa.struct([
        pa.field("source_file", pa.string()),
        pa.field("source_page", pa.int32()),
        pa.field("source_bbox", bbox),
        pa.field("parser_source", pa.string()),
        pa.field("source_index", pa.int32()),
    ])
    section = pa.struct([
        pa.field("section_id", pa.string()),
        pa.field("section_path", pa.list_(pa.string())),
        pa.field("major_section", pa.string()),
        pa.field("heading", pa.string()),
        pa.field("level", pa.int32()),
        pa.field("page", pa.int32()),
        pa.field("bbox", bbox),
        pa.field("provenance", provenance),
    ])
    paragraph = pa.struct([
        pa.field("paragraph_id", pa.string()),
        pa.field("raw_text", pa.string()),
        pa.field("clean_text", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("bbox", bbox),
        pa.field("section_id", pa.string()),
        pa.field("suspected_word_split", pa.bool_()),
        pa.field("provenance", provenance),
    ])
    obj = pa.struct([
        pa.field("object_id", pa.string()),
        pa.field("object_type", pa.string()),
        pa.field("object_subtype", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("bbox", bbox),
        pa.field("caption", pa.string()),
        pa.field("table_html", pa.string()),
        pa.field("formula_latex", pa.string()),
        pa.field("image_path", pa.string()),
        pa.field("object_text", pa.string()),
        pa.field("provenance", provenance),
    ])
    uncertain = pa.struct([
        pa.field("element_id", pa.string()),
        pa.field("raw_type", pa.string()),
        pa.field("raw_text", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("bbox", bbox),
        pa.field("reason", pa.string()),
        pa.field("provenance", provenance),
    ])
    return pa.schema([
        pa.field("paper_id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("version_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("source_path", pa.string()),
        pa.field("schema_version", pa.string()),
        pa.field("parser_version", pa.string()),
        pa.field("parser_source", pa.string()),
        pa.field("page_numbering", pa.string()),
        pa.field("sections", pa.list_(section)),
        pa.field("paragraphs", pa.list_(paragraph)),
        pa.field("objects", pa.list_(obj)),
        pa.field("uncertain_elements", pa.list_(uncertain)),
        pa.field("excluded_counts_json", pa.string()),
    ])
