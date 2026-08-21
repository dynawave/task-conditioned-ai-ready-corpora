from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from canonical_schema import PARSER_VERSION, SCHEMA_VERSION, validate_document
from utils import canonical_json, load_json, relative_posix


EXCLUDED_TYPES = {
    "header": "header",
    "page_header": "header",
    "footer": "footer",
    "page_footer": "footer",
    "page_number": "page_number",
    "page_footnote": "page_footnote",
    "footnote": "page_footnote",
    "ref_text": "reference",
    "reference": "reference",
}
REFERENCE_HEADINGS = {"references", "reference", "bibliography", "literaturecited", "workscited"}
ABSTRACT_HEADINGS = {"abstract", "summary", "keywords", "keyword", "highlights", "articleinfo", "articleinformation"}
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"\s+")
SPLIT_RE = re.compile(
    r"\b(?:[A-Za-z]{3,}-\s+[a-z]{2,}|[A-Za-z]{2,}\s+(?:tion|tions|ment|ments|ness|ously|ively|ically|ations|ability|ities|jection|centrations|cant|ances|diate|izontal|istry|ficult))\b"
)


def clean_text(raw_text: str) -> str:
    value = unicodedata.normalize("NFKC", raw_text)
    value = "".join(
        char for char in value
        if char in "\t\n\r" or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    return SPACE_RE.sub(" ", value).strip()


def suspected_word_split(raw_text: str) -> bool:
    return bool(SPLIT_RE.search(raw_text))


def _norm_label(value: str | None) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value or "").casefold()).strip(" .:;")


def _heading_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm_label(value))


def _semantic_values(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("value"), str):
                parts.append(item["value"])
        joined = " ".join(x.strip() for x in parts if x.strip()).strip()
        return joined or None
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"].strip() or None
    return None


def _render_nodes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "".join(_render_nodes(item) for item in value)
    if not isinstance(value, dict):
        return ""
    typ = str(value.get("type", value.get("item_type", ""))).casefold()
    content = value.get("content", value.get("item_content"))
    if typ in {"equation_inline", "inline_equation"}:
        return f"${_render_nodes(content)}$"
    if typ == "text" and isinstance(content, str):
        return content
    if content is not None:
        return _render_nodes(content)
    preferred = (
        "paragraph_content", "title_content", "list_items", "page_header_content", "page_footer_content",
        "page_number_content", "page_footnote_content", "page_aside_text_content", "algorithm_content",
        "code_content", "image_caption", "chart_caption", "table_caption",
    )
    return "".join(_render_nodes(value.get(key)) for key in preferred if key in value)


def _caption(value: Any) -> str | None:
    rendered = clean_text(_render_nodes(value))
    return rendered or None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def extract_metadata(semantic: dict[str, Any]) -> dict[str, Any]:
    record = semantic.get("record", {}) if isinstance(semantic, dict) else {}
    content = record.get("content", {}) if isinstance(record, dict) else {}
    contributors = content.get("contributor", []) if isinstance(content, dict) else []
    authors = [str(item.get("name")).strip() for item in contributors if isinstance(item, dict) and item.get("name")]
    identifier = record.get("identifier") if isinstance(record, dict) else None
    return {
        "title": _semantic_values(content.get("title")) if isinstance(content, dict) else None,
        "abstract": _semantic_values(content.get("abstract")) if isinstance(content, dict) else None,
        "authors": authors,
        "version_id": str(identifier).strip() if identifier not in (None, "") else None,
    }


def _flat_records(data: list[Any]) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type", ""))
        raw_text = item.get("text") if isinstance(item.get("text"), str) else ""
        record: dict[str, Any] = {
            "raw_type": raw_type,
            "raw_text": raw_text,
            "page": item.get("page_idx") if isinstance(item.get("page_idx"), int) else None,
            "bbox": _bbox(item.get("bbox")),
            "level": item.get("text_level") if isinstance(item.get("text_level"), int) else None,
            "source_file": "content_list_process.json",
            "source_index": index,
            "parser_source": "content_list_process",
        }
        if raw_type == "text":
            record["kind"] = "heading" if record["level"] is not None else "paragraph"
        elif raw_type in {"image", "chart"}:
            record.update(kind="object", object_type="figure", object_subtype=raw_type,
                          caption=_caption(item.get("image_caption" if raw_type == "image" else "chart_caption")),
                          image_path=item.get("img_path") if isinstance(item.get("img_path"), str) else None)
        elif raw_type == "table":
            record.update(kind="object", object_type="table", object_subtype="table",
                          caption=_caption(item.get("table_caption")),
                          table_html=item.get("table_body") if isinstance(item.get("table_body"), str) else None,
                          image_path=item.get("img_path") if isinstance(item.get("img_path"), str) else None)
        elif raw_type in {"equation", "equation_interline"}:
            record.update(kind="object", object_type="formula", object_subtype=raw_type,
                          formula_latex=raw_text or None, object_text=raw_text or None)
        elif raw_type in EXCLUDED_TYPES:
            record.update(kind="excluded", exclusion=EXCLUDED_TYPES[raw_type])
        else:
            record.update(kind="uncertain", reason="unclassified_not_body")
        yield record


def _source_records(data: list[Any]) -> Iterable[dict[str, Any]]:
    source_index = 0
    for page, items in enumerate(data):
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("type", ""))
            content = item.get("content") if isinstance(item.get("content"), dict) else {}
            record: dict[str, Any] = {
                "raw_type": raw_type,
                "raw_text": "",
                "page": page,
                "bbox": _bbox(item.get("bbox")),
                "level": content.get("level") if isinstance(content.get("level"), int) else None,
                "source_file": "paragraph_article.source.json",
                "source_index": source_index,
                "parser_source": "paragraph_article_source",
            }
            source_index += 1
            if raw_type == "paragraph":
                record.update(kind="paragraph", raw_text=_render_nodes(content.get("paragraph_content")))
            elif raw_type == "title":
                record.update(kind="heading", raw_text=_render_nodes(content.get("title_content")))
            elif raw_type == "list" and content.get("list_type") == "reference_list":
                record.update(kind="excluded", exclusion="reference", raw_text=_render_nodes(content.get("list_items")))
            elif raw_type in {"image", "chart"}:
                image = content.get("image_source") if isinstance(content.get("image_source"), dict) else {}
                cap_key = "image_caption" if raw_type == "image" else "chart_caption"
                record.update(kind="object", object_type="figure", object_subtype=raw_type,
                              caption=_caption(content.get(cap_key)), image_path=image.get("path"))
            elif raw_type == "table":
                image = content.get("image_source") if isinstance(content.get("image_source"), dict) else {}
                record.update(kind="object", object_type="table", object_subtype="table",
                              caption=_caption(content.get("table_caption")),
                              table_html=content.get("html") if isinstance(content.get("html"), str) else None,
                              image_path=image.get("path"))
            elif raw_type in {"equation", "equation_interline"}:
                latex = content.get("math_content") if isinstance(content.get("math_content"), str) else None
                record.update(kind="object", object_type="formula", object_subtype=raw_type,
                              formula_latex=latex, object_text=latex, raw_text=latex or "")
            elif raw_type in {"algorithm", "code"}:
                cap_key = "algorithm_caption" if raw_type == "algorithm" else "code_caption"
                text_key = "algorithm_content" if raw_type == "algorithm" else "code_content"
                record.update(kind="object", object_type="other", object_subtype=raw_type,
                              caption=_caption(content.get(cap_key)), object_text=_render_nodes(content.get(text_key)))
            elif raw_type in EXCLUDED_TYPES:
                key = {
                    "page_header": "page_header_content", "page_footer": "page_footer_content",
                    "page_number": "page_number_content", "page_footnote": "page_footnote_content",
                }.get(raw_type)
                record.update(kind="excluded", exclusion=EXCLUDED_TYPES[raw_type],
                              raw_text=_render_nodes(content.get(key)) if key else "")
            else:
                record.update(kind="uncertain", reason="unclassified_not_body",
                              raw_text=_render_nodes(content))
            yield record


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": record["source_file"],
        "source_page": record["page"],
        "source_bbox": record["bbox"],
        "parser_source": record["parser_source"],
        "source_index": record["source_index"],
    }


def parse_document(
    paper_dir: Path,
    project_root: Path,
    source_id: str | None = None,
    version_id: str | None = None,
    force_parser: str | None = None,
) -> dict[str, Any]:
    paper_id = paper_dir.name
    semantic_path = paper_dir / "paragraph_article.json"
    source_path = paper_dir / "paragraph_article.source.json"
    flat_path = paper_dir / "content_list_process.json"
    semantic = load_json(semantic_path)
    metadata = extract_metadata(semantic)

    if force_parser == "content_list_process" or (force_parser is None and flat_path.is_file()):
        if not flat_path.is_file():
            raise FileNotFoundError(flat_path)
        parser_source = "content_list_process"
        records = list(_flat_records(load_json(flat_path)))
    elif force_parser in (None, "paragraph_article_source"):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        parser_source = "paragraph_article_source"
        records = list(_source_records(load_json(source_path)))
    else:
        raise ValueError(f"Unknown parser source: {force_parser}")

    title_norm = _norm_label(metadata["title"])
    structural_headings = [
        r for r in records if r["kind"] == "heading" and not (r.get("level") == 1 and _norm_label(r["raw_text"]) == title_norm)
    ]
    has_structural_headings = bool(structural_headings)
    sections: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    excluded = Counter()
    current_path: list[str] = []
    current_section_id: str | None = None
    current_major = ""

    for record in records:
        kind = record["kind"]
        if kind == "heading":
            heading = clean_text(record["raw_text"])
            raw_level = record.get("level") or 2
            if raw_level == 1 and title_norm and _norm_label(heading) == title_norm:
                excluded["document_title"] += 1
                continue
            level = max(1, int(raw_level) - 1)
            current_path = current_path[: level - 1] + [heading]
            current_major = current_path[0] if current_path else heading
            current_section_id = f"{paper_id}:s{len(sections) + 1:04d}"
            sections.append({
                "section_id": current_section_id,
                "section_path": current_path.copy(),
                "major_section": current_major or None,
                "heading": heading or None,
                "level": level,
                "page": record["page"],
                "bbox": record["bbox"],
                "provenance": _provenance(record),
            })
            continue
        if kind == "excluded":
            excluded[record["exclusion"]] += 1
            continue
        if kind == "paragraph":
            raw = record["raw_text"]
            cleaned = clean_text(raw)
            if not cleaned:
                excluded["empty_paragraph"] += 1
                continue
            major_norm = _heading_key(current_major)
            if major_norm in REFERENCE_HEADINGS:
                excluded["reference_section_paragraph"] += 1
                continue
            if major_norm in ABSTRACT_HEADINGS:
                excluded["paper_level_abstract_or_keywords"] += 1
                continue
            if has_structural_headings and current_section_id is None:
                excluded["front_matter_unsectioned"] += 1
                continue
            paragraphs.append({
                "paragraph_id": f"{paper_id}:p{len(paragraphs) + 1:06d}",
                "raw_text": raw,
                "clean_text": cleaned,
                "page": record["page"],
                "bbox": record["bbox"],
                "section_id": current_section_id,
                "suspected_word_split": suspected_word_split(raw),
                "provenance": _provenance(record),
            })
            continue
        if kind == "object":
            objects.append({
                "object_id": f"{paper_id}:o{len(objects) + 1:06d}",
                "object_type": record.get("object_type"),
                "object_subtype": record.get("object_subtype") or record["raw_type"],
                "page": record["page"],
                "bbox": record["bbox"],
                "caption": record.get("caption"),
                "table_html": record.get("table_html"),
                "formula_latex": record.get("formula_latex"),
                "image_path": record.get("image_path"),
                "object_text": record.get("object_text"),
                "provenance": _provenance(record),
            })
            continue
        uncertain.append({
            "element_id": f"{paper_id}:u{len(uncertain) + 1:06d}",
            "raw_type": record["raw_type"] or None,
            "raw_text": record.get("raw_text") or None,
            "page": record["page"],
            "bbox": record["bbox"],
            "reason": record.get("reason", "unclassified_not_body"),
            "provenance": _provenance(record),
        })

    document = {
        "paper_id": paper_id,
        "source_id": source_id or paper_id,
        "version_id": version_id if version_id is not None else metadata["version_id"],
        "title": metadata["title"],
        "abstract": metadata["abstract"],
        "authors": metadata["authors"],
        "source_path": relative_posix(paper_dir, project_root),
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "parser_source": parser_source,
        "page_numbering": "zero_based_mineru_page_idx",
        "sections": sections,
        "paragraphs": paragraphs,
        "objects": objects,
        "uncertain_elements": uncertain,
        "excluded_counts_json": canonical_json(dict(sorted(excluded.items()))),
    }
    errors = validate_document(document)
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return document
