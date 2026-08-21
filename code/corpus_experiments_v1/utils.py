from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_checksums(directory: Path, filenames: Iterable[str]) -> dict[str, str]:
    rows: list[str] = []
    checksums: dict[str, str] = {}
    for name in sorted(filenames):
        path = directory / name
        if not path.is_file():
            continue
        digest = sha256_file(path)
        checksums[name] = digest
        rows.append(f"{digest}  {name}")
    (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="ascii")
    return checksums


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")


def parse_simple_yaml_model(path: Path) -> dict[str, Any]:
    """Read the frozen model block without adding a YAML dependency."""
    result: dict[str, Any] = {}
    in_model = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw and not raw.startswith(" "):
            in_model = raw.strip() == "model:"
            continue
        if not in_model or ":" not in raw:
            continue
        key, value = raw.strip().split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        elif value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value
    required = {"repository_id", "snapshot", "revision", "tokenizer_bundle_sha256"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Frozen model config lacks fields: {sorted(missing)}")
    return result
