"""Load task name mapping and item_uid -> local paths index; expose lookup for loaders."""

from pathlib import Path
from typing import Any

from levante_bench.config import get_data_root, get_task_mapping_path


def load_task_mapping() -> list[dict[str, str]]:
    """Load task_name_mapping.csv; returns list of dicts with benchmark_name, internal_name, corpus_file."""
    path = get_task_mapping_path()
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        import csv

        for row in csv.DictReader(f):
            internal = (row.get("internal_name") or "").strip()
            corpus = (row.get("corpus_file") or "").strip()
            if internal and corpus:
                rows.append(
                    {
                        "benchmark_name": (row.get("benchmark_name") or "").strip(),
                        "internal_name": internal,
                        "corpus_file": corpus,
                    }
                )
    return rows


def _asset_index_path(version: str) -> Path:
    return get_data_root() / "assets" / version / "item_uid_index.json"


def load_asset_index(version: str) -> dict[str, dict[str, Any]]:
    """Load item_uid -> { task, internal_name, corpus_row, image_paths } from data/assets/<version>/item_uid_index.json."""
    path = _asset_index_path(version)
    if not path.exists():
        return {}
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_paths(item_uid: str, task_id: str, version: str = "latest") -> dict[str, Any] | None:
    """
    Look up (item_uid, task_id) in the asset index and return corpus row + local paths.

    task_id can be benchmark_name or internal_name. Returns dict with keys:
    - corpus_row: dict of corpus columns for this item
    - image_paths: list of local Paths (or paths as strings)
    - task, internal_name

    Returns None if not found.
    """
    if version == "latest":
        assets_base = get_data_root() / "assets"
        if not assets_base.exists():
            return None
        subdirs = sorted((p for p in assets_base.iterdir() if p.is_dir()), reverse=True)
        version = subdirs[0].name if subdirs else ""
    index = load_asset_index(version)
    if not index:
        return None
    entry = index.get(item_uid)
    if not entry:
        return None
    if entry.get("task") != task_id and entry.get("internal_name") != task_id:
        return None
    image_paths = entry.get("image_paths") or []
    base = get_data_root() / "assets" / version
    resolved: list[Path] = []
    for p in image_paths:
        pp = Path(p) if not isinstance(p, Path) else p
        resolved.append(base / pp if not pp.is_absolute() else pp)
    return {
        "task": entry.get("task"),
        "internal_name": entry.get("internal_name"),
        "corpus_row": entry.get("corpus_row") or {},
        "image_paths": resolved,
    }
