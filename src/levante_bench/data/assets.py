"""Asset index: item_uid -> local paths lookup for loaders."""

from pathlib import Path
from typing import Any

from levante_bench.config import get_data_root


def _asset_index_path(version: str) -> Path:
    return get_data_root() / "assets" / version / "item_uid_index.json"


def load_asset_index(version: str) -> dict[str, dict[str, Any]]:
    """Load item_uid index from data/assets/<version>/item_uid_index.json."""
    path = _asset_index_path(version)
    if not path.exists():
        return {}
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_paths(item_uid: str, task_id: str, version: str = "latest") -> dict[str, Any] | None:
    """Look up item_uid in asset index. Returns corpus_row + resolved image_paths, or None."""
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
