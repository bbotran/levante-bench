#!/usr/bin/env python3
"""
Download LEVANTE corpus and visual assets from the public GCP bucket.
Writes to data/assets/<version>/ and builds an item_uid -> local paths index.
Version defaults to today (YYYY-MM-DD). Idempotent.
"""

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# Resolve package from script: repo root is parent of scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _project_root() -> Path:
    return _REPO_ROOT


def _add_src_to_path() -> None:
    src = _REPO_ROOT / "src"
    if src.exists() and str(src) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(src))


_add_src_to_path()

from levante_bench.config import get_assets_base_url, get_task_mapping_path  # noqa: E402


def _bucket_name_from_base(base: str) -> str:
    """e.g. https://storage.googleapis.com/levante-assets-prod -> levante-assets-prod"""
    return base.rstrip("/").split("/")[-1]


def _list_bucket_keys(bucket_name: str, prefix: str) -> list[str]:
    """List object keys under prefix via GCP XML API (public bucket)."""
    url = f"https://{bucket_name}.storage.googleapis.com"
    ns = "http://doc.s3.amazonaws.com/2006-03-01"
    params: dict[str, str] = {"prefix": prefix, "list-type": "2"}
    keys: list[str] = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall(f".//{{{ns}}}Contents"):
            k = c.find(f"{{{ns}}}Key")
            if k is not None and k.text:
                keys.append(k.text)
        truncated = root.find(f".//{{{ns}}}IsTruncated")
        if truncated is not None and truncated.text == "true":
            next_token = root.find(f".//{{{ns}}}NextContinuationToken")
            if next_token is not None and next_token.text:
                params = {"prefix": prefix, "list-type": "2", "continuation-token": next_token.text}
            else:
                break
        else:
            break
    return keys


def _download_file(base_url: str, key: str, dest: Path) -> None:
    url = f"{base_url.rstrip('/')}/{key}"
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)


def _load_task_mapping(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
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


def run(
    version: str | None = None,
    task_filter: str | None = None,
    data_root: Path | None = None,
    base_url: str | None = None,
) -> None:
    from datetime import datetime

    data_root = data_root or _project_root() / "data"
    base_url = base_url or get_assets_base_url()
    version = version or datetime.now().strftime("%Y-%m-%d")
    assets_dir = data_root / "assets" / version
    corpus_dir = assets_dir / "corpus"
    visual_dir = assets_dir / "visual"

    mapping_path = get_task_mapping_path()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Task mapping not found: {mapping_path}")
    tasks = _load_task_mapping(mapping_path)
    if task_filter:
        tasks = [t for t in tasks if t["internal_name"] == task_filter or t["benchmark_name"] == task_filter]
    if not tasks:
        return

    bucket_name = _bucket_name_from_base(base_url)
    index: dict[str, dict] = {}  # item_uid -> { task, internal_name, corpus_row, image_paths }

    for t in tasks:
        internal_name = t["internal_name"]
        corpus_file = t["corpus_file"]
        # 1) Corpus CSV
        corpus_key = f"corpus/{internal_name}/{corpus_file}"
        out_corpus = corpus_dir / internal_name / corpus_file
        if not out_corpus.exists():
            _download_file(base_url, corpus_key, out_corpus)
        df = pd.read_csv(out_corpus)
        if "item_uid" not in df.columns:
            continue
        # 2) Visual assets under visual/{internal_name}/
        prefix = f"visual/{internal_name}/"
        try:
            keys = _list_bucket_keys(bucket_name, prefix)
        except Exception:
            keys = []
        visual_local_dir = visual_dir / internal_name
        for key in keys:
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            local_path = visual_local_dir / rel
            if not local_path.exists():
                _download_file(base_url, key, local_path)
        # 3) Build index rows from corpus
        for _, row in df.iterrows():
            uid = row.get("item_uid")
            if pd.isna(uid):
                continue
            uid = str(uid).strip()
            corpus_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            # Optional: derive image paths from corpus (e.g. columns that look like image paths)
            image_paths: list[str] = []
            for c in df.columns:
                if re.match(r"image[0-9]+", c, re.I) or "image" in c.lower() and "path" in c.lower():
                    v = row.get(c)
                    if not pd.isna(v) and v:
                        rel = str(v).strip()
                        local = visual_local_dir / rel
                        # Store path relative to assets_dir for portability
                        try:
                            image_paths.append(str(local.relative_to(assets_dir)))
                        except ValueError:
                            image_paths.append(str(local))
            index[uid] = {
                "task": t["benchmark_name"],
                "internal_name": internal_name,
                "corpus_row": corpus_row,
                "image_paths": image_paths,
            }

    index_path = assets_dir / "item_uid_index.json"
    # Convert non-JSON-serializable (e.g. numpy) in corpus_row
    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    index_serializable = {}
    for uid, v in index.items():
        cr = v.get("corpus_row") or {}
        paths = v.get("image_paths") or []
        index_serializable[uid] = {
            "task": v.get("task"),
            "internal_name": v.get("internal_name"),
            "corpus_row": {k: _serialize(cr[k]) for k in cr},
            "image_paths": paths,
        }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_serializable, f, indent=2)
    print(f"Wrote {index_path} ({len(index_serializable)} item_uids)")


def main() -> None:
    p = argparse.ArgumentParser(description="Download LEVANTE assets and build item_uid index.")
    p.add_argument("--version", default=None, help="Asset version (default: today YYYY-MM-DD)")
    p.add_argument("--task", default=None, help="Only download this task (internal_name or benchmark_name)")
    p.add_argument("--data-root", type=Path, default=None, help="Data root (default: project data/)")
    p.add_argument("--base-url", default=None, help="Bucket base URL (default: config)")
    args = p.parse_args()
    run(version=args.version, task_filter=args.task, data_root=args.data_root, base_url=args.base_url)


if __name__ == "__main__":
    main()
