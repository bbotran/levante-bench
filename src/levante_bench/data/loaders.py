"""Load manifest and human data from disk for a task."""

from pathlib import Path

import pandas as pd

from levante_bench.config import get_data_root, get_task_def
from levante_bench.data.assets import get_paths


def load_trials_csv(path: Path) -> pd.DataFrame:
    """Load trials from a CSV (trials.csv or tasks/<task>_trials.csv)."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_trials_for_task(
    task_id: str,
    version: str,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """
    Load trials for a task and resolve image/text paths via asset index.
    Returns a DataFrame with trial_id, item_uid, and columns image1..imageN, text1
    (paths and prompt) for evaluation.
    """
    data_root = data_root or get_data_root()
    task_def = get_task_def(task_id, version)
    if not task_def:
        return pd.DataFrame()
    raw = data_root / "raw" / version
    trials_path = raw / "trials.csv"
    safe = _safe_task_id(task_id)
    task_trials_path = raw / "tasks" / f"{safe}_trials.csv"
    if task_trials_path.exists():
        df = pd.read_csv(task_trials_path)
    elif trials_path.exists():
        df = pd.read_csv(trials_path)
        if "task_id" in df.columns:
            df = df[df["task_id"] == task_id]
    else:
        return pd.DataFrame()
    # Normalize column names (strip BOM / whitespace)
    df.columns = df.columns.str.strip()
    item_uid_col = "item_uid"
    if item_uid_col not in df.columns:
        # Try case-insensitive or common variant
        for c in df.columns:
            if c.strip().lower() == "item_uid":
                item_uid_col = c
                break
        else:
            return pd.DataFrame()
    if df.empty:
        return df
    n_options = getattr(task_def, "n_options", 4)
    rows = []
    for _, row in df.iterrows():
        item_uid = str(row.get(item_uid_col, "")).strip()
        trial_id = str(row.get("trial_id", row.get("trial_number", len(rows))))
        paths = get_paths(item_uid, task_id, version) if item_uid else None
        if not paths:
            paths = {"corpus_row": {}, "image_paths": []}
        corpus = paths.get("corpus_row") or {}
        prompt = corpus.get("prompt") or ""
        image_paths = paths.get("image_paths") or []
        out = {"trial_id": trial_id, "item_uid": item_uid}
        for i in range(1, n_options + 1):
            out[f"image{i}"] = str(image_paths[i - 1]) if i <= len(image_paths) else ""
        out["text1"] = str(prompt) if prompt else ""
        rows.append(out)
    df_out = pd.DataFrame(rows)
    # One row per unique item_uid so model runs once per item; human comparison is by item_uid (and age_bin)
    if "item_uid" in df_out.columns and not df_out.empty:
        df_out = df_out.drop_duplicates(subset=["item_uid"], keep="first").reset_index(drop=True)
    return df_out


def _safe_task_id(task_id: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)


def load_human_response_path(task_id: str, version: str) -> Path | None:
    """Path to human response aggregates CSV for task (if any)."""
    task_def = get_task_def(task_id, version)
    if not task_def or not task_def.human_response_path:
        return None
    return task_def.human_response_path if task_def.human_response_path.exists() else None
