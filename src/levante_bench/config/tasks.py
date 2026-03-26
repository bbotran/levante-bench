"""Task registry: LEVANTE tasks with paths and schema (manifest, human response, n_options)."""

from levante_bench.config.defaults import get_data_root, get_task_mapping_path
from levante_bench.data.schema import TaskDef

# LEVANTE item-based tasks (from task_name_mapping.csv). Hearts and Flowers and Memory excluded (no items per se).
# task_id used for paths and lookup; we use internal_name when set, else normalized benchmark_name.
def _load_registry() -> list[dict]:
    path = get_task_mapping_path()
    if not path.exists():
        return _default_registry()
    import csv

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            benchmark_name = (row.get("benchmark_name") or "").strip()
            internal_name = (row.get("internal_name") or "").strip()
            corpus_file = (row.get("corpus_file") or "").strip() or None
            task_id = internal_name if internal_name else benchmark_name.lower().replace(" ", "_")
            rows.append(
                {
                    "task_id": task_id,
                    "benchmark_name": benchmark_name,
                    "internal_name": internal_name or task_id,
                    "corpus_file": corpus_file,
                    "task_type": "forced-choice",
                    "n_options": 4,
                    "has_correct": True,
                }
            )
    return rows


def _default_registry() -> list[dict]:
    """Fallback when task_name_mapping.csv is missing; matches current item-based tasks only."""
    return [
        {"task_id": "egma-math", "benchmark_name": "Math", "internal_name": "egma-math", "corpus_file": "test-combined-math-cat.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
        {"task_id": "trog", "benchmark_name": "Sentence Understanding", "internal_name": "trog", "corpus_file": "trog-item-bank-full-params.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
        {"task_id": "vocab", "benchmark_name": "Vocabulary", "internal_name": "vocab", "corpus_file": "vocab-item-bank-cat.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
        {"task_id": "theory-of-mind", "benchmark_name": "Stories", "internal_name": "theory-of-mind", "corpus_file": "theory-of-mind-item-bank.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
        {"task_id": "same-different-selection", "benchmark_name": "Same and Different", "internal_name": "same-different-selection", "corpus_file": "same-different-selection-item-bank.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
        {"task_id": "matrix-reasoning", "benchmark_name": "Pattern Matching", "internal_name": "matrix-reasoning", "corpus_file": "matrix-reasoning-corpus-retest.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
        {"task_id": "mental-rotation", "benchmark_name": "Shape Rotation", "internal_name": "mental-rotation", "corpus_file": "mental-rotation-item-bank.csv", "task_type": "forced-choice", "n_options": 4, "has_correct": True},
    ]


def _safe_task_id(task_id: str) -> str:
    """Filename-safe task id (matches R script: gsub('[^a-zA-Z0-9_-]', '_', tid))."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)


def get_task_def(task_id: str, version: str) -> TaskDef | None:
    """Return TaskDef for task_id with paths resolved under data/responses/<version>/."""
    registry = _load_registry()
    data_root = get_data_root()
    raw = data_root / "responses" / version
    safe = _safe_task_id(task_id)
    for r in registry:
        if r["task_id"] == task_id or r["benchmark_name"] == task_id or r["internal_name"] == task_id:
            manifest_path = raw / "tasks" / f"{safe}_trials.csv"
            if not manifest_path.exists():
                manifest_path = raw / "trials.csv"  # fallback: load from global and filter by task_id
            human_path = raw / "human" / f"{safe}.csv"
            if not human_path.exists():
                human_path = None
            return TaskDef(
                task_id=r["task_id"],
                benchmark_name=r["benchmark_name"],
                internal_name=r["internal_name"],
                manifest_path=manifest_path if manifest_path.exists() else None,
                human_response_path=human_path,
                task_type=r["task_type"],
                n_options=r["n_options"],
                has_correct=r["has_correct"],
                corpus_file=r.get("corpus_file"),
            )
    return None


def list_tasks() -> list[str]:
    """Return list of registered task_ids."""
    return [r["task_id"] for r in _load_registry()]
