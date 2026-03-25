"""Cache for model responses. Stored at results/<model_id>/cache/responses.json."""

import json
from pathlib import Path


def trial_hash(trial: dict) -> str:
    """Deterministic hash of trial inputs for cache lookup."""
    pass


def load_cache(cache_path: Path) -> dict:
    """Load cache dict from disk, or return empty dict if missing."""
    pass


def save_cache(cache_path: Path, cache: dict) -> None:
    """Write cache dict to disk."""
    pass
