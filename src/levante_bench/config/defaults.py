"""Default paths and LEVANTE assets bucket URL (public, no auth)."""

import os
from pathlib import Path


def get_task_mapping_path() -> Path:
    """Path to task_name_mapping.csv in the config package."""
    return Path(__file__).resolve().parent / "task_name_mapping.csv"

# Public GCP bucket for LEVANTE production assets (no .secrets required)
LEVANTE_ASSETS_BUCKET_URL = "https://storage.googleapis.com/levante-assets-prod"

# Project root: directory containing "src" and "data" or pyproject.toml
def _project_root() -> Path:
    p = Path(__file__).resolve()
    # From config/defaults.py: config -> levante_bench -> src -> project_root
    for _ in range(4):
        p = p.parent
        if (p / "data").is_dir() or (p / "pyproject.toml").exists():
            return p
    return p


def get_data_root() -> Path:
    """Directory for data/ (raw trials, assets). Default: project_root/data."""
    return _project_root() / "data"


def get_assets_base_url() -> str:
    """Base URL for LEVANTE assets bucket. Override via env LEVANTE_ASSETS_BUCKET_URL if needed."""
    return os.environ.get("LEVANTE_ASSETS_BUCKET_URL", LEVANTE_ASSETS_BUCKET_URL)
