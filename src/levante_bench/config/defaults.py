"""Default paths and LEVANTE assets bucket URL."""

import os
from pathlib import Path

LEVANTE_ASSETS_BUCKET_URL = "https://storage.googleapis.com/levante-assets-prod"


def _project_root() -> Path:
    """Project root: directory containing pyproject.toml or data/."""
    p = Path(__file__).resolve()
    for _ in range(4):
        p = p.parent
        if (p / "data").is_dir() or (p / "pyproject.toml").exists():
            return p
    return p


def get_data_root() -> Path:
    """Data directory. Override with LEVANTE_BENCH_DATA_ROOT env var."""
    env = os.environ.get("LEVANTE_BENCH_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return _project_root() / "data"


def get_assets_base_url() -> str:
    """Base URL for LEVANTE assets bucket."""
    return os.environ.get("LEVANTE_ASSETS_BUCKET_URL", LEVANTE_ASSETS_BUCKET_URL)
