"""Save model outputs (e.g. .npy per task/model) for R comparison."""

from pathlib import Path

import numpy as np


def write_task_output(
    output_dir: Path | str,
    task_id: str,
    model_id: str,
    scores: np.ndarray,
) -> Path:
    """
    Write model output for one task/model to output_dir/<model_id>/<task_id>.npy.
    scores: shape (n_trials, n_options) of logits or similarity scores.
    """
    output_dir = Path(output_dir)
    model_dir = output_dir / _safe_id(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{_safe_id(task_id)}.npy"
    np.save(path, scores)
    return path


def _safe_id(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)
