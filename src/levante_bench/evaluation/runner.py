"""Run one or all tasks for one or many models; write model outputs to disk."""

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from levante_bench.config import get_data_root, get_task_def, list_tasks
from levante_bench.data.datasets import LevanteDataset, collate_levante
from levante_bench.data.loaders import load_trials_for_task
from levante_bench.evaluation.outputs import write_task_output
from levante_bench.models import get_model_class


def run_eval(
    task_ids: list[str] | None = None,
    model_ids: list[str] | None = None,
    version: str = "current",
    device: str = "cpu",
    output_dir: Path | str | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Run evaluation for given tasks and models; write .npy per task/model to output_dir.
    Returns dict of (task_id, model_id) -> output path.
    """
    data_root = data_root or get_data_root()
    output_dir = Path(output_dir) if output_dir else data_root.parent / "results" / version
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = task_ids or list_tasks()
    if not model_ids:
        from levante_bench.models import list_models
        model_ids = list_models()
    if not model_ids:
        return {}
    results: dict[tuple[str, str], Path] = {}
    for task_id in task_ids:
        task_def = get_task_def(task_id, version)
        if not task_def:
            continue
        manifest = load_trials_for_task(task_id, version, data_root=data_root)
        if manifest.empty:
            continue
        base_path = data_root / "assets" / version
        dataset = LevanteDataset(manifest, base_path=base_path)
        if len(dataset) == 0:
            continue
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_levante,
        )
        for model_id in model_ids:
            model_cls = get_model_class(model_id)
            if not model_cls:
                continue
            try:
                model = model_cls(device=device)
            except Exception:
                continue
            scores = _get_scores(model, dataloader, task_def.n_options)
            if scores is not None:
                path = write_task_output(output_dir, task_id, model_id, scores)
                results[(task_id, model_id)] = path
    return results


def _get_scores(model: Any, dataloader: DataLoader, n_options: int) -> np.ndarray | None:
    """Get (n_trials, n_options) scores from model (EvalModel or GenEvalModel)."""
    if hasattr(model, "get_all_sim_scores"):
        raw = model.get_all_sim_scores(dataloader)
    else:
        return None
    # raw shape: (n_batches, ...) e.g. (n_trials, 4, 1) for CLIP image-text
    if raw.ndim == 3:
        if raw.shape[2] == 1:
            raw = raw.squeeze(-1)
        else:
            raw = raw.reshape(raw.shape[0], -1)
    if raw.ndim == 2 and raw.shape[1] >= n_options:
        return raw[:, :n_options].astype(np.float64)
    return raw.astype(np.float64)
