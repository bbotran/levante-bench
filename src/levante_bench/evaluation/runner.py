"""Run evaluation: for each model, evaluate all tasks, write results."""

import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from levante_bench.config import get_task_def, load_model_config, load_task_config
from levante_bench.evaluation.cache import load_cache, save_cache, trial_hash
from levante_bench.evaluation.outputs import write_task_csv, write_summary_csv
from levante_bench.models import get_model_class
from levante_bench.tasks import get_task_dataset


def run_eval(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate each model across all tasks using experiment config.

    Flow:
        for model in cfg.models:
            load model config + model once
            cache at results/<model>/<version>/cache/responses.json
            for task in cfg.tasks:
                check model capabilities vs task context_type
                load task dataset
                for trial in dataset:
                    check cache → skip if exists
                    model.evaluate_trial(trial)
                    write to cache
                write per-task CSV to results/<model>/<version>/<task_id>.csv
            write summary CSV to results/<model>/<version>/summary.csv

    Returns dict of model_id -> summary CSV path.
    """
    pass
