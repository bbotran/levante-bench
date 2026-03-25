"""Config loader using OmegaConf. Merges experiment + model + task configs."""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def get_configs_root() -> Path:
    """Path to configs/ directory at project root."""
    return Path(__file__).resolve().parent.parent.parent.parent / "configs"


def load_experiment(experiment_path: str | Path | None = None, cli_overrides: list[str] | None = None) -> DictConfig:
    """Load experiment config, merge in model/task configs and CLI overrides.

    Args:
        experiment_path: Path to experiment YAML. Defaults to configs/experiment.yaml.
        cli_overrides: List of dotlist overrides e.g. ["device=cuda", "models=[smolvlm2]"].

    Returns:
        Merged DictConfig with experiment, model, and task configs resolved.
    """
    pass


def load_model_config(model_name: str) -> DictConfig:
    """Load a single model config from configs/models/<model_name>.yaml."""
    pass


def load_task_config(task_id: str) -> DictConfig:
    """Load a single task config from configs/tasks/<task_id>.yaml."""
    pass
