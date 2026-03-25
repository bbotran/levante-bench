"""CLI entry point. Uses OmegaConf for config loading + CLI overrides."""

import sys

from levante_bench.config.loader import load_experiment


def main() -> int:
    """Load experiment config with CLI overrides, run evaluation.

    Usage:
        python -m levante_bench.cli experiment=configs/experiment.yaml
        python -m levante_bench.cli experiment=configs/experiment.yaml device=cuda
    """
    cli_args = sys.argv[1:]

    # Extract experiment path from CLI args
    experiment_path = None
    overrides = []
    for arg in cli_args:
        if arg.startswith("experiment="):
            experiment_path = arg.split("=", 1)[1]
        else:
            overrides.append(arg)

    if not experiment_path:
        print("Required: experiment=<path_to_config.yaml>", file=sys.stderr)
        return 1

    cfg = load_experiment(experiment_path=experiment_path, cli_overrides=overrides)

    from levante_bench.evaluation.runner import run_eval
    results = run_eval(cfg)

    if not results:
        print("No results produced.", file=sys.stderr)
        return 1

    for model_id, path in results.items():
        print(f"  {model_id}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
