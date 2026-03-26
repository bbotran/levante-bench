"""CLI: run-eval, run-benchmark, run-workflow, run-comparison."""

import argparse
import sys
from pathlib import Path

from levante_bench.cli_workflows import (
    DEFAULT_DATA_VERSION,
    DEFAULT_SMOLVLM2_MODEL,
    WORKFLOW_SCRIPTS,
    benchmark_command,
    normalize_passthrough,
    project_root,
    run_command,
    workflow_command,
    workflow_script_path,
)


def _project_root() -> Path:
    return project_root()


def cmd_list_tasks(_: argparse.Namespace) -> int:
    from levante_bench.config import list_tasks
    for t in list_tasks():
        print(t)
    return 0


def cmd_list_models(_: argparse.Namespace) -> int:
    from levante_bench.models import list_models
    for m in list_models():
        print(m)
    return 0


def cmd_run_eval(args: argparse.Namespace) -> int:
    from levante_bench.evaluation.runner import resolve_device, run_eval
    task_ids = args.task if args.task else None
    model_ids = args.model if args.model else None
    version = args.version or "current"
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = _project_root() / "results" / version
    data_root = _project_root() / "data"
    print(f"Running evaluation: version={version}, device={device}, output={output_dir}")
    print(f"  Data root: {data_root}")
    if task_ids:
        print(f"  Tasks: {', '.join(task_ids)}")
    if model_ids:
        print(f"  Models: {', '.join(model_ids)}")
    results = run_eval(
        task_ids=task_ids,
        model_ids=model_ids,
        version=version,
        device=device,
        output_dir=output_dir,
        data_root=data_root,
    )
    if not results:
        print("No outputs written. Check that data/responses/<version>/ and data/assets/<version>/ exist and item_uid index matches trials.", file=sys.stderr)
        return 1
    print(f"Success: wrote {len(results)} file(s)")
    for (task_id, model_id), path in results.items():
        print(f"  {task_id}\t{model_id}\t{path}")
    return 0


def cmd_check_gpu(_: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed in the active environment.", file=sys.stderr)
        return 1
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    print(f"cuda_available={available}")
    print(f"gpu_count={count}")
    for i in range(count):
        print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")
    return 0


def cmd_run_workflow(args: argparse.Namespace) -> int:
    root = _project_root()
    script_path = workflow_script_path(root, args.workflow)
    if not script_path.exists():
        print(f"Workflow script not found: {script_path}", file=sys.stderr)
        return 1
    cmd = workflow_command(root, args.workflow, args.script_args)
    print("Running workflow:", " ".join(cmd))
    return run_command(cmd, cwd=root)


def cmd_run_benchmark(args: argparse.Namespace) -> int:
    from levante_bench.evaluation.runner import resolve_device

    root = _project_root()
    device = resolve_device(args.device)
    data_version = args.data_version or DEFAULT_DATA_VERSION
    model_id = args.model_id or DEFAULT_SMOLVLM2_MODEL

    try:
        cmd = benchmark_command(
            root=root,
            benchmark=args.benchmark,
            data_version=data_version,
            model_id=model_id,
            device=device,
            max_items_math=args.max_items_math,
            max_items_tom=args.max_items_tom,
            max_items_vocab=args.max_items_vocab,
            extra_args=normalize_passthrough(args.benchmark_args),
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print("Running benchmark:", " ".join(cmd))
    return run_command(cmd, cwd=root)


def cmd_run_comparison(args: argparse.Namespace) -> int:
    root = _project_root()
    script = root / "comparison" / "compare_levante.R"
    if not script.exists():
        print("comparison/compare_levante.R not found", file=sys.stderr)
        return 1
    if not args.task or not args.model:
        print("run-comparison requires --task and --model", file=sys.stderr)
        return 1
    cmd = [
        "Rscript",
        str(script),
        "--task", args.task,
        "--model", args.model,
        "--version", args.version or "current",
        "--results-dir", args.results_dir or "results",
        "--project-root", str(root),
        "--output-dir", str(root / (args.output_dir or "results/comparison")),
    ]
    if getattr(args, "output_dkl", None):
        cmd.extend(["--output-dkl", args.output_dkl])
    if getattr(args, "output_accuracy", None):
        cmd.extend(["--output-accuracy", args.output_accuracy])
    return run_command(cmd, cwd=root)


def add_list_tasks_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list-tasks", help="List registered task IDs")


def add_list_models_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list-models", help="List registered model IDs")


def add_check_gpu_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("check-gpu", help="Report local CUDA/GPU availability")


def add_run_eval_parser(sub: argparse._SubParsersAction) -> None:
    pe = sub.add_parser("run-eval", help="Run evaluation (write .npy per task/model)")
    pe.add_argument("--task", action="append", help="Task ID (repeat for multiple)")
    pe.add_argument("--model", action="append", help="Model ID (repeat for multiple)")
    pe.add_argument("--version", default="current", help="Data/asset version")
    pe.add_argument("--device", default="auto", help="Device for model: auto|cpu|cuda")
    pe.add_argument("--output-dir", help="Output directory (default: results/<version>)")


def add_run_workflow_parser(sub: argparse._SubParsersAction) -> None:
    pw = sub.add_parser("run-workflow", help="Run integrated benchmark/test workflow scripts")
    pw.add_argument("--workflow", required=True, choices=sorted(WORKFLOW_SCRIPTS.keys()))
    pw.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the workflow script (prefix with --).",
    )


def add_run_benchmark_parser(sub: argparse._SubParsersAction) -> None:
    pb = sub.add_parser("run-benchmark", help="Run integrated benchmark presets (v1, vocab)")
    pb.add_argument("--benchmark", required=True, choices=["v1", "vocab"])
    pb.add_argument("--data-version", default=DEFAULT_DATA_VERSION, help="Data/assets version")
    pb.add_argument("--model-id", default=DEFAULT_SMOLVLM2_MODEL, help="Model id")
    pb.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    pb.add_argument("--max-items-math", type=int, default=None, help="Optional cap for v1 math")
    pb.add_argument("--max-items-tom", type=int, default=None, help="Optional cap for v1 ToM")
    pb.add_argument("--max-items-vocab", type=int, default=None, help="Optional cap for vocab benchmark")
    pb.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the underlying benchmark script (prefix with --).",
    )


def add_run_comparison_parser(sub: argparse._SubParsersAction) -> None:
    pc = sub.add_parser("run-comparison", help="Run R comparison (D_KL by age+item_uid, accuracy by item_uid)")
    pc.add_argument("--task", required=True, help="Task ID")
    pc.add_argument("--model", required=True, help="Model ID")
    pc.add_argument("--version", default="current", help="Data version")
    pc.add_argument("--results-dir", default="results", help="Results directory name")
    pc.add_argument("--output-dir", default="results/comparison", help="Directory for D_KL and accuracy CSVs")
    pc.add_argument("--output-dkl", help="Output path for D_KL CSV (default: <output-dir>/<task>_<model>_d_kl.csv)")
    pc.add_argument("--output-accuracy", help="Output path for accuracy CSV (default: <output-dir>/<task>_<model>_accuracy.csv)")


def main() -> int:
    parser = argparse.ArgumentParser(prog="levante-bench", description="LEVANTE VLM benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    add_list_tasks_parser(sub)
    add_list_models_parser(sub)
    add_check_gpu_parser(sub)
    add_run_eval_parser(sub)
    add_run_workflow_parser(sub)
    add_run_benchmark_parser(sub)
    add_run_comparison_parser(sub)
    args = parser.parse_args()
    if args.command == "list-tasks":
        return cmd_list_tasks(args)
    if args.command == "list-models":
        return cmd_list_models(args)
    if args.command == "check-gpu":
        return cmd_check_gpu(args)
    if args.command == "run-eval":
        return cmd_run_eval(args)
    if args.command == "run-workflow":
        return cmd_run_workflow(args)
    if args.command == "run-benchmark":
        return cmd_run_benchmark(args)
    if args.command == "run-comparison":
        return cmd_run_comparison(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
