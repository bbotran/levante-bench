"""Write evaluation outputs: per-task CSV and cross-task summary CSV."""

from pathlib import Path


def write_task_csv(output_dir: Path, task_id: str, results: list[dict]) -> Path:
    """Write per-task detailed results CSV.

    Columns: trial_id, item_uid, generated_text, predicted_label, correct_label, is_correct
    Written to: <output_dir>/<task_id>.csv
    """
    pass


def write_summary_csv(output_dir: Path, task_accuracies: dict[str, float]) -> Path:
    """Write cross-task summary CSV.

    Columns: task_id, accuracy
    Written to: <output_dir>/summary.csv
    """
    pass
