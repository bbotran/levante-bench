#!/usr/bin/env python3
"""Compare model accuracy vs random-guess baseline by math problem type."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze per-type math accuracy vs guessing.")
    p.add_argument("--prompts-jsonl", type=Path, required=True, help="Prompt JSONL path")
    p.add_argument("--preds-jsonl", type=Path, required=True, help="Predictions JSONL path")
    p.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    p.add_argument("--output-png", type=Path, default=None, help="Optional bar chart PNG path")
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    prompts = _read_jsonl(args.prompts_jsonl)
    preds = _read_jsonl(args.preds_jsonl)
    if len(prompts) != len(preds):
        raise ValueError(f"Row count mismatch: prompts={len(prompts)} preds={len(preds)}")

    agg = defaultdict(lambda: {"n": 0, "correct": 0, "chance_sum": 0.0, "parsed": 0})
    for p, r in zip(prompts, preds):
        t = str(p.get("trial_type") or "UNKNOWN")
        n_opts = max(len(p.get("options") or []), 1)
        chance = 1.0 / n_opts
        a = agg[t]
        a["n"] += 1
        a["correct"] += 1 if r.get("correct") else 0
        a["chance_sum"] += chance
        a["parsed"] += 1 if r.get("pred_letter") is not None else 0

    rows = []
    for t, a in agg.items():
        n = a["n"]
        acc = a["correct"] / n if n else 0.0
        guess = a["chance_sum"] / n if n else 0.0
        rows.append(
            {
                "trial_type": t,
                "n": n,
                "accuracy": acc,
                "guess_baseline": guess,
                "lift_vs_guess": acc - guess,
                "parse_rate": a["parsed"] / n if n else 0.0,
            }
        )

    rows.sort(key=lambda x: (-x["accuracy"], -x["n"], x["trial_type"]))
    n_total = sum(r["n"] for r in rows)
    c_total = sum(int(round(r["accuracy"] * r["n"])) for r in rows)
    chance_total = sum(r["guess_baseline"] * r["n"] for r in rows)
    parsed_total = sum(r["parse_rate"] * r["n"] for r in rows)
    rows.append(
        {
            "trial_type": "OVERALL",
            "n": n_total,
            "accuracy": c_total / n_total if n_total else 0.0,
            "guess_baseline": chance_total / n_total if n_total else 0.0,
            "lift_vs_guess": (c_total - chance_total) / n_total if n_total else 0.0,
            "parse_rate": parsed_total / n_total if n_total else 0.0,
        }
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial_type",
                "n",
                "accuracy",
                "guess_baseline",
                "lift_vs_guess",
                "parse_rate",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.output_csv}")

    if args.output_png is not None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipped PNG generation")
            return
        plot_rows = [r for r in rows if r["trial_type"] != "OVERALL"]
        types = [r["trial_type"] for r in plot_rows]
        acc = [r["accuracy"] for r in plot_rows]
        guess = [r["guess_baseline"] for r in plot_rows]
        x = list(range(len(types)))

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar([i - 0.2 for i in x], acc, width=0.4, label="Model accuracy")
        ax.bar([i + 0.2 for i in x], guess, width=0.4, label="Guess baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title("Math accuracy by problem type")
        ax.legend()
        fig.tight_layout()
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=150)
        print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
