#!/usr/bin/env python3
"""
Convert EGMA math corpus rows into SmolVLMv2-ready prompt records.

Output format is JSONL, one record per item:
{
  "item_uid": "...",
  "prompt_text": "...",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "..."}]}],
  "options": ["...", "...", "...", "..."],
  "gold_index": 2,
  "gold_letter": "C",
  ...
}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SmolVLMv2 prompts from EGMA math corpus CSV.")
    p.add_argument("--corpus-csv", type=Path, required=True, help="Path to test-combined-math-cat.csv")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice rows (default: only test_response rows)",
    )
    p.add_argument(
        "--include-instructions",
        action="store_true",
        help="Include instructions rows (default: exclude)",
    )
    p.add_argument(
        "--include-audio-dependent",
        action="store_true",
        help="Include rows likely requiring audio (default: exclude)",
    )
    p.add_argument(
        "--shuffle-options",
        action="store_true",
        default=True,
        help="Shuffle answer choices while preserving gold index (default: enabled)",
    )
    p.add_argument(
        "--no-shuffle-options",
        dest="shuffle_options",
        action="store_false",
        help="Disable option shuffling (legacy behavior).",
    )
    p.add_argument(
        "--numberline-hint",
        choices=["none", "coarse", "exact"],
        default="coarse",
        help=(
            "Optional extra hint for Number Line prompts derived from item_uid. "
            "'coarse' adds approximate location (left/middle/right), "
            "'exact' adds the marked number."
        ),
    )
    p.add_argument(
        "--numberline-graphics-dir",
        type=Path,
        default=None,
        help="Optional directory of numberline images to attach for numberline items.",
    )
    p.add_argument(
        "--numberline-instruction-style",
        choices=["minimal", "stepwise"],
        default="stepwise",
        help=(
            "How explicitly to instruct numberline interpretation when image is attached. "
            "'minimal' gives compact guidance; 'stepwise' gives a short procedure."
        ),
    )
    p.add_argument(
        "--numbers-as-words",
        action="store_true",
        help="Convert standalone integer numerals in prompts/options to words.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed when --shuffle-options is used")
    return p.parse_args()


def _split_alternatives(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _is_audio_dependent(row: dict[str, str]) -> bool:
    prompt = (row.get("prompt") or "").strip().lower()
    trial_type = (row.get("trial_type") or "").strip().lower()
    audio_file = (row.get("audio_file") or "").strip()
    return bool(audio_file) and ("hear" in prompt or "listening" in prompt or "identification" in trial_type)


def _should_include_row(
    row: dict[str, str],
    include_practice: bool,
    include_instructions: bool,
    include_audio_dependent: bool,
) -> bool:
    stage = (row.get("assessment_stage") or "").strip().lower()
    answer = (row.get("answer") or "").strip()
    item_uid = (row.get("item_uid") or "").strip()

    if not item_uid:
        return False
    if not include_instructions and stage == "instructions":
        return False
    if not include_practice and stage and stage != "test_response":
        return False
    if not answer:
        return False
    if not include_audio_dependent and _is_audio_dependent(row):
        return False
    return True


def _numberline_hint(item_uid: str, numberline_hint: str) -> str | None:
    if numberline_hint == "none":
        return None
    # Example item_uid: math_line_4_10  -> marked=4, upper=10
    m = re.search(r"math_line_(\d+)_([0-9]+)$", item_uid)
    if not m:
        return None
    marked = int(m.group(1))
    upper = int(m.group(2))
    if numberline_hint == "exact":
        return f"The marked point is at {marked} on the number line from 0 to {upper}."
    if upper <= 0:
        return None
    ratio = marked / upper
    if ratio < 0.33:
        loc = "left side"
    elif ratio < 0.67:
        loc = "middle"
    else:
        loc = "right side"
    return f"The marked point is near the {loc} of the number line (0 to {upper})."


def _is_numberline_row(row: dict[str, str]) -> bool:
    trial_type = (row.get("trial_type") or "").strip().lower()
    return trial_type.startswith("number line")


def _pair_from_text(value: str) -> tuple[int, int] | None:
    nums = re.findall(r"\d+", value or "")
    if len(nums) < 2:
        return None
    return (int(nums[-2]), int(nums[-1]))


def _build_numberline_index(graphics_dir: Path | None) -> dict[tuple[int, int], Path]:
    idx: dict[tuple[int, int], Path] = {}
    if graphics_dir is None or not graphics_dir.exists():
        return idx
    for p in graphics_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        pair = _pair_from_text(p.stem)
        if pair is not None and pair not in idx:
            idx[pair] = p
    return idx


def _resolve_numberline_image(item_uid: str, index: dict[tuple[int, int], Path]) -> Path | None:
    pair = _pair_from_text(item_uid)
    if pair is None:
        return None
    return index.get(pair)


def _numberline_interpretation_text(style: str) -> str:
    if style == "minimal":
        return (
            "Interpretation: read the numeric labels at the left and right endpoints in the image to determine the scale. "
            "The number line runs left-to-right, equal spacing means equal numeric intervals, and the marked point is the target value."
        )
    return (
        "Interpretation procedure: "
        "(1) Read the left and right endpoint labels in the image to determine the scale. "
        "(2) Use equal spacing between ticks to infer each interval value. "
        "(3) Locate the marked point and estimate its numeric value. "
        "(4) Select the option whose value is closest to that point."
    )


def _numbers_to_words(text: str) -> str:
    if not text:
        return text
    try:
        from num2words import num2words
    except Exception:
        return text

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            return str(num2words(int(token)))
        except Exception:
            return token

    return re.sub(r"\b\d+\b", repl, text)


def _build_prompt_text(
    row: dict[str, str],
    options: list[str],
    numberline_hint: str,
    has_numberline_image: bool,
    numberline_instruction_style: str,
    numbers_as_words: bool,
) -> str:
    trial_type = (row.get("trial_type") or "").strip()
    stem = (row.get("prompt") or "").strip()
    item = (row.get("item") or "").strip()
    item_uid = (row.get("item_uid") or "").strip()
    if numbers_as_words:
        stem = _numbers_to_words(stem)
        item = _numbers_to_words(item)
        options = [_numbers_to_words(opt) for opt in options]

    lines = [
        "Solve this multiple-choice math problem.",
        "Return only the option letter (A, B, C, ...).",
    ]
    if trial_type:
        lines.append(f"Category: {trial_type}")
    if stem:
        lines.append(f"Instruction: {stem}")
    if item:
        lines.append(f"Problem: {item}")
    if _is_numberline_row(row):
        lines.append(_numberline_interpretation_text(numberline_instruction_style))
        if has_numberline_image:
            lines.append("Use the attached numberline image to answer.")
        else:
            hint = _numberline_hint(item_uid, numberline_hint)
            if hint is not None:
                lines.append(f"Hint: {hint}")
    lines.append("Options:")
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return "\n".join(lines)


def _record_from_row(
    row: dict[str, str],
    shuffle_options: bool,
    rng: random.Random,
    numberline_hint: str,
    numberline_index: dict[tuple[int, int], Path],
    numberline_instruction_style: str,
    numbers_as_words: bool,
) -> dict[str, object] | None:
    answer = (row.get("answer") or "").strip()
    distractors = _split_alternatives((row.get("response_alternatives") or "").strip())
    if numbers_as_words:
        answer = _numbers_to_words(answer)
        distractors = [_numbers_to_words(d) for d in distractors]
    options = [answer] + [d for d in distractors if d != answer]
    if len(options) < 2:
        return None

    # Keep options unique but preserve order.
    deduped: list[str] = []
    seen: set[str] = set()
    for o in options:
        if o not in seen:
            deduped.append(o)
            seen.add(o)
    options = deduped

    if shuffle_options:
        rng.shuffle(options)
    gold_index = options.index(answer)
    item_uid = (row.get("item_uid") or "").strip()
    numberline_image = _resolve_numberline_image(item_uid, numberline_index) if _is_numberline_row(row) else None
    prompt_text = _build_prompt_text(
        row,
        options,
        numberline_hint=numberline_hint,
        has_numberline_image=numberline_image is not None,
        numberline_instruction_style=numberline_instruction_style,
        numbers_as_words=numbers_as_words,
    )

    return {
        "item_uid": item_uid,
        "task": (row.get("task") or "").strip(),
        "trial_type": (row.get("trial_type") or "").strip(),
        "assessment_stage": (row.get("assessment_stage") or "").strip(),
        "difficulty": (row.get("difficulty") or "").strip(),
        "audio_file": (row.get("audio_file") or "").strip(),
        "options": options,
        "gold_answer": answer,
        "gold_index": gold_index,
        "gold_letter": LETTERS[gold_index],
        "prompt_text": prompt_text,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
        "image_paths": [str(numberline_image)] if numberline_image is not None else [],
    }


def run(args: argparse.Namespace) -> tuple[int, int]:
    rng = random.Random(args.seed)
    numberline_index = _build_numberline_index(args.numberline_graphics_dir)
    kept = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.corpus_csv, newline="", encoding="utf-8") as f_in, open(
        args.output, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if not _should_include_row(
                row=row,
                include_practice=args.include_practice,
                include_instructions=args.include_instructions,
                include_audio_dependent=args.include_audio_dependent,
            ):
                skipped += 1
                continue
            rec = _record_from_row(
                row,
                shuffle_options=args.shuffle_options,
                rng=rng,
                numberline_hint=args.numberline_hint,
                numberline_index=numberline_index,
                numberline_instruction_style=args.numberline_instruction_style,
                numbers_as_words=args.numbers_as_words,
            )
            if rec is None:
                skipped += 1
                continue
            f_out.write(json.dumps(rec, ensure_ascii=True) + "\n")
            kept += 1
    return kept, skipped


def main() -> None:
    args = parse_args()
    kept, skipped = run(args)
    print(f"Wrote {args.output} ({kept} prompts, {skipped} skipped)")


if __name__ == "__main__":
    main()
