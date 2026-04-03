"""Unit tests for answer parsing logic in VLMModel."""

from __future__ import annotations

import pytest

from levante_bench.models.base import VLMModel


@pytest.mark.parametrize(
    "text,labels,expected",
    [
        ('{"answer": "B", "reason": "because"}', ["A", "B", "C", "D"], "B"),
        ('{"answer": "b"}', ["A", "B", "C", "D"], "B"),
        ('{"answer":"C","reason":"truncated"', ["A", "B", "C", "D"], "C"),
        ("The correct answer is C.", ["A", "B", "C", "D"], "C"),
        ("The correct option is D.", ["A", "B", "C", "D"], "D"),
        ("Answer: A", ["A", "B", "C", "D"], "A"),
        ("my answer=B", ["A", "B", "C", "D"], "B"),
        ("B.", ["A", "B", "C", "D"], "B"),
        ("B) The largest one", ["A", "B", "C", "D"], "B"),
        ("None of the above", ["A", "B", "C", "D"], None),
        ("", ["A", "B", "C", "D"], None),
    ],
)
def test_parse_answer_branches(text: str, labels: list[str], expected: str | None) -> None:
    model = VLMModel(model_name="dummy")
    label, _ = model.parse_answer(text, labels)
    assert label == expected


@pytest.mark.parametrize(
    "text,strict_json,slider_mode,expected",
    [
        ("0.75", False, False, 0.75),
        ("-1.5", False, False, -1.5),
        ('{"answer": 2.5, "reason":"ok"}', False, False, 2.5),
        ('{"answer":"-3.0"}', False, False, -3.0),
        ('noise {"answer":"4.25"} trailing', False, False, 4.25),
        ("score is 7.0 now", False, False, 7.0),
        ('{"answer":{"value": 1.2}}', True, False, None),
        ('{"answer":"1.2"}', True, False, 1.2),
        ("1.5", False, True, 1.5),
        ('{"answer":"0.25"}', False, True, 0.25),
        ("answer is 0.40", False, True, 0.40),
        ("nonsense", False, True, None),
    ],
)
def test_parse_numeric_answer_modes(
    text: str,
    strict_json: bool,
    slider_mode: bool,
    expected: float | None,
) -> None:
    model = VLMModel(model_name="dummy")
    value, _ = model.parse_numeric_answer(
        text, strict_json=strict_json, slider_mode=slider_mode
    )
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)
