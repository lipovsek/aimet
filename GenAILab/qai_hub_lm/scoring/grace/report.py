# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Turn raw grades into the Grace report payload.

Shared verbatim with consumers outside this repo: no host imports, and no
rendering -- console output, dashboards and CSV stay on the host side.
"""

from __future__ import annotations

from .grace import (
    default_categories_by_idx,
    default_categories_by_prompt,
)
from .grader import MAX_POINTS, GradeResult, GradeSummary


def resolve_categories(items: list[dict]) -> list[str | None]:
    """Category per item, backfilled for response files that record none.

    Device runs write only ``{idx, prompt, output}``, so the category is looked
    up in the built-in prompt set by prompt text, then by ``idx``. None only for
    a prompt that is not in the built-in set at all.
    """
    by_prompt = default_categories_by_prompt()
    by_idx = default_categories_by_idx()
    return [
        item.get("category")
        or by_prompt.get(str(item.get("prompt", "")).strip())
        or by_idx.get(item.get("idx", -1))
        for item in items
    ]


def category_scores(
    categories: list[str | None], results: list[GradeResult]
) -> dict[str, tuple[float, int, int]]:
    """Per-category (score_pct, points, num_scored), in first-seen order.

    Mirrors the overall score: an item the grader failed to rate scores 0 and
    stays in its category's denominator.
    """
    points: dict[str, int] = {}
    scored: dict[str, int] = {}
    for category, result in zip(categories, results, strict=True):
        if category is None:
            continue
        points[category] = points.get(category, 0) + result.points
        scored[category] = scored.get(category, 0) + 1
    return {
        name: (
            100.0 * points[name] / (MAX_POINTS * scored[name]),
            points[name],
            scored[name],
        )
        for name in points
    }


def build_summary(
    items: list[dict],
    graded: GradeSummary,
    *,
    metric_name: str,
    grader_model: str,
    input_file: str,
) -> dict:
    """The full grader report, as written to ``grader_summary.json``.

    ``score_pct`` is 0-100 on every host; rescale downstream if you report
    otherwise. ``metric_name`` is the host's label -- GenAI Lab files these
    numbers under ``Grace``, AI Hub Models under ``Grace2``. Keyword-only
    because the three label strings are easy to transpose.
    """
    categories = resolve_categories(items)
    per_category = category_scores(categories, graded.results)
    return {
        "input_file": input_file,
        "metric": metric_name,
        "grader_model": grader_model,
        "num_items": len(items),
        "score_pct": graded.score_pct,
        "total_points": graded.total_points,
        "max_points": graded.max_points,
        "num_unparsed": graded.num_unparsed,
        "num_forced": graded.num_forced,
        "summary_items": graded.summary_items,
        "category_scores": {
            name: {"score_pct": pct, "points": pts, "num_scored": num}
            for name, (pct, pts, num) in per_category.items()
        },
        "items": [
            {
                "idx": item["idx"],
                "category": category,
                "points": result.points,
                "skipped": result.skipped,
                "parsed": result.parsed,
                "forced": result.forced,
                "rationale": result.rationale,
            }
            for item, category, result in zip(
                items, categories, graded.results, strict=True
            )
        ],
    }


def detail_items(responses: list[dict], graded_items: list[dict]) -> list[dict]:
    """One record per prompt: the response, and the grade it was given.

    Joins the two halves ``responses.json`` and ``grader_summary.json`` hold
    separately, so a score in the stats file can be traced back to the text
    behind it without re-running the generation and grading passes.
    """
    return [
        {**graded, "prompt": response["prompt"], "output": response["output"]}
        for response, graded in zip(responses, graded_items, strict=True)
    ]
