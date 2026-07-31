# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Persistence and interactive visualization for quantization sensitivity results.

The sensitivity analyses in :mod:`aimet_onnx.analysis.sensitivity` return a plain
``{name: score}`` dict, ordered most-sensitive-first. This module renders that
dict as an interactive HTML chart and serializes it to / from JSON:

    * :func:`save_sensitivity_plot` -- Bokeh scatter/line chart with an
      interactive score-threshold filter and name-pattern highlight toggles.
    * :func:`save_sensitivity_results` / :func:`load_sensitivity_results` --
      ranked JSON round-trip.

Both take the dict returned by the analysis functions plus the
:class:`~aimet_onnx.analysis.SensitivityMetric` used to produce it (so the
ranking direction is preserved).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from aimet_onnx.analysis.sensitivity import SensitivityMetric
from aimet_onnx.common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# Default name substrings highlighted by the plot's toggle buttons. These match
# common transformer projection weights; override via ``highlight_patterns``.
_DEFAULT_HIGHLIGHT_PATTERNS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _check_html_path(path: str):
    """Validate an output path: existing directory and a ``.html`` suffix."""
    directory = os.path.dirname(path)
    if directory != "" and not os.path.exists(directory):
        raise NotADirectoryError(f"'{directory}' is not a directory.")
    if not path.endswith(".html"):
        raise ValueError("'save_path' must end with '.html'.")


def _ranked_entries(scores: Dict[str, float], metric: SensitivityMetric):
    """Return ``[(rank, name, score)]`` ordered most-sensitive-first per metric."""
    ordered = sorted(
        scores.items(),
        key=lambda kv: metric.sensitivity_score(kv[1]),
        reverse=True,
    )
    return [(rank, name, score) for rank, (name, score) in enumerate(ordered, start=1)]


def save_sensitivity_results(
    scores: Dict[str, float],
    save_path: str = "./sensitivity_results.json",
) -> None:
    """Save sensitivity ``scores`` to JSON, preserving their order.

    ``scores`` from the analysis functions is already ranked most-sensitive-first;
    that order is preserved in the ``ranking`` list. Inverse of
    :func:`load_sensitivity_results`.

    :param scores: ``{name: score}`` dict from an analysis function.
    :param save_path: Output JSON path.
    """
    payload = {
        "ranking": [
            {"rank": rank, "name": name, "score": score}
            for rank, (name, score) in enumerate(scores.items(), start=1)
        ],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved ranked sensitivity JSON to %s", save_path)


def load_sensitivity_results(input_path: str) -> Dict[str, float]:
    """Load a sensitivity JSON written by :func:`save_sensitivity_results`.

    :param input_path: Path to the JSON file.
    :return: ``{name: score}`` dict, preserving the file's ranked order.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {entry["name"]: entry["score"] for entry in payload["ranking"]}


def save_sensitivity_plot(
    scores: Dict[str, float],
    metric: SensitivityMetric,
    save_path: str = "./sensitivity_plot.html",
    highlight_patterns: Optional[List[str]] = None,
) -> None:
    """Render an interactive sensitivity chart and save it as standalone HTML.

    Points are plotted along the x-axis in the order of ``scores`` (so passing a
    dict in topological/layer order plots them that way); the y-axis is the
    metric score. The sensitivity rank (most-sensitive-first) is still shown in
    the hover tooltip and table. Interactive features:

        * A numeric threshold input; entries scoring below it are listed in a
          table and a guide line is drawn.
        * Toggle buttons (one per ``highlight_patterns`` entry) that highlight
          points whose name contains the pattern and add them to the table.

    :param scores: ``{name: score}`` dict from an analysis function. Iteration
        order of this dict sets the x-axis order of the plotted points.
    :param metric: The :class:`SensitivityMetric` used to produce ``scores``.
    :param save_path: Output HTML path (must end with ``.html``).
    :param highlight_patterns: Name substrings for the highlight toggles. If
        omitted, defaults to common transformer projection weight names.
    """
    # Imported lazily so the analysis package is usable without bokeh installed.
    from bokeh.layouts import column, row
    from bokeh.models import (
        ColumnDataSource,
        CustomJS,
        DataTable,
        Div,
        HoverTool,
        NumericInput,
        NumberFormatter,
        Span,
        TableColumn,
        Toggle,
    )
    from bokeh.plotting import figure, output_file, save

    _check_html_path(save_path)

    if highlight_patterns is None:
        highlight_patterns = list(_DEFAULT_HIGHLIGHT_PATTERNS)

    if not scores:
        raise ValueError("scores is empty; nothing to plot.")

    # Plot points in the order given by ``scores`` (e.g. topological/layer
    # order), but keep the sensitivity rank for the tooltip and table.
    rank_by_name = {name: rank for rank, name, _ in _ranked_entries(scores, metric)}
    names = list(scores.keys())
    score_vals = list(scores.values())
    indices = list(range(len(names)))
    ranks = [rank_by_name[name] for name in names]

    n = len(names)
    base_color = "#4a8cc7"
    source = ColumnDataSource(
        data=dict(
            index=indices,
            rank=ranks,
            name=names,
            score=score_vals,
            color=[base_color] * n,
            size=[5] * n,
            line_color=["#ffffff"] * n,
            line_width=[0] * n,
        )
    )

    plot = figure(
        height=450,
        width=1000,
        title=f"Quantization Sensitivity ({metric.name})",
        toolbar_location="above",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    plot.xgrid.grid_line_alpha = 0.3
    plot.ygrid.grid_line_alpha = 0.3
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None

    plot.line(
        x="index",
        y="score",
        source=source,
        color=base_color,
        line_width=1.5,
        line_alpha=0.7,
    )
    plot.scatter(
        x="index",
        y="score",
        source=source,
        color="color",
        size="size",
        line_color="line_color",
        line_width="line_width",
    )
    plot.xaxis.axis_label = "Unit index (analysis order)"
    plot.yaxis.axis_label = metric.name

    threshold_span = Span(
        location=0,
        dimension="width",
        line_color="#d62728",
        line_dash="dashed",
        line_alpha=0.6,
        line_width=1.5,
        visible=False,
    )
    plot.add_layout(threshold_span)

    plot.add_tools(
        HoverTool(
            tooltips=[
                ("Rank", "@rank"),
                ("Name", "@name"),
                (metric.name, "@score{0.0000}"),
            ],
            mode="mouse",
        )
    )

    threshold_input = NumericInput(
        title=f"Show entries with {metric.name} below:",
        value=None,
        mode="float",
        width=300,
        placeholder="e.g. 40.0",
    )
    filtered_source = ColumnDataSource(data={key: [] for key in source.data})

    table_columns = [
        TableColumn(field="rank", title="Rank", width=60),
        TableColumn(field="name", title="Name", width=520),
        TableColumn(
            field="score",
            title=metric.name,
            formatter=NumberFormatter(format="0.0000"),
            width=140,
        ),
    ]
    data_table = DataTable(
        source=filtered_source,
        columns=table_columns,
        width=1000,
        height=320,
        index_position=None,
        autosize_mode="none",
    )

    toggles = [
        Toggle(label=pattern, button_type="default", width=100)
        for pattern in highlight_patterns
    ]

    js_path = (
        Path(__file__).parent / "sensitivity_plot_JS_code" / "sensitivity_filter.js"
    )
    filter_js = js_path.read_text("utf8")

    match_count_div = Div(
        text="<span style='color:#888;font-size:12px'>No filter active.</span>",
        width=400,
    )

    filter_callback = CustomJS(
        args=dict(
            source=source,
            filtered_source=filtered_source,
            threshold_input=threshold_input,
            threshold_span=threshold_span,
            toggles=toggles,
            patterns=highlight_patterns,
            match_count_div=match_count_div,
        ),
        code=filter_js,
    )
    threshold_input.js_on_change("value", filter_callback)
    for toggle in toggles:
        toggle.js_on_change("active", filter_callback)

    header = Div(
        text=(
            "<h3 style='margin:0'>Units below threshold</h3>"
            "<div style='color:#555;font-size:12px'>"
            "Enter a value and/or toggle pattern buttons to highlight matching "
            "points; the table lists entries passing all active filters, "
            "sorted lowest-first.</div>"
        )
    )

    controls = column(row(threshold_input), row(*toggles))
    layout = column(plot, controls, row(header, match_count_div), data_table)

    output_file(save_path, title=f"Quantization Sensitivity ({metric.name})")
    save(layout, save_path)
    logger.info("Saved interactive sensitivity plot to %s", save_path)
