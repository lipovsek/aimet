# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Tool to visualize min and max activation ranges of quantized modules in a given ONNX model"""

import os
from pathlib import Path
from bokeh.events import DocumentReady, Reset
from bokeh.layouts import row, column
from bokeh.models import (
    ColumnDataSource,
    TextInput,
    CustomJS,
    Range1d,
    HoverTool,
    CustomJSHover,
    Div,
    BooleanFilter,
    CDSView,
    Spacer,
    DataTable,
    StringFormatter,
    ScientificFormatter,
    TableColumn,
    Tooltip,
    Select,
)
from bokeh.models.tools import ResetTool
from bokeh.models.dom import HTML
from bokeh.plotting import figure, save, curdoc, output_file
from aimet_onnx.quantsim import QuantizationSimModel


def visualize_stats(
    sim: QuantizationSimModel,
    save_path: str = "./quant_stats_visualization.html",
) -> None:
    """Produces two interactive htmls to view the activation and weight encoding ranges from a calibrated QuantSim model.

    .. note::

        The QuantizationSimModel input is expected to have been calibrated before using this function. Stats will only
        be plotted for quantizers containing calibration statistics.

    Creates interactive visualizations of min and max ranges of all quantized activations and weights in the input
    QuantSim object. The features include:

        - Adjustable threshold values to flag layers whose min or max values exceed the set thresholds
        - Tables containing names and ranges for layers exceeding threshold values

    Two files are saved, derived from ``save_path`` by inserting ``_activations`` and ``_weights`` before the
    extension. For example, the default ``./quant_stats_visualization.html`` produces
    ``./quant_stats_visualization_activations.html`` and ``./quant_stats_visualization_weights.html``.

    Example:

        >>> sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf)
        >>> with compute_encodings(sim):
        ...     for input_data in data_loader:
        ...         sim.session.run(None, input_data)
        ...
        >>> visualize_stats(sim, save_path="./quant_stats_visualization.html")

    :param sim: Calibrated QuantizationSimModel
    :param save_path: Base path for saving the visualizations. Default is "./quant_stats_visualization.html"
    """
    if not isinstance(sim, QuantizationSimModel):
        raise TypeError(
            f"Expected type 'aimet_onnx.quantsim.QuantizationSimModel', got '{type(sim)}'."
        )

    _check_path(save_path)

    base, ext = os.path.splitext(save_path)
    activation_save_path = f"{base}_activations{ext}"
    weight_save_path = f"{base}_weights{ext}"

    activation_stats = _collect_activation_stats(sim)
    weight_stats = _collect_weight_stats(sim)

    if not activation_stats and not weight_stats:
        raise RuntimeError(
            "No stats found to plot. Either there were no quantized activations or weights, "
            "or calibration was not performed before calling this function."
        )

    if activation_stats:
        stats_dict = _build_stats_dict(activation_stats)
        QuantStatsVisualizer(stats_dict, entity_label="Activation").export_plot_as_html(
            activation_save_path
        )

    if weight_stats:
        stats_dict = _build_stats_dict(weight_stats)
        QuantStatsVisualizer(stats_dict, entity_label="Weight").export_plot_as_html(
            weight_save_path
        )


def _check_path(path: str):
    """Function for sanity check on the given path"""
    path_to_directory = os.path.dirname(path)
    if path_to_directory != "" and not os.path.exists(path_to_directory):
        raise NotADirectoryError(f"'{path_to_directory}' is not a directory.")
    if not path.endswith(".html"):
        raise ValueError("'save_path' must end with '.html'.")


def _collect_activation_stats(sim: QuantizationSimModel):
    """Collect min/max stats from activation quantizers."""
    stats_list = []

    for activation_name in sim.activation_names:
        quantizer = sim.qc_quantize_op_dict.get(activation_name)
        if quantizer is None or not quantizer.enabled:
            continue

        encodings = quantizer.get_encodings()
        if encodings is None or len(encodings) == 0:
            continue

        # For per-tensor quantization, there's one encoding
        # For per-channel, we take the global min/max across all channels
        min_val = float("inf")
        max_val = float("-inf")
        for enc in encodings:
            if enc.min < min_val:
                min_val = enc.min
            if enc.max > max_val:
                max_val = enc.max

        if min_val < float("inf") and max_val > float("-inf"):
            stats_list.append(
                {
                    "name": activation_name,
                    "min": min_val,
                    "max": max_val,
                }
            )

    return stats_list


def _collect_weight_stats(sim: QuantizationSimModel):
    """Collect min/max stats from weight quantizers."""
    stats_list = []

    for param_name in sim.param_names:
        quantizer = sim.qc_quantize_op_dict.get(param_name)
        if quantizer is None or not quantizer.enabled:
            continue

        encodings = quantizer.get_encodings()
        if encodings is None or len(encodings) == 0:
            continue

        # For per-tensor quantization, there's one encoding
        # For per-channel, we take the global min/max across all channels
        min_val = float("inf")
        max_val = float("-inf")
        for enc in encodings:
            if enc.min < min_val:
                min_val = enc.min
            if enc.max > max_val:
                max_val = enc.max

        if min_val < float("inf") and max_val > float("-inf"):
            stats_list.append(
                {
                    "name": param_name,
                    "min": min_val,
                    "max": max_val,
                }
            )

    return stats_list


def _build_stats_dict(stats_list):
    """Build dictionary format expected by the visualizer."""
    stats_dict = {
        "idx": list(range(len(stats_list))),
        "name": [s["name"] for s in stats_list],
        "min": [s["min"] for s in stats_list],
        "max": [s["max"] for s in stats_list],
    }
    return stats_dict


class DataSources:
    """
    Class to hold the Bokeh ColumnDataSource objects needed in the visualization.
    """

    def __init__(self, stats_dict: dict, plot: figure, default_values: dict):
        self.data_source = ColumnDataSource(
            {
                "idx": stats_dict["idx"],
                "namelist": stats_dict["name"],
                "minlist": stats_dict["min"],
                "maxlist": stats_dict["max"],
                "marker_yminlist": [default_values["default_ymin"]]
                * len(stats_dict["idx"]),
                "marker_ymaxlist": [default_values["default_ymax"]]
                * len(stats_dict["idx"]),
                "selected": [False] * len(stats_dict["idx"]),
            }
        )

        self.default_values_source = ColumnDataSource(
            {
                "default_ymax": [default_values["default_ymax"]],
                "default_ymin": [default_values["default_ymin"]],
                "default_maxclip": [default_values["default_maxclip"]],
                "default_minclip": [default_values["default_minclip"]],
                "default_xmax": [default_values["default_xmax"]],
                "default_xmin": [default_values["default_xmin"]],
            }
        )

        self.limits_source = ColumnDataSource(
            {
                "ymax": [default_values["default_ymax"]],
                "ymin": [default_values["default_ymin"]],
                "xmin": [plot.x_range.start],
                "xmax": [plot.x_range.end],
                "minclip": [default_values["default_minclip"]],
                "maxclip": [default_values["default_maxclip"]],
            }
        )

        self.table_data_source = ColumnDataSource(
            {
                "idx": [],
                "namelist": [],
                "minlist": [],
                "maxlist": [],
            }
        )

        self.selected_data_source = ColumnDataSource(
            {
                "idx": [],
                "namelist": [],
                "floor": [],
                "ceil": [],
                "minlist": [],
                "maxlist": [],
            }
        )


class TableFilters:
    """
    Class for holding data filters.
    """

    def __init__(self, data_sources: DataSources):
        self.name_filter = BooleanFilter()
        self.name_filter.booleans = [
            True for _ in range(len(data_sources.data_source.data["idx"]))
        ]
        self.min_thresh_filter = BooleanFilter()
        self.min_thresh_filter.booleans = [
            True for _ in range(len(data_sources.data_source.data["idx"]))
        ]
        self.max_thresh_filter = BooleanFilter()
        self.max_thresh_filter.booleans = [
            True for _ in range(len(data_sources.data_source.data["idx"]))
        ]


class TableViews:
    """
    Class for holding views of the data sources.
    """

    def __init__(self, tablefilters: TableFilters):
        self.min_thresh_view = CDSView(filter=tablefilters.min_thresh_filter)
        self.max_thresh_view = CDSView(filter=tablefilters.max_thresh_filter)


class TableObjects:
    """
    Class for holding various objects related to the table elements in the visualization.
    """

    def __init__(self, datasources: DataSources):
        self.filters = TableFilters(datasources)
        self.views = TableViews(self.filters)

        columns = [
            TableColumn(
                field="idx",
                title="Layer Index",
                width=QuantStatsVisualizer.table_column_widths["Layer Index"],
            ),
            TableColumn(
                field="namelist",
                title="Tensor Name",
                formatter=StringFormatter(font_style="bold"),
                width=QuantStatsVisualizer.table_column_widths["Activation Name"],
            ),
            TableColumn(
                field="minlist",
                title="Min Value",
                formatter=ScientificFormatter(precision=3),
                width=QuantStatsVisualizer.table_column_widths["Min Value"],
            ),
            TableColumn(
                field="maxlist",
                title="Max Value",
                formatter=ScientificFormatter(precision=3),
                width=QuantStatsVisualizer.table_column_widths["Max Value"],
            ),
        ]

        self.data_table = DataTable(
            source=datasources.table_data_source,
            columns=columns,
            sortable=True,
            width=QuantStatsVisualizer.plot_dims["table_width"],
            selectable="checkbox",
            index_position=None,
        )


class InputWidgets:
    """
    Class to hold various input widgets.
    """

    def __init__(self, default_values: dict):
        self.ymin_input = TextInput(
            value=str(default_values["default_ymin"]),
            title="Enter lower display limit of the plot",
        )
        self.ymax_input = TextInput(
            value=str(default_values["default_ymax"]),
            title="Enter upper display limit of the plot",
        )
        self.minclip_input = TextInput(
            value=str(default_values["default_minclip"]),
            title="Enter lower threshold value",
        )
        self.maxclip_input = TextInput(
            value=str(default_values["default_maxclip"]),
            title="Enter upper threshold value",
        )

        self.name_input = TextInput(value="", title="Enter Name Filter")

        tooltip_table_mode = Tooltip(
            content=HTML("""
                <h3> Select Table View </h3>
                <p> Following table views are available </p>
                <ol>
                <li> <b> All: </b> All quantized activations </li>
                <li> <b> Min: </b> Activations with min value below lower threshold </li>
                <li> <b> Max: </b> Activations with max value above upper threshold </li>
                <li> <b> Min | Max: </b> Union of Min and Max </li>
                <li> <b> Min & Max: </b> Intersection of Min and Max </li>
                </ol>
            """),
            position="right",
        )
        self.table_view_select = Select(
            title="Select Table View",
            value="Min | Max",
            options=["All", "Min", "Max", "Min | Max", "Min & Max"],
            width=200,
            description=tooltip_table_mode,
        )


class CustomCallbacks:
    """
    Class to hold Custom JavaScript Callbacks for interactivity in the visualization.
    """

    def __init__(self):
        self.limit_change_callback = None
        self.reset_callback = None
        self.name_filter_callback = None
        self.select_table_view_callback = None
        self.table_selection_callback = None


class QuantStatsVisualizer:
    """
    Class for constructing the visualization with functionality to export the plot as HTML.

    :param stats_dict: Dictionary containing the tensor names, indices, and min/max statistics
    :param entity_label: Label for the quantized entity being plotted ("Activation" or "Weight"),
        used in plot titles, axis labels, and headings
    """

    # Class level constants
    plot_dims = {
        "plot_width": 700,
        "plot_height": 400,
        "table_width": 800,
    }
    initial_vals = {"default_ymin": -1e5, "default_ymax": 1e5}
    spacer_dims = {"sp1_width": 50, "sp1_height": 40}
    table_column_widths = {
        "Layer Index": 100,
        "Activation Name": 400,
        "Min Value": 100,
        "Max Value": 100,
    }

    def __init__(self, stats_dict: dict, entity_label: str = "Activation"):
        self.stats_dict = stats_dict
        self.entity_label = entity_label
        self.plot = figure(
            title=f"Min Max {entity_label} Ranges of quantized {entity_label.lower()}s",
            x_axis_label="Layer index",
            y_axis_label=f"{entity_label} Value",
            tools="pan,wheel_zoom,box_zoom",
        )
        self.default_values = {}

    def _add_plot_lines(self, datasources: DataSources):
        self.plot.segment(
            x0="xmin",
            x1="xmax",
            y0="ymin",
            y1="ymin",
            line_width=4,
            line_color="black",
            source=datasources.limits_source,
        )
        self.plot.segment(
            x0="xmin",
            x1="xmax",
            y0="ymax",
            y1="ymax",
            line_width=4,
            line_color="black",
            source=datasources.limits_source,
        )
        self.plot.segment(
            x0="xmin",
            x1="xmax",
            y0="minclip",
            y1="minclip",
            line_width=2,
            line_color="black",
            line_dash="dashed",
            source=datasources.limits_source,
        )
        self.plot.segment(
            x0="xmin",
            x1="xmax",
            y0="maxclip",
            y1="maxclip",
            line_width=2,
            line_color="black",
            line_dash="dashed",
            source=datasources.limits_source,
        )
        self.plot.line(
            "idx",
            "maxlist",
            source=datasources.data_source,
            legend_label="Max Activation",
            line_width=2,
            line_color="red",
        )
        self.plot.line(
            "idx",
            "minlist",
            source=datasources.data_source,
            legend_label="Min Activation",
            line_width=2,
            line_color="blue",
        )
        selections = self.plot.segment(
            x0="idx",
            x1="idx",
            y0="floor",
            y1="ceil",
            line_width=2,
            line_color="goldenrod",
            line_alpha=0.5,
            source=datasources.selected_data_source,
        )

        return selections

    def _add_min_max_markers(
        self, datasources: DataSources, tableobjects: TableObjects
    ):
        min_markers = self.plot.scatter(
            "idx",
            "marker_yminlist",
            source=datasources.data_source,
            size=10,
            marker="circle_x",
            color="orange",
            line_color="navy",
        )
        min_markers.view = tableobjects.views.min_thresh_view
        max_markers = self.plot.scatter(
            "idx",
            "marker_ymaxlist",
            source=datasources.data_source,
            size=10,
            marker="circle_x",
            color="orange",
            line_color="navy",
        )
        max_markers.view = tableobjects.views.max_thresh_view

        return min_markers, max_markers

    @staticmethod
    def _get_marker_hovertool(min_markers, max_markers):
        format_code = """
            if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e5) {
            return value.toExponential(3);
            } else {
            return value.toFixed(3);
            }
        """

        format_hover = CustomJSHover(code=format_code)

        marker_hover = HoverTool(
            renderers=[min_markers, max_markers],
            tooltips=[
                ("Layer Index", "@idx"),
                ("Name", "@namelist"),
                ("Max Activation", "@maxlist{custom}"),
                ("Min Activation", "@minlist{custom}"),
            ],
            formatters={
                "@minlist": format_hover,
                "@maxlist": format_hover,
            },
        )

        return marker_hover

    @staticmethod
    def _get_selection_hovertool(selections):
        format_code = """
            if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e5) {
            return value.toExponential(3);
            } else {
            return value.toFixed(3);
            }
        """

        format_hover = CustomJSHover(code=format_code)

        selection_hover = HoverTool(
            renderers=[selections],
            tooltips=[
                ("Layer Index", "@idx"),
                ("Name", "@namelist"),
                ("Max Activation", "@maxlist{custom}"),
                ("Min Activation", "@minlist{custom}"),
            ],
            formatters={
                "@minlist": format_hover,
                "@maxlist": format_hover,
            },
        )

        return selection_hover

    def _define_callbacks(self, datasources, tableobjects, inputwidgets):
        customcallbacks = CustomCallbacks()

        table_columns = ["idx", "namelist", "minlist", "maxlist"]
        selection_columns = ["idx", "namelist", "minlist", "maxlist"]

        customcallbacks.limit_change_callback = CustomJS(
            args={
                "limits_source": datasources.limits_source,
                "data_source": datasources.data_source,
                "table_data_source": datasources.table_data_source,
                "selected_data_source": datasources.selected_data_source,
                "ymax_input": inputwidgets.ymax_input,
                "ymin_input": inputwidgets.ymin_input,
                "maxclip_input": inputwidgets.maxclip_input,
                "minclip_input": inputwidgets.minclip_input,
                "plot": self.plot,
                "min_thresh_filter": tableobjects.filters.min_thresh_filter,
                "max_thresh_filter": tableobjects.filters.max_thresh_filter,
                "name_filter": tableobjects.filters.name_filter,
                "select": inputwidgets.table_view_select,
                "table_columns": table_columns,
            },
            code=(
                Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js"
            ).read_text("utf8")
            + (
                Path(__file__).parent
                / "quant_stats_visualization_JS_code/limit_change_callback.js"
            ).read_text("utf8"),
        )

        customcallbacks.reset_callback = CustomJS(
            args={
                "limits_source": datasources.limits_source,
                "data_source": datasources.data_source,
                "table_data_source": datasources.table_data_source,
                "selected_data_source": datasources.selected_data_source,
                "default_values_source": datasources.default_values_source,
                "ymax_input": inputwidgets.ymax_input,
                "ymin_input": inputwidgets.ymin_input,
                "maxclip_input": inputwidgets.maxclip_input,
                "minclip_input": inputwidgets.minclip_input,
                "select": inputwidgets.table_view_select,
                "name_input": inputwidgets.name_input,
                "plot": self.plot,
                "min_thresh_filter": tableobjects.filters.min_thresh_filter,
                "max_thresh_filter": tableobjects.filters.max_thresh_filter,
                "name_filter": tableobjects.filters.name_filter,
                "selection_columns": selection_columns,
                "table_columns": table_columns,
            },
            code=(
                Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js"
            ).read_text("utf8")
            + (
                Path(__file__).parent
                / "quant_stats_visualization_JS_code/reset_callback.js"
            ).read_text("utf8"),
        )

        customcallbacks.name_filter_callback = CustomJS(
            args={
                "data_source": datasources.data_source,
                "table_data_source": datasources.table_data_source,
                "limits_source": datasources.limits_source,
                "min_thresh_filter": tableobjects.filters.min_thresh_filter,
                "max_thresh_filter": tableobjects.filters.max_thresh_filter,
                "name_filter": tableobjects.filters.name_filter,
                "select": inputwidgets.table_view_select,
                "table_columns": table_columns,
                "table": tableobjects.data_table,
            },
            code=(
                Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js"
            ).read_text("utf8")
            + (
                Path(__file__).parent
                / "quant_stats_visualization_JS_code/name_filter_callback.js"
            ).read_text("utf8"),
        )

        customcallbacks.select_table_view_callback = CustomJS(
            args={
                "data_source": datasources.data_source,
                "table_data_source": datasources.table_data_source,
                "select": inputwidgets.table_view_select,
                "min_thresh_filter": tableobjects.filters.min_thresh_filter,
                "max_thresh_filter": tableobjects.filters.max_thresh_filter,
                "name_filter": tableobjects.filters.name_filter,
                "table": tableobjects.data_table,
                "table_columns": table_columns,
            },
            code=(
                Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js"
            ).read_text("utf8")
            + (
                Path(__file__).parent
                / "quant_stats_visualization_JS_code/select_table_view_callback.js"
            ).read_text("utf8"),
        )

        customcallbacks.table_selection_callback = CustomJS(
            args={
                "data_source": datasources.data_source,
                "table_data_source": datasources.table_data_source,
                "selected_data_source": datasources.selected_data_source,
                "limits_source": datasources.limits_source,
                "selection_columns": selection_columns,
            },
            code=(
                Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js"
            ).read_text("utf8")
            + (
                Path(__file__).parent
                / "quant_stats_visualization_JS_code/table_selection_callback.js"
            ).read_text("utf8"),
        )

        return customcallbacks

    def _attach_callbacks(self, datasources, inputwidgets, customcallbacks):
        self.plot.js_on_event(Reset, customcallbacks.reset_callback)
        inputwidgets.ymax_input.js_on_change(
            "value", customcallbacks.limit_change_callback
        )
        inputwidgets.ymin_input.js_on_change(
            "value", customcallbacks.limit_change_callback
        )
        inputwidgets.maxclip_input.js_on_change(
            "value", customcallbacks.limit_change_callback
        )
        inputwidgets.minclip_input.js_on_change(
            "value", customcallbacks.limit_change_callback
        )
        inputwidgets.name_input.js_on_change(
            "value", customcallbacks.name_filter_callback
        )
        inputwidgets.table_view_select.js_on_change(
            "value", customcallbacks.select_table_view_callback
        )
        datasources.table_data_source.selected.js_on_change(
            "indices", customcallbacks.table_selection_callback
        )

    def _create_layout(self, inputwidgets, tableobjects):
        heading_1 = Div(text="<h2>ONNX Quant Stats Visualizer</h2>")
        heading_2 = Div(text=f"<h2>{self.entity_label} Stats Data Table</h2>")

        sp1 = Spacer(
            width=QuantStatsVisualizer.spacer_dims["sp1_width"],
            height=QuantStatsVisualizer.spacer_dims["sp1_height"],
        )
        row1 = row(inputwidgets.ymin_input, inputwidgets.ymax_input)
        row2 = row(inputwidgets.minclip_input, inputwidgets.maxclip_input)
        inputs1 = column(row1, row2)

        layout = column(
            heading_1,
            inputs1,
            sp1,
            self.plot,
            column(
                heading_2,
                row(inputwidgets.table_view_select, inputwidgets.name_input),
                tableobjects.data_table,
            ),
        )

        return layout

    def export_plot_as_html(self, save_path: str) -> None:
        """
        Method for constructing the visualization and saving it to the given path.

        :param save_path: Path for saving the visualization.
        """
        curdoc().theme = "light_minimal"

        self.plot.width = QuantStatsVisualizer.plot_dims["plot_width"]
        self.plot.height = QuantStatsVisualizer.plot_dims["plot_height"]

        # Defining the default values of plotting parameters
        self.default_values["default_ymax"] = QuantStatsVisualizer.initial_vals[
            "default_ymax"
        ]
        self.default_values["default_ymin"] = QuantStatsVisualizer.initial_vals[
            "default_ymin"
        ]
        self.default_values["default_xmax"] = len(self.stats_dict["idx"]) - 1
        self.default_values["default_xmin"] = 0
        self.default_values["default_maxclip"] = self.default_values["default_ymax"] / 2
        self.default_values["default_minclip"] = self.default_values["default_ymin"] / 2

        self.plot.x_range = Range1d(0, len(self.stats_dict["idx"]))
        self.plot.y_range = Range1d(
            self.default_values["default_ymax"] * 1.05,
            self.default_values["default_ymin"] * 1.05,
        )

        # Creating and adding a reset tool
        rt = ResetTool()
        self.plot.add_tools(rt)

        # Defining Bokeh ColumnDataSources
        datasources = DataSources(
            stats_dict=self.stats_dict,
            plot=self.plot,
            default_values=self.default_values,
        )

        # Creating plot objects
        selections = self._add_plot_lines(datasources)

        # Defining the table objects and name filter views
        tableobjects = TableObjects(datasources)

        # Marker points to see which layers cross the thresholds
        min_markers, max_markers = self._add_min_max_markers(datasources, tableobjects)

        # Defining a hover functionality to see layer details on hovering on the marker points and selections
        marker_hover = self._get_marker_hovertool(min_markers, max_markers)
        selection_hover = self._get_selection_hovertool(selections)
        self.plot.add_tools(marker_hover, selection_hover)

        # Creating the input widgets
        inputwidgets = InputWidgets(self.default_values)

        # Defining Custom JavaScript callbacks
        customcallbacks = self._define_callbacks(
            datasources, tableobjects, inputwidgets
        )

        # Attach events to corresponding callbacks
        curdoc().js_on_event(DocumentReady, customcallbacks.reset_callback)
        self._attach_callbacks(datasources, inputwidgets, customcallbacks)

        # Define the formatting
        layout = self._create_layout(inputwidgets, tableobjects)

        # Save as standalone html
        output_file(
            save_path, title=f"ONNX Quant Stats Visualizer ({self.entity_label})"
        )
        save(layout, save_path)
