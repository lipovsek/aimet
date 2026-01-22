# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Create visualizations on the weights in each conv and linear layer in a model"""

import pandas as pd
from bokeh.models import HoverTool, WheelZoomTool, ColumnDataSource
from bokeh.plotting import figure

# Some magic stuff happening during import that ties pandas dataframe to hvplot
# Need this import, please don't remove
import hvplot.pandas  # pylint:disable=unused-import


def style(p):
    """
    Style bokeh figure object p and return the styled object
    :param p: Bokeh figure object
    :return: Bokeh figure object
    """
    # Title
    p.title.align = "center"
    p.title.text_font_size = "14pt"
    p.title.text_font = "serif"

    # Axis titles
    p.xaxis.axis_label_text_font_size = "12pt"
    # p.xaxis.axis_label_text_font_style = 'bold'
    p.yaxis.axis_label_text_font_size = "12pt"
    #     p.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    p.xaxis.major_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "10pt"

    p.add_tools(WheelZoomTool())

    return p


def plot_optimal_compression_ratios(comp_ratios, layer_names):
    """
    Makes a plot for compression ratios and layers, such that you can hover over the point and see the layer and its comp ratio
    :param comp_ratios: python list of compression ratios
    :param layer_names: python list of string layer names
    :return: bokeh figure object
    """
    df = pd.DataFrame.from_dict(
        {
            "layers": layer_names,
            "comp_ratios": comp_ratios,
            "index": [i + 1 for i in range(len(comp_ratios))],
        }
    )

    # None means that the layer was not compressed at all, which is equivalent to a compression ratio of 1.
    df.replace({None: 1}, inplace=True)
    source = ColumnDataSource(data=df)

    plot = figure(
        x_axis_label="Layers",
        y_axis_label="Compression Ratios",
        title="Optimal Compression Ratios For Each Layer",
        tools="pan, box_zoom, crosshair, reset, save",
        sizing_mode="stretch_width",
    )

    plot.line(
        x="index", y="comp_ratios", line_width=2, line_color="green", source=source
    )
    plot.circle(
        x="index", y="comp_ratios", color="black", alpha=0.7, size=10, source=source
    )

    plot.add_tools(
        HoverTool(
            tooltips=[("Layer", "@layers"), ("Comp Ratio", "@comp_ratios")],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode="vline",
        )
    )
    style(plot)
    plot.xaxis.major_label_text_color = (
        None  # note that this leaves space between the axis and the axis label
    )
    return plot
