// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Interactive filtering/highlighting for the sensitivity plot.
//
// Shared callback: invoked on NumericInput `value` changes AND on Toggle
// `active` changes. It recomputes highlight (color/size/halo on the scatter
// points), the threshold guide line, the match-count label, and the table rows
// from scratch based on the current UI state.
//
// CustomJS args:
//   source           -- full ColumnDataSource backing the plot (includes per-point
//                       `color`, `size`, `line_color`, `line_width` columns that
//                       drive the scatter glyph)
//   filtered_source  -- ColumnDataSource backing the DataTable
//   threshold_input  -- NumericInput model for the score threshold
//   threshold_span   -- Span model for the horizontal threshold guide line
//   toggles          -- array of Toggle models, one per pattern in `patterns`
//   patterns         -- array of strings (e.g. "down_proj"); parallel to `toggles`
//   match_count_div  -- Div model for the "N matching" label

const BASE_COLOR = "#4a8cc7";
const HIGHLIGHT_COLOR = "#d62728";
const BASE_SIZE = 5;
const HIGHLIGHT_SIZE = 10;
const HALO_COLOR = "#ffffff";
const HALO_WIDTH = 2;

const threshold = threshold_input.value;
const full = source.data;
const n = full.index.length;

// Flip toggle button color to match highlight red when active.
for (let i = 0; i < toggles.length; i++) {
    toggles[i].button_type = toggles[i].active ? "danger" : "default";
}

const activePatterns = [];
for (let i = 0; i < toggles.length; i++) {
    if (toggles[i].active) activePatterns.push(patterns[i]);
}

const matchesPattern = (name) => activePatterns.some(p => name && name.indexOf(p) !== -1);

// Update scatter colors/sizes/halo in-place.
const colors = new Array(n);
const sizes = new Array(n);
const lineColors = new Array(n);
const lineWidths = new Array(n);
for (let i = 0; i < n; i++) {
    const hit = matchesPattern(full.name[i]);
    colors[i] = hit ? HIGHLIGHT_COLOR : BASE_COLOR;
    sizes[i] = hit ? HIGHLIGHT_SIZE : BASE_SIZE;
    lineColors[i] = HALO_COLOR;
    lineWidths[i] = hit ? HALO_WIDTH : 0;
}
full.color = colors;
full.size = sizes;
full.line_color = lineColors;
full.line_width = lineWidths;
source.change.emit();

// Threshold guide line visibility.
const thresholdActive = threshold !== null && threshold !== undefined && !isNaN(threshold);
if (thresholdActive) {
    threshold_span.location = threshold;
    threshold_span.visible = true;
} else {
    threshold_span.visible = false;
}

// Recompute filtered table rows: threshold and pattern filters intersect.
// If neither filter is active, the table stays empty.
const patternActive = activePatterns.length > 0;

const rowIndices = [];
if (thresholdActive || patternActive) {
    for (let i = 0; i < n; i++) {
        const passThresh = !thresholdActive || full.score[i] < threshold;
        const passPattern = !patternActive || matchesPattern(full.name[i]);
        if (passThresh && passPattern) rowIndices.push(i);
    }
    rowIndices.sort((a, b) => full.score[a] - full.score[b]);
}

const filtered = {};
for (const key of Object.keys(full)) {
    filtered[key] = rowIndices.map(i => full[key][i]);
}
filtered_source.data = filtered;
filtered_source.change.emit();

// Match count label.
if (!thresholdActive && !patternActive) {
    match_count_div.text = "<span style='color:#888;font-size:12px'>No filter active.</span>";
} else {
    match_count_div.text = (
        "<span style='color:#333;font-size:13px;font-weight:500'>"
        + rowIndices.length + " matching"
        + "</span>"
    );
}
