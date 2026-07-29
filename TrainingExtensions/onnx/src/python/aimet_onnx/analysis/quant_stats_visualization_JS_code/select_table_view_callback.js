// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

var view = select.value;
let table_booleans = process_table_view(view, name_filter, min_thresh_filter, max_thresh_filter);

for (let j = 0; j < table_columns.length; j++) {
    table_data_source.data[table_columns[j]] = [];
}

for (let i = 0; i < data_source.data['idx'].length; i++) {
    if (table_booleans[i] == true) {
        for (let j = 0; j < table_columns.length; j++) {
            table_data_source.data[table_columns[j]].push(data_source.data[table_columns[j]][i]);
        }
    }
}

table_data_source.change.emit();

const selected_indices = [];
var layer_idx;
for (let i = 0; i < table_data_source.data['idx'].length; i++) {
    layer_idx = table_data_source.data['idx'][i];
    if (data_source.data["selected"][layer_idx] == true) {
        selected_indices.push(i);
    }
}

table_data_source.selected.indices = selected_indices;

table_data_source.change.emit();

// Force redraw of table by making an inert change to table properties
table.name = "placeholder_1";
table.name = "placeholder_0";
