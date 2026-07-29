// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Resetting the limits source with default values
limits_source.data['ymax'] = default_values_source.data['default_ymax'];
limits_source.data['ymin'] = default_values_source.data['default_ymin'];
limits_source.data['xmax'] = default_values_source.data['default_xmax'];
limits_source.data['xmin'] = default_values_source.data['default_xmin'];
limits_source.data['maxclip'] = default_values_source.data['default_maxclip'];
limits_source.data['minclip'] = default_values_source.data['default_minclip'];

// Resetting the plot ranges
plot.y_range.start = limits_source.data['ymin'][0]*1.05;
plot.y_range.end = limits_source.data['ymax'][0]*1.05;
plot.x_range.start = limits_source.data['xmin'][0];
plot.x_range.end = limits_source.data['xmax'][0];

// Resetting the input widget values
ymax_input.value = limits_source.data['ymax'][0].toString();
ymin_input.value = limits_source.data['ymin'][0].toString();
maxclip_input.value = limits_source.data['maxclip'][0].toString();
minclip_input.value = limits_source.data['minclip'][0].toString();

data_source.data['marker_yminlist'] = data_source.data['minlist'].map(t => findMax(t, limits_source.data['ymin'][0]));
data_source.data['marker_ymaxlist'] = data_source.data['maxlist'].map(t => findMin(t, limits_source.data['ymax'][0]));

min_thresh_filter.booleans = data_source.data['minlist'].map(t => t <= limits_source.data['minclip'][0]);
max_thresh_filter.booleans = data_source.data['maxlist'].map(t => t >= limits_source.data['maxclip'][0]);

name_filter.booleans = Array(data_source.data['idx'].length).fill(true);
name_input.value = "";

select.value = "Min | Max";
var view = select.value;
let table_booleans = process_table_view(view, name_filter, min_thresh_filter, max_thresh_filter);

for (let i = 0; i < data_source.data['idx'].length; i++) {
    if (table_booleans[i] == true) {
        for (let j = 0; j < table_columns.length; j++) {
            table_data_source.data[table_columns[j]].push(data_source.data[table_columns[j]][i]);
        }
    }
}

table_data_source.selected.indices = [];
data_source.data["selected"] = Array(data_source.data["idx"].length).fill(false);

selected_data_source.data["floor"] = [];
selected_data_source.data["ceil"] = [];

for (let j = 0; j < selection_columns.length; j++) {
    selected_data_source.data[selection_columns[j]] = [];
}

// Emitting the changes made to ColumnDataSources
limits_source.change.emit();
data_source.change.emit();
table_data_source.change.emit();
selected_data_source.change.emit();
