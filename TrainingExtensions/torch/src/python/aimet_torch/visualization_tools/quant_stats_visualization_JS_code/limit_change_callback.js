// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Reading values from input widgets and setting plot y axis range
limits_source.data['ymax'] = [parseFloat(ymax_input.value)];
limits_source.data['ymin'] = [parseFloat(ymin_input.value)];
plot.y_range.start = limits_source.data['ymin'][0]*1.05;
plot.y_range.end = limits_source.data['ymax'][0]*1.05;
limits_source.data['maxclip'] = [parseFloat(maxclip_input.value)];
limits_source.data['minclip'] = [parseFloat(minclip_input.value)];

data_source.data['marker_yminlist'] = data_source.data['minlist'].map(t => findMax(t, limits_source.data['ymin'][0]));
data_source.data['marker_ymaxlist'] = data_source.data['maxlist'].map(t => findMin(t, limits_source.data['ymax'][0]));

// Updating the filters for finding layers that cross the min or max thresholds
min_thresh_filter.booleans = data_source.data['minlist'].map(t => t <= limits_source.data['minclip'][0]);
max_thresh_filter.booleans = data_source.data['maxlist'].map(t => t >= limits_source.data['maxclip'][0]);

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

selected_data_source.data["floor"].push(limits_source.data['ymin'][0]*1.05);
selected_data_source.data["ceil"].push(limits_source.data['ymax'][0]*1.05);

// Emitting the changes made to ColumnDataSources
limits_source.change.emit();
data_source.change.emit();
table_data_source.change.emit();
selected_data_source.change.emit();