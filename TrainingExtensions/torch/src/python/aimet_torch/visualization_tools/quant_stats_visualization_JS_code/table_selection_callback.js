// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

table_data_source.data["idx"].forEach(i => {
    data_source.data["selected"][i] = false;
});
table_data_source.selected.indices.forEach(i => {
    let layer_idx = table_data_source.data["idx"][i];
    data_source.data["selected"][layer_idx] = true;
});
data_source.change.emit();

selected_data_source.data["floor"] = [];
selected_data_source.data["ceil"] = [];

for (let j = 0; j < selection_columns.length; j++) {
    selected_data_source.data[selection_columns[j]] = [];
}

selected_data_source.change.emit();

data_source.data["selected"].forEach((bool,index) => {
    if (bool==true) {
        selected_data_source.data["floor"].push(limits_source.data['ymin'][0]*1.05);
        selected_data_source.data["ceil"].push(limits_source.data['ymax'][0]*1.05);
        for (let j = 0; j < selection_columns.length; j++) {
            selected_data_source.data[selection_columns[j]].push(data_source.data[selection_columns[j]][index]);
        }
    }
})

if (mode=="advanced") {
    boxplot.x_range.factors = selected_data_source.data["stridx"];
    let box_min = arrayMin(selected_data_source.data["minlist"]) * 1.2;
    let box_max = arrayMax(selected_data_source.data["maxlist"]) * 1.2;
    let whisker_min = arrayMin(selected_data_source.data["boxplot_lower_list"]) * 1.2;
    let whisker_max = arrayMax(selected_data_source.data["boxplot_upper_list"]) * 1.2;
    let box_abs = arrayMax([box_max, -box_min, whisker_max, -whisker_min]);
    boxplot.y_range.start = -box_abs;
    boxplot.y_range.end = box_abs;
    if (selected_data_source.data['idx'].length > 5) {
        boxplot.width = boxplot_unit_width * selected_data_source.data['idx'].length;
    }
    else {
        boxplot.width = boxplot_unit_width * 5;
    }
}

selected_data_source.change.emit();
