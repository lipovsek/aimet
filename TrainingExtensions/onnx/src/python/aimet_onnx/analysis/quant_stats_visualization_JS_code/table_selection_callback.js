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
});

selected_data_source.change.emit();
