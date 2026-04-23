// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

function booleanAnd(arr1, arr2) {
    return arr1.map((value, index) => value && arr2[index]);
}

function booleanOr(arr1, arr2) {
    return arr1.map((value, index) => value || arr2[index]);
}

function findMin(a, b) {
    if (a<=b) {
        return a;
    }
    return b;
}

function findMax(a, b) {
    if (a>=b) {
        return a;
    }
    return b;
}

function arrayMin(arr) {
    var min;
    arr.forEach((val, index) => {
        if (index==0) {
            min = val;
        } else {
            min = findMin(val, min);
        }
    })
    return min;
}

function arrayMax(arr) {
    var max;
    arr.forEach((val, index) => {
        if (index==0) {
            max = val;
        } else {
            max = findMax(val, max);
        }
    })
    return max;
}

function process_table_view(view, name_filter, min_thresh_filter, max_thresh_filter) {
    let table_booleans;

    if (view == "All") {
        table_booleans = name_filter.booleans;
    } else if (view == "Min") {
        table_booleans = booleanAnd(name_filter.booleans, min_thresh_filter.booleans);
    } else if (view == "Max") {
        table_booleans = booleanAnd(name_filter.booleans, max_thresh_filter.booleans);
    } else if (view == "Min | Max") {
        table_booleans = booleanAnd(name_filter.booleans, booleanOr(min_thresh_filter.booleans, max_thresh_filter.booleans));
    } else if (view == "Min & Max") {
        table_booleans = booleanAnd(name_filter.booleans, booleanAnd(min_thresh_filter.booleans, max_thresh_filter.booleans));
    }

    return table_booleans
}
