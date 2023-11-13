package com.plcoding.landmarkrecognitiontensorflow.presentation

fun FloatArray.indexOf(value: Float): Int {
    for (i in indices) {
        if (this[i] == value) {
            return i
        }
    }
    return -1
}