package com.plcoding.landmarkrecognitiontensorflow.presentation

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.plcoding.landmarkrecognitiontensorflow.domain.Classification
import com.plcoding.landmarkrecognitiontensorflow.domain.LandmarkClassifier

class LandmarkImageAnalyzer(
    private val classifier: LandmarkClassifier,
    private val onResults: (List<Classification>) -> Unit
): ImageAnalysis.Analyzer {

    private var frameSkipCounter = 0

    override fun analyze(image: ImageProxy) {
        if(frameSkipCounter % 2 == 0) {
            val rotationDegrees = image.imageInfo.rotationDegrees
            val bitmap = image
                .toBitmap()
                .rotateBitmap(rotationDegrees)
                .squareCrop()
                .scaleSquareSize(224)

            val results = classifier.classify(bitmap, rotationDegrees)
            onResults(results)
        }
        frameSkipCounter++

        image.close()
    }
}