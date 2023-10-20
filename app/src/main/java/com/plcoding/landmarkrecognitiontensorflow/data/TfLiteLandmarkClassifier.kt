package com.plcoding.landmarkrecognitiontensorflow.data

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.view.Surface
import com.plcoding.landmarkrecognitiontensorflow.domain.Classification
import com.plcoding.landmarkrecognitiontensorflow.domain.LandmarkClassifier
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
//import java.util.*

class TfLiteLandmarkClassifier (
    private val context: Context,
    private val threshold: Float = 0.5f,
    private val maxResults: Int = 3//,
    //private val labels: List<String>
): LandmarkClassifier {
    private var classifier: ImageClassifier? = null

    private fun setupClassifier() {
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()
        val options = ImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(maxResults)
            .setScoreThreshold(threshold)
            .build()
        try {
            classifier = ImageClassifier.createFromFileAndOptions(
                context,
                "calendar_224_224_3_1.tflite",
                options
            )
        } catch (e: IllegalStateException) {
            e.printStackTrace()
        }
    }

    override fun classify(bitmap: Bitmap, rotation: Int): List<Classification> {
        if(classifier == null) {
            setupClassifier()
        }

        val imageProcessor = ImageProcessor.Builder().build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build()

        // Log the processed image dimensions
        val width = tensorImage.width
        val height = tensorImage.height
        Log.d("TfLiteLandmarkClassifier", "Image Dimensions: $width x $height")

        val results = classifier?.classify(tensorImage, imageProcessingOptions)

            /*if (results != null) {
            results.forEach { classification ->
                val labelIndex = classification.categoryLabel
                if (labelIndex >= 0 && labelIndex < labels.size) {
                    val label = labels[labelIndex]
                    Log.d("TfLiteLandmarkClassifier", "Label: $label, Score: ${classification.categories[0].score}")
                }
            }
        } else {
            Log.e("TfLiteLandmarkClassifier", "No results from classification")
        }*/

        return results?.flatMap { classifications ->
            classifications.categories.map { category ->
                Classification(
                    name = category.label,
                    score = category.score
                )
            }
        }?.distinctBy { it.name } ?: emptyList()
        /*return results?.flatMap { classification ->
            classification.categories.map { category ->
                val labelIndex = category.label
                //val name = labels.getOrNull(labelIndex) ?: category.displayName
                Classification(
                    name = labelIndex,
                    score = category.score
                )
            }
        }?.distinctBy { it.name } ?: emptyList()*/
    }

    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        return when(rotation) {
            Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_90  -> ImageProcessingOptions.Orientation.TOP_LEFT
            Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            else -> ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }
}