package com.plcoding.landmarkrecognitiontensorflow.data

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import android.view.Surface
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.core.app.ActivityCompat.requestPermissions
import androidx.core.content.ContextCompat
import androidx.core.content.ContextCompat.checkSelfPermission
import com.plcoding.landmarkrecognitiontensorflow.SelectedLabel
import com.plcoding.landmarkrecognitiontensorflow.domain.Classification
import com.plcoding.landmarkrecognitiontensorflow.domain.LandmarkClassifier
import com.plcoding.landmarkrecognitiontensorflow.presentation.isBitmapRGBA
import com.plcoding.landmarkrecognitiontensorflow.presentation.removeAlphaChannel
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.UUID

//import java.util.*

class TfLiteLandmarkClassifier (
    private val context: Context,
    private val threshold: Float = 0.5f,
    private val maxResults: Int = 3,
    private val selectedLabel: MutableState<SelectedLabel?>,
    private val selectedMode: MutableState<Boolean>
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
    init {
        setupClassifier()
    }

    // Генерация GUID
    private fun generateGuid(): String? {
        return UUID.randomUUID().toString()
    }
    private fun saveImageToStorage(imageToSave: Bitmap, selectedClass: String, fileName: String) {
        // Получите путь к директории "Загрузки"
        val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)

        val classDir = File(storageDir, selectedClass)

        // Убедитесь, что подпапка существует или создайте ее
        if (!classDir.exists()) {
            classDir.mkdirs()
        }

        val imageFile = File(classDir, fileName)
        try {
            val fos = FileOutputStream(imageFile)
            imageToSave
                .removeAlphaChannel().compress(Bitmap.CompressFormat.PNG, 100, fos)
            fos.flush()
            fos.close()
            Log.d("TfLiteLandmarkClassifier", "Изображение (alpha "+imageToSave.hasAlpha()+") сохранено: " + imageFile.absolutePath)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
    override fun classify(bitmap: Bitmap, rotation: Int): List<Classification> {
        /*if(classifier == null) {
            setupClassifier()
        }*/
        /*if (bitmap.isBitmapRGBA()) {
            Log.d("TfLiteLandmarkClassifier", "Image consists alpha channel!")
        }*/

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build()

        // Log the processed image dimensions
        //val width = tensorImage.width
        //val height = tensorImage.height
        //Log.d("TfLiteLandmarkClassifier", "Image Dimensions: $width x $height")

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
        val label = selectedLabel?.value?.label

        val maxScoreCategory = results?.flatMap { it.categories }
            ?.maxByOrNull { it.score }

        val maxScoreLabel = maxScoreCategory?.label ?: ""
        if (selectedMode.value &&
            (label != null) && (label != "") &&
            (maxScoreLabel != "") && (maxScoreLabel != null) &&
            (label != maxScoreLabel)) {
            Log.d("TfLiteLandmarkClassifier", "Real class of object: $label not equal predicted: $maxScoreLabel")
            // Генерируйте GUID (пример)
            val guid = generateGuid(); // Необходимо создать функцию generateGuid()

            // Создайте имя файла
            val fileName = "err_" + maxScoreLabel + "_" + guid + ".png";

            // Сохраните изображение в файл
            label?.let {
                saveImageToStorage(tensorImage.bitmap, it, fileName)
            }
            Log.d("TfLiteLandmarkClassifier", "$fileName was saved.")
        }

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
            Surface.ROTATION_270 -> {Log.d("TfLiteLandmarkClassifier", "Surface.ROTATION_270")
                                     ImageProcessingOptions.Orientation.BOTTOM_RIGHT}
            Surface.ROTATION_90  -> {Log.d("TfLiteLandmarkClassifier", "Surface.ROTATION_90")
                                     ImageProcessingOptions.Orientation.TOP_LEFT}
            Surface.ROTATION_180 -> {Log.d("TfLiteLandmarkClassifier", "Surface.ROTATION_180")
                                     ImageProcessingOptions.Orientation.RIGHT_BOTTOM}
            else -> {//Log.d("TfLiteLandmarkClassifier", "Surface.ROTATION_0")
                     ImageProcessingOptions.Orientation.LEFT_TOP
                     //ImageProcessingOptions.Orientation.RIGHT_TOP
                    }
        }
    }
}