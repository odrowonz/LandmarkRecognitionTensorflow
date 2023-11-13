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
import com.plcoding.landmarkrecognitiontensorflow.presentation.indexOf
import com.plcoding.landmarkrecognitiontensorflow.presentation.isBitmapRGBA
import com.plcoding.landmarkrecognitiontensorflow.presentation.removeAlphaChannel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.UUID

//import java.util.*

class TfLiteLandmarkClassifier (
    private val context: Context,
    private val threshold: Float = 0.5f,
    private val maxResults: Int = 3,
    private var labels: List<String> = emptyList(),
    private val selectedLabel: MutableState<SelectedLabel?>,
    private val selectedMode: MutableState<Boolean>
): LandmarkClassifier {
    private var classifier: ImageClassifier? = null
    private var interpreter: Interpreter

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
                "calendar_224_224_4_meta.tflite",
                options
            )
        } catch (e: IllegalStateException) {
            e.printStackTrace()
        }
    }
    init {
        //setupClassifier()

        // Загрузка TFLite модели
        val assetManager = context.assets
        val model = "calendar_224_224_4_meta.tflite"

        val modelFileDescriptor = assetManager.openFd(model)
        val modelInputStream = modelFileDescriptor.createInputStream()
        val modelFileChannel = modelInputStream.channel
        val modelBuffer = modelFileChannel.map(FileChannel.MapMode.READ_ONLY, modelFileDescriptor.startOffset, modelFileDescriptor.declaredLength)
        modelBuffer.order(ByteOrder.nativeOrder())
        interpreter = Interpreter(modelBuffer)
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
            imageToSave.removeAlphaChannel().compress(Bitmap.CompressFormat.PNG, 100, fos)
            fos.flush()
            fos.close()
            Log.d("TfLiteLandmarkClassifier", "Изображение (alpha "+imageToSave.removeAlphaChannel().hasAlpha()+") сохранено: " + imageFile.absolutePath)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
    override fun classify(bitmap: Bitmap, rotation: Int): List<Classification> {
        val intValues = IntArray(bitmap.width * bitmap.height)
        /*val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        resizedBitmap.getPixels(
            intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height
        )*/
        bitmap.getPixels(
            intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height
        )

        val inputBuffer = ByteBuffer.allocateDirect(1 * bitmap.width * bitmap.height * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val pixelValue = intValues[y * bitmap.width + x]
                // BYTE started from 25 (Alpha) bit IS IGNORED
                // Normalization of BYTE started from 17 bit (Red)
                inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) / 255.0f))
                // Normalization of BYTE started from 9 bit (Green)
                inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) / 255.0f))
                // Normalization of BYTE started from 0 bit (Blue)
                inputBuffer.putFloat(((pixelValue and 0xFF) / 255.0f))
            }
        }

        val outputBuffer = ByteBuffer.allocateDirect(1 * 8 * 4) // 8 is NUM_CLASSES
        outputBuffer.order(ByteOrder.nativeOrder())

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind();
        val result = outputBuffer.asFloatBuffer();
        Log.d("TfLiteLandmarkClassifier", result.toString())
        Log.d("TfLiteLandmarkClassifier", labels.toString())

        // All classes with probabilties
        val classifications = mutableListOf<Classification>()
        for (i in 0 until result.capacity() step 1) {
            val name = labels[i]
            val score = result.get(i)
            val classification = Classification(name, score)
            classifications.add(classification)
        }

        // Selected classes with the highest probability
        val filteredClassifications = classifications
            .filter { it.score >= threshold } // Оставить только те, у кого score >= threshold
            .sortedByDescending { it.score } // Отсортировать по убыванию score
            .take(maxResults) // Взять первые maxResults результатов

        // Protection against empty result
        val resultClassifications =
            if (filteredClassifications.isEmpty())
            {
                val noObject = Classification("no_objects", 1.0f)
                mutableListOf(noObject)
            } else {
                filteredClassifications
            }

        val label = selectedLabel.value?.label
        val maxScoreCategory = resultClassifications.maxByOrNull { it.score }
        val maxScoreLabel = maxScoreCategory?.name ?: ""

        if (selectedMode.value &&
            (label != null) && (label != "") &&
            (maxScoreLabel != "") &&
            (label != maxScoreLabel)) {
            Log.d("TfLiteLandmarkClassifier", "Real class of object: $label not equal predicted: $maxScoreLabel")
            // Генерируйте GUID (пример)
            val guid = generateGuid(); // Необходимо создать функцию generateGuid()

            // Создайте имя файла
            val fileName = "err_" + maxScoreLabel + "_" + guid + ".png";

            // Сохраните изображение в файл
            label.let {
                //saveImageToStorage(resizedBitmap, it, fileName)
                saveImageToStorage(bitmap, it, fileName)
            }
            Log.d("TfLiteLandmarkClassifier", "$fileName was saved.")
        }

        return resultClassifications
    }
    fun base_classify(bitmap: Bitmap): List<Classification> {
        val imageProcessor = ImageProcessor.Builder()
            //.add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Нормализация значений от 0 до 1
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap) // TensorImage.fromBitmap(bitmap)
        Log.d("TfLiteLandmarkClassifier","tensorImage0: " + tensorImage.getTensorBuffer().getFloatArray()[0])

        val processedTensorImage = imageProcessor.process(tensorImage)
        Log.d("TfLiteLandmarkClassifier","tensorImage0: " + processedTensorImage.getTensorBuffer().getFloatArray()[0])

        val imageProcessingOptions = ImageProcessingOptions.builder()
            //.setOrientation(getOrientationFromRotation(rotation))
            .build()

        val results = classifier?.classify(processedTensorImage, imageProcessingOptions)
        Log.d("TfLiteLandmarkClassifier-Results", results.toString())
        val label = selectedLabel.value?.label

        val maxScoreCategory = results?.flatMap { it.categories }
            ?.maxByOrNull { it.score }

        val maxScoreLabel = maxScoreCategory?.label ?: ""
        if (selectedMode.value &&
            (label != null) && (label != "") &&
            (maxScoreLabel != "") &&
            (label != maxScoreLabel)) {
            Log.d("TfLiteLandmarkClassifier", "Real class of object: $label not equal predicted: $maxScoreLabel")
            // Генерируйте GUID (пример)
            val guid = generateGuid(); // Необходимо создать функцию generateGuid()

            // Создайте имя файла
            val fileName = "err_" + maxScoreLabel + "_" + guid + ".png";

            // Сохраните изображение в файл
            label.let {
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