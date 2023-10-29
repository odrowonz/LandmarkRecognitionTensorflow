package com.plcoding.landmarkrecognitiontensorflow

import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.plcoding.landmarkrecognitiontensorflow.data.TfLiteLandmarkClassifier
import com.plcoding.landmarkrecognitiontensorflow.domain.Classification
import com.plcoding.landmarkrecognitiontensorflow.presentation.CameraPreview
import com.plcoding.landmarkrecognitiontensorflow.presentation.LandmarkImageAnalyzer
import com.plcoding.landmarkrecognitiontensorflow.ui.theme.LandmarkRecognitionTensorflowTheme
import android.Manifest
import android.util.Log
import androidx.compose.material3.Button
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.colorResource

data class SelectedLabel(val label: String)

class MainActivity : ComponentActivity() {
    private var labels: List<String> = emptyList()
    private val selectedLabel: MutableState<SelectedLabel?> = mutableStateOf(null)
    private val selectedMode: MutableState<Boolean> = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if(!hasCameraPermission()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), 0
            )
        }
        // Проверяем, есть ли у нас разрешение WRITE_EXTERNAL_STORAGE
        if(!hasWriteExternalStoragePermission()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), 0
            )
        }

        // Инициализируйте labels внутри onCreate
        labels = loadLabelsFromAssets("labels.txt")
        setContent {
            LandmarkRecognitionTensorflowTheme {
                var classifications by remember {
                    mutableStateOf(emptyList<Classification>())
                }
                val analyzer = remember {
                    LandmarkImageAnalyzer(
                        classifier = TfLiteLandmarkClassifier(
                            context = applicationContext,
                            selectedLabel = selectedLabel,
                            selectedMode = selectedMode
                        ),
                        onResults = {
                            classifications = it
                        }
                    )
                }
                val controller = remember {
                    LifecycleCameraController(applicationContext).apply {
                        setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
                        setImageAnalysisAnalyzer(
                            ContextCompat.getMainExecutor(applicationContext),
                            analyzer
                        )
                    }
                }
                var expanded by remember { mutableStateOf(false) }

                Box(
                    modifier = Modifier
                        .fillMaxSize()
                ) {
                    // CameraPreview(controller, Modifier.fillMaxSize())
                    CameraPreview(
                        controller = controller,
                        modifier = Modifier.fillMaxSize(),
                        edgingColor = colorResource(id = R.color.purple_200)
                    )

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .align(Alignment.TopCenter)
                    ) {
                        classifications.forEach {
                            Text(
                                text = "${it.name} ${it.score}",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(MaterialTheme.colorScheme.primaryContainer)
                                    .padding(8.dp),
                                textAlign = TextAlign.Center,
                                fontSize = 20.sp,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }

                        OutlinedButton(
                            onClick = { expanded = true },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(text = selectedLabel.value?.label ?: "Select Label")
                        }

                        DropdownMenu(
                            expanded = expanded,
                            onDismissRequest = { expanded = false },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            labels.forEach { label ->
                                Log.d("TestKrokozabra", label)
                                DropdownMenuItem(
                                    onClick = {
                                        selectedLabel.value = SelectedLabel(label)
                                        expanded = false
                                    },
                                    text = {
                                        Text(text = label ?: "")
                                    }
                                )
                            }
                        }

                        Button(
                            onClick = {
                                if (selectedLabel.value != null) {
                                    selectedMode.value = !selectedMode.value
                                }
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(if (selectedMode.value) Color.Red else Color.Green)
                        ) {
                            Text(text = if (selectedMode.value) "Recording" else "Start")
                        }
                    }
                }
            }
        }
    }

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun hasWriteExternalStoragePermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.WRITE_EXTERNAL_STORAGE
    ) == PackageManager.PERMISSION_GRANTED

    private fun loadLabelsFromAssets(filename: String): List<String> {
        return try {
            assets.open(filename).bufferedReader().useLines { it.toList() }
        } catch (e: Exception) {
            Log.d("TestKrokozabra", e.stackTraceToString())
            e.printStackTrace()
            emptyList()
        }
    }
}