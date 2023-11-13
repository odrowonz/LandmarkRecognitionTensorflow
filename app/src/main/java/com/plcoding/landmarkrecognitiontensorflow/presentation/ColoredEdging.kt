package com.plcoding.landmarkrecognitiontensorflow.presentation

import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun ColoredEdging(
    //controller: LifecycleCameraController,
    modifier: Modifier = Modifier,
    edgingColor: Color = Color.Magenta // Цвет фона окантовки
) {
    /*val display = LocalDensity.current.density
    val displayMetrics = LocalDensity.current.density

    val screenSize = with(LocalDensity.current) {
        IntSize(
            (displayMetrics.widthPixels / display).toInt(),
            (displayMetrics.heightPixels / display).toInt()
        )
    }
    val minSize = min(screenSize.width, screenSize.height)

    val colorBarsModifier = Modifier
        .background(color)
        .fillMaxWidth()
        .fillMaxHeight((screenSize.height - minSize) / 2 / screenSize.height.toFloat())
    val squareModifier = Modifier
        .fillMaxSize()
        .aspectRatio(1f) // Установим соотношение сторон 1:1 для квадрата
    */

    Column(modifier = modifier) {
        // Верхняя полоска
        Box(modifier = Modifier
            .weight(1f)
            .background(edgingColor)
            .fillMaxWidth()
            //.height(100.dp) // Установите высоту верхней полоски
        ) { }

        Spacer(modifier = Modifier
            .aspectRatio(1f)
            .fillMaxWidth()
        )

        // Нижняя полоска
        Box(modifier = Modifier
            .weight(1f)
            .background(edgingColor)
            .fillMaxWidth()
            //.height(100.dp) // Установите высоту нижней полоски
        ) { }
    }
}