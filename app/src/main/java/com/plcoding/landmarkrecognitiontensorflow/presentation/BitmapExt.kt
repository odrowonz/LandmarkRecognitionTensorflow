package com.plcoding.landmarkrecognitiontensorflow.presentation

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import java.lang.Math.min

fun Bitmap.centerCrop(desiredWidth: Int, desiredHeight: Int): Bitmap {
    val xStart = (width - desiredWidth) / 2
    val yStart = (height - desiredHeight) / 2

    if(xStart < 0 || yStart < 0 || desiredWidth > width || desiredHeight > height) {
        throw IllegalArgumentException("Invalid arguments for center cropping")
    }

    return Bitmap.createBitmap(this, xStart, yStart, desiredWidth, desiredHeight)
}

fun Bitmap.squareCrop(): Bitmap {
    val demension = min(width, height)
    return this.centerCrop(demension, demension)
}

fun Bitmap.scaleSize(desiredWidth: Int, desiredHeight: Int): Bitmap {
    return Bitmap.createScaledBitmap(this, desiredWidth, desiredHeight, false);
}
fun Bitmap.scaleSquareSize(desiredSize: Int): Bitmap {
    return this.scaleSize(desiredSize, desiredSize)
}

fun Bitmap.rotateBitmap(degrees: Int): Bitmap {
    val matrix = Matrix()
    matrix.postRotate(degrees.toFloat())
    return Bitmap.createBitmap(this, 0, 0, this.width, this.height, matrix, true)
}

fun Bitmap.isBitmapRGBA(): Boolean {
    return this.hasAlpha()
}

fun Bitmap.removeAlphaChannel(): Bitmap {
    // Получите ширину и высоту оригинального изображения
    val width = this.width
    val height = this.height

    // Создайте новый Bitmap без альфа-канала
    val config = Bitmap.Config.RGB_565 // Выберите RGB_565 для 16 битных цветов
    val outputBitmap = Bitmap.createBitmap(width, height, config)

    // Создайте холст для рисования нового изображения
    val canvas = Canvas(outputBitmap)

    // Создайте пустой Paint
    val paint = Paint()

    // Отрисуйте оригинальное изображение на новом Bitmap без альфа-канала
    canvas.drawBitmap(this, 0f, 0f, paint)

    return outputBitmap
}

