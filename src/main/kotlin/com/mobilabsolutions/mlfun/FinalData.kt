package com.mobilabsolutions.mlfun

import java.io.Serializable

data class FinalData(
        val pClass: Int,
        val isMale: Boolean,
        val fare: Double?,
        val familyBool: Boolean,
        val parchBool: Boolean,
        val sibSpBool: Boolean,
        val survived: Boolean) : Serializable