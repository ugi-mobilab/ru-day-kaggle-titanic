package com.mobilabsolutions.mlfun

import java.io.Serializable

class InitialData(passengerId: String,
                  survived: String,
                  pClass: String,
                  name: String,
                  sex: String,
                  age: String,
                  sibSp: String,
                  parch: String,
                  ticket: String,
                  fare: String,
                  cabin: String,
                  embarked: String) : Serializable {

    val passengerId: Int = passengerId.toInt()
    val survived: Boolean = survived == "1"
    val pClass: Int = pClass.toInt()
    val isMale: Boolean = sex == "male"
    val age: Double? = age.toDoubleOrNull()
    val sibSp: Int = sibSp.toInt()
    val parch: Int = parch.toInt()
    val fare: Double? = fare.toDoubleOrNull()
    val cabin: String? = if (cabin.isEmpty()) null else cabin
    val embarked: String? = if (embarked.isEmpty()) null else embarked

    override fun toString(): String {
        return "InitialData(passengerId=$passengerId, survived=$survived, pClass=$pClass, isMale=$isMale, age=$age, sibSp=$sibSp, parch=$parch, fare=$fare, cabin=$cabin, embarked=$embarked)"
    }
}