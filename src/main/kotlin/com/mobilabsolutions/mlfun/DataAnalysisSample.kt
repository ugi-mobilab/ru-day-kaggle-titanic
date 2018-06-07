package com.mobilabsolutions.mlfun

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.IntWritable
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.slf4j.LoggerFactory
import java.io.File
import kotlin.math.roundToInt


class DataAnalysisSample {

    val conf = SparkConf()

    init {
        conf.setMaster("local[*]")
        conf.setAppName("DataVec Example")
    }

    companion object {
        private val log = LoggerFactory.getLogger(DataAnalysisSample::class.java)

        @Throws(Exception::class)
        @JvmStatic fun main(args: Array<String>) {
            DataAnalysisSample().analyze()
        }
    }

    private fun analyze() {
        val sparkContext = JavaSparkContext(conf)

        val parsedStringData = sparkContext.textFile(File("data/train.csv").absolutePath)

        val schema = Schema.Builder()
                .addColumnsInteger("PassangerId", "Survived", "Pclass")
                .addColumnsString("Name", "Sex")
                .addColumnsInteger("Age", "SibSp", "Parch")
                .addColumnString("Ticket")
                .addColumnDouble("Fare")
                .addColumnsString("Embarked")
                .build()

        log.info(schema.toString())

        val transformProcess = TransformProcess.Builder(schema)
                .conditionalReplaceValueTransform(
                        "Age",
                        IntWritable(getAverageAge(parsedStringData)),
                        InvalidValueColumnCondition("Age")
                )
                .build()

        val parsedInputData = parsedStringData.map(StringToWritablesFunction(CSVRecordReader()))

        val transformedData = SparkTransformExecutor.execute(parsedInputData, transformProcess)

        //transformedData.collect().forEach { println(it) }
    }

    private fun getAverageAge(parsedStringData: JavaRDD<String>): Int {
        val ages = parsedStringData.cache().map { it.split(",")[6] }
                .map { it.toDoubleOrNull() }
                .filter { it != null }

        val rowCountWithAge = ages.count()

        return ages.reduce { v1, v2 -> v1!!.plus(v2!!) }!!.div(rowCountWithAge).roundToInt()
    }
}