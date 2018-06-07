package com.mobilabsolutions.mlfun

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition
import org.datavec.api.transform.schema.Schema
import org.datavec.api.transform.transform.doubletransform.MinMaxNormalizer
import org.datavec.api.transform.transform.doubletransform.StandardizeNormalizer
import org.datavec.api.transform.ui.HtmlAnalysis
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Text
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.AnalyzeSpark
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Files
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
        @JvmStatic
        fun main(args: Array<String>) {
            DataAnalysisSample().analyze()
        }
    }

    private fun analyze() {
        val sparkContext = JavaSparkContext(conf)

        val parsedStringData = sparkContext.textFile(File("data/sedat-train.csv").absolutePath)

//        val schema = Schema.Builder()
//                .addColumnsInteger("PassengerId", "Survived", "Pclass")
//                .addColumnsString("Name", "Sex")
//                .addColumnsInteger("Age", "SibSp", "Parch")
//                .addColumnString("Ticket")
//                .addColumnDouble("Fare")
//                .addColumnsString("Cabin", "Embarked")
//                .build()
//
//        log.info(schema.toString())
//
//        val transformProcess = TransformProcess.Builder(schema)
//                .conditionalReplaceValueTransform(
//                        "Age",
//                        IntWritable(getAverageAge(parsedStringData)),
//                        InvalidValueColumnCondition("Age")
//                )
//                .conditionalReplaceValueTransform(
//                        "Embarked",
//                        Text("C"),
//                        InvalidValueColumnCondition("Embarked")
//                )
//                .stringMapTransform("Sex", mapOf("male" to "0", "female" to "1"))
//                .stringMapTransform("Embarked", mapOf("C" to "0", "Q" to "1", "S" to "2", "" to "0"))
//                //.transform(MinMaxNormalizer("Age", 0.0, 100.0, 0.0, 1.0))
//                .transform(MinMaxNormalizer("Fare", 0.0, 95.0, 0.0, 1.0))
//                .removeColumns("Name", "PassengerId", "Ticket", "Cabin")
//                .build()
//
//
//
//
//        var stringData = sparkContext.textFile(File("data/train.csv").absolutePath)
//        stringData = stringData.filter { !it.contains("Survived") }
//        val rr = CSVRecordReader()
//        val parsedInputData = stringData.map(StringToWritablesFunction(rr))
//
//        val transformedData = SparkTransformExecutor.execute(parsedInputData, transformProcess)
//
//        val dataAnalysis = AnalyzeSpark.analyze(transformProcess.finalSchema, transformedData)
//
//        val transformedCsv = transformedData.collect().fold("",
//                { acc: String, row: List<Writable> ->
//                    acc + "\n" + row.fold("",
//                            { a: String, b: Writable -> a + ", " + b.toString() }).removeRange(0..1)
//                }
//        )
//        File("data/train-ugi.csv").writeText(transformedCsv)
//
//
//
//        println(dataAnalysis)
//
//        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, File("DataVecTitanicAnalysis.html"))

        val finalData = getStructuredData(parsedStringData)
                .map {
                    FinalData(it.pClass, it.isMale, it.fare, it.sibSp > 0 || it.parch > 0,
                            it.parch > 0, it.sibSp > 0, it.survived)
                }
                .map {
                    val isMale = if (it.isMale) "1" else "0"
                    val familyBool = if (it.familyBool) "1" else "0"
                    val parchBool = if (it.parchBool) "1" else "0"
                    val sibSpBool = if (it.sibSpBool) "1" else "0"
                    val survived = if (it.survived) "1" else "0"
                    "${it.pClass},$isMale,${it.fare},$familyBool,$parchBool,$sibSpBool,$survived"
                }

        Files.write(File("data/sedat-trained.csv").toPath(), finalData)
    }

    private fun getAverageAge(parsedStringData: JavaRDD<String>): Int {
        val ages = parsedStringData.cache().map { it.split(",")[6] }
                .map { it.toDoubleOrNull() }
                .filter { it != null }

        val rowCountWithAge = ages.count()

        return ages.reduce { v1, v2 -> v1!!.plus(v2!!) }!!.div(rowCountWithAge).roundToInt()
    }

    private fun getStructuredData(parsedStringData: JavaRDD<String>): List<InitialData> {
        return parsedStringData.cache()
                .map {
                    val name = it.substringBeforeLast("\"").substringAfter("\"")
                    val row = it.replace("\"$name\",", "")
                    val rowArray = row.split(",")
                    val initialData = InitialData(rowArray[0], rowArray[1], rowArray[2], name,
                            rowArray[3], rowArray[4], rowArray[5], rowArray[6],
                            rowArray[7], rowArray[8], rowArray[9], rowArray[10])
                    initialData
                }
                .collect()
    }
}