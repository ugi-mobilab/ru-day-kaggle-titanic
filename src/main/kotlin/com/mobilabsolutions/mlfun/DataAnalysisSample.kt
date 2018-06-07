package com.mobilabsolutions.mlfun

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Text
import org.datavec.spark.transform.AnalyzeSpark
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
        @JvmStatic
        fun main(args: Array<String>) {
            DataAnalysisSample().analyze()
        }
    }

    private fun analyze() {
        val sparkContext = JavaSparkContext(conf)

        val parsedStringData = sparkContext.textFile(File("data/sedat-train.csv").absolutePath)

        val schema = Schema.Builder()
                .addColumnsInteger("PassengerId", "Survived", "Pclass")
                .addColumnsString("Name", "Sex")
                .addColumnsInteger("Age", "SibSp", "Parch")
                .addColumnString("Ticket")
                .addColumnDouble("Fare")
                .addColumnsString("Cabin", "Embarked")
                .build()

        log.info(schema.toString())

        val transformProcess = TransformProcess.Builder(schema)
                .conditionalReplaceValueTransform(
                        "Age",
                        IntWritable(getAverageAge(parsedStringData)),
                        InvalidValueColumnCondition("Age")
                )
                .conditionalReplaceValueTransform(
                        "Embarked",
                        Text("C"),
                        InvalidValueColumnCondition("Embarked")
                )
                .stringMapTransform("Sex", mapOf("male" to "0", "female" to "1"))
                .stringMapTransform("Embarked", mapOf("C" to "0", "Q" to "1", "S" to "2", "" to "0"))
                .removeColumns("Name", "PassengerId", "Ticket", "Cabin")
                .build()


        val conf = SparkConf()
        conf.setMaster("local[*]")
        conf.setAppName("DataVec Example")

        val sc = JavaSparkContext(conf)

        var stringData = sc.textFile(File("data/train.csv").absolutePath)
        stringData = stringData.filter { !it.contains("Survived") }
        val rr = CSVRecordReader()
        val parsedInputData = stringData.map(StringToWritablesFunction(rr))

        val transformedData = SparkTransformExecutor.execute(parsedInputData, transformProcess)

        val dataAnalysis = AnalyzeSpark.analyze(transformProcess.finalSchema, transformedData)

        val transformedCsv = transformedData.collect().fold("",
                { acc: String, row: List<Writable> ->
                    acc + "\n" + row.fold("",
                            { a: String, b: Writable -> a + ", " + b.toString() }).removeRange(0..1)
                }
        )
        File("data/train-ugi.csv").writeText(transformedCsv)



        println(dataAnalysis)

        //We can get statistics on a per-column basis:
//        val da = dataAnalysis.getColumnAnalysis("Sepal length") as DoubleAnalysis
//        val minValue = da.min
//        val maxValue = da.max
//        val mean = da.mean

        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, File("DataVecTitanicAnalysis.html"))

    }

    private fun getAverageAge(parsedStringData: JavaRDD<String>): Int {
        val ages = parsedStringData.cache().map { it.split(",")[6] }
                .map { it.toDoubleOrNull() }
                .filter { it != null }

        val rowCountWithAge = ages.count()

        return ages.reduce { v1, v2 -> v1!!.plus(v2!!) }!!.div(rowCountWithAge).roundToInt()
    }
}