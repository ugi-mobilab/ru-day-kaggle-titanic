package com.mobilabsolutions.mlfun

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition
import org.datavec.api.transform.schema.Schema
import org.datavec.api.transform.transform.doubletransform.MinMaxNormalizer
import org.datavec.api.transform.ui.HtmlAnalysis
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Text
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.AnalyzeSpark
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*
import kotlin.collections.HashMap
import kotlin.math.roundToInt


class DataAnalysisSample {

    val conf = SparkConf()
    var testData : Map<Int, INDArray> = HashMap()

    init {
        conf.setMaster("local[*]")
        conf.setAppName("DataVec Example")
    }

    companion object {
        private val log = LoggerFactory.getLogger(DataAnalysisSample::class.java)

        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            DataAnalysisSample().transform()
        }
    }

    fun getSchema(trainSet: Boolean): Schema {
        val schema = Schema.Builder()

        if (trainSet) {
            schema.addColumnsInteger("PassengerId", "Survived", "Pclass")
        } else {
            schema.addColumnsInteger("PassengerId", "Pclass")
        }
        schema.addColumnsString("Name", "Sex")
                .addColumnsInteger("Age", "SibSp", "Parch")
                .addColumnString("Ticket")
                .addColumnDouble("Fare")
                .addColumnsString("Cabin", "Embarked")
                .build()

        return schema.build()

    }


    fun getTransformationBuidler(trainSet: Boolean, averageAge: Int): TransformProcess.Builder {
        val transformProcess = TransformProcess.Builder(getSchema(trainSet))
                .conditionalReplaceValueTransform(
                        "Age",
                        IntWritable(averageAge),
                        InvalidValueColumnCondition("Age")
                )
                .conditionalReplaceValueTransform(
                        "Fare",
                        IntWritable(35),
                        InvalidValueColumnCondition("Fare")
                )
                .conditionalReplaceValueTransform(
                        "Embarked",
                        Text("C"),
                        InvalidValueColumnCondition("Embarked")
                )
                .stringMapTransform("Sex", mapOf("male" to "0", "female" to "1"))
                .stringMapTransform("Embarked", mapOf("C" to "0", "Q" to "1", "S" to "2", "" to "0"))
                .transform(MinMaxNormalizer("Age", 0.0, 100.0, 0.0, 1.0))
                .transform(MinMaxNormalizer("Fare", 0.0, 95.0, 0.0, 1.0))
        if (trainSet) {
            transformProcess.removeColumns("Name", "PassengerId", "Ticket", "Cabin")
        } else {
            transformProcess.removeColumns("Name", "Ticket", "Cabin")
        }
        return transformProcess

    }

    public fun transform() {
        val sparkContext = JavaSparkContext(conf)

        val parsedStringData = sparkContext.textFile(File("data/train.csv").absolutePath)

        val schema = getSchema(true)
        log.info(schema.toString())

        val transformProcess = getTransformationBuidler(true, getAverageAge(parsedStringData))

                .build()


        var stringData = sparkContext.textFile(File("data/train.csv").absolutePath)
        stringData = stringData.filter { !it.contains("Survived") }
        val recordReader = CSVRecordReader()
        val parsedInputData = stringData.map(StringToWritablesFunction(recordReader))

        val transformedData = SparkTransformExecutor.execute(parsedInputData, transformProcess)

        val dataAnalysis = AnalyzeSpark.analyze(transformProcess.finalSchema, transformedData)

        val transformedCsv = transformedData.collect().fold("",
                { acc: String, row: List<Writable> ->
                    acc + "\n" + row.fold("",
                            { a: String, b: Writable -> a + ", " + b.toString() }).removeRange(0..1)
                }
        )
        File("data/train-ugi.csv").writeText(transformedCsv)

        val testTransformProcess = getTransformationBuidler(false, getAverageAge(parsedStringData)).build()
        var testStringData = sparkContext.textFile(File("data/test.csv").absolutePath)
        testStringData = testStringData.filter { !it.contains("Fare") }
        val testRecordReader = CSVRecordReader()

        val testParsedInputData = testStringData.map(StringToWritablesFunction(testRecordReader))
        val testTransformedData = SparkTransformExecutor.execute(testParsedInputData, testTransformProcess)
        val testTransformedCsv = testTransformedData.collect().fold("",
                { acc: String, row: List<Writable> ->
                    acc + "\n" + row.fold("",
                            { a: String, b: Writable -> a + ", " + b.toString() }).removeRange(0..1)
                }
        )
        File("data/test-ugi.csv").writeText(testTransformedCsv)

        val testMap : MutableMap<Int, INDArray> = HashMap()

        testTransformedData.foreach {
            row ->
            val passangerId = row.get(0).toString().toInt()
            val input = Nd4j.zeros(row.size - 1)
            for (i in 1 until row.size - 1) {
                input.putScalar(i, row.get(i).toFloat())
            }
            testMap.put(passangerId, input)
        }

        testData = testMap



        println(dataAnalysis)

        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, File("DataVecTitanicAnalysis.html"))

    }

    private fun getAverageAge(parsedStringData: JavaRDD<String>): Int {
        val ages = parsedStringData.cache().map { it.split(",")[6] }
                .map { it.toDoubleOrNull() }
                .filter { it != null }

        val rowCountWithAge = ages.count()

        return ages.reduce { v1, v2 -> v1!!.plus(v2!!) }!!.div(rowCountWithAge).roundToInt()
    }

    fun prepareTestData() {

    }
}