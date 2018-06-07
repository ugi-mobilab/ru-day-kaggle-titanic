package com.mobilabsolutions.mlfun

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.transform.ui.HtmlAnalysis
import sun.management.MemoryUsageCompositeData.getMax
import org.datavec.api.transform.analysis.columns.DoubleAnalysis
import org.datavec.api.transform.analysis.DataAnalysis
import org.datavec.api.writable.Writable
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.IntegerColumnCondition
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Text
import org.datavec.spark.transform.AnalyzeSpark
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.slf4j.LoggerFactory
import java.io.File


class DataAnalysisSample {

    private val log = LoggerFactory.getLogger(DataAnalysisSample::class.java)

    companion object {
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            DataAnalysisSample().analyze()
        }
    }


    fun analyze() {
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
                        IntWritable(0),
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
}