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
import org.datavec.spark.transform.AnalyzeSpark
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.slf4j.LoggerFactory
import java.io.File


class DataAnalysisSample {

    private val log = LoggerFactory.getLogger(DataAnalysisSample::class.java)

    companion object {
        @Throws(Exception::class)
        @JvmStatic fun main(args: Array<String>) {
            DataAnalysisSample().analyze()
        }
    }


    fun analyze() {
        val schema = Schema.Builder()
                .addColumnsInteger("PassangerId", "Survived", "Pclass")
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
                .build()
//                .conditional("Age", IntWritable())




        val conf = SparkConf()
        conf.setMaster("local[*]")
        conf.setAppName("DataVec Example")

        val sc = JavaSparkContext(conf)

        var stringData = sc.textFile(File("data/train.csv").absolutePath)
        log.info(stringData.fold("After loading: ", {first,second -> first + "\n" + second}))
        stringData = stringData.filter { !it.contains("PassengerId") }
        stringData = stringData.filter { !it.contains(",,") }
        log.info(stringData.fold("After processing: ", {first,second -> first + "\n" + second}))
        //We first need to parse this comma-delimited (CSV) format; we can do this using CSVRecordReader:
        val rr = CSVRecordReader()
        val parsedInputData = stringData.map(StringToWritablesFunction(rr))

        val transformedData = SparkTransformExecutor.execute(parsedInputData, transformProcess)

        val maxHistogramBuckets = 10
        val dataAnalysis = AnalyzeSpark.analyze(schema, transformedData)

        println(dataAnalysis)

        //We can get statistics on a per-column basis:
//        val da = dataAnalysis.getColumnAnalysis("Sepal length") as DoubleAnalysis
//        val minValue = da.min
//        val maxValue = da.max
//        val mean = da.mean

        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, File("DataVecTitanicAnalysis.html"))
    }
}