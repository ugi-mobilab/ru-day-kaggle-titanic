package com.mobilabsolutions.mlfun

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.io.ClassPathResource
import java.io.File
import javax.xml.crypto.Data

class TitanicDataSetProvider  {

    companion object {
        @Throws(Exception::class)
        @JvmStatic fun main(args: Array<String>) {
            TitanicDataSetProvider().check()
        }
    }

    val trainDataSet : DataSetIterator
    var testData : Map<Int, INDArray> = HashMap()

    init {

        trainDataSet = getDataSet(File("data/train-ugi.csv"), 891, 0, 2)
        val transformator = DataAnalysisSample()
        transformator.transform()
        testData = transformator.testData


    }

    fun getDataSet(file : File, numberOfEntries : Int, labelIndex : Int, numberOfClasses : Int) : DataSetIterator {
        val recordParser = CSVRecordReader(1, ',')
        recordParser.initialize(FileSplit(file))
        val dataSetIterator = RecordReaderDataSetIterator(recordParser, numberOfEntries, labelIndex, numberOfClasses)
        return dataSetIterator
    }

    fun check() {
        System.out.println(trainDataSet.next().toString());
    }



}