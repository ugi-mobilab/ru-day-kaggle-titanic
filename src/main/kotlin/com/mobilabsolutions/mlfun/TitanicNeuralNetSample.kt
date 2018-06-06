package com.mobilabsolutions.mlfun

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage


class TitanicNeuralNetSample {

    private val log = LoggerFactory.getLogger(TitanicNeuralNetSample::class.java)

    companion object {
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            TitanicNeuralNetSample().train()
        }
    }

    val titanicDataSetIterator = TitanicDataSetProvider().trainDataSet

    val titanicNeuralNetworkConf = buildModelConfiguration()

    fun buildModelConfiguration() : MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(Nesterovs(0.01, 0.9))
                .list()
                .layer(0, DenseLayer.Builder()
                        .nIn(2)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .layer(1, DenseLayer.Builder()
                        .nIn(100)
                        .nOut(500)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .layer(2, DenseLayer.Builder()
                        .nIn(500)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .layer(3, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .pretrain(false)
                .backprop(true)
                .build()
    }

    fun train() {
        val model = MultiLayerNetwork(titanicNeuralNetworkConf)


        val uiServer = UIServer.getInstance()
        val statsStorage = InMemoryStatsStorage()
        uiServer.attach(statsStorage)

        model.setListeners(ScoreIterationListener(1), StatsListener(statsStorage))


        for (i in 0 .. 500) {
            model.fit(titanicDataSetIterator)
        }

        val eval = Evaluation(2)
        titanicDataSetIterator.reset()
        while (titanicDataSetIterator.hasNext()) {
            val next = titanicDataSetIterator.next()
            val output = model.output(next.features)
            eval.eval(next.getLabels(), output)
        }

        log.info(eval.stats())


    }


}