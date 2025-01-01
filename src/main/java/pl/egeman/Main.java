package pl.egeman;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.autodiff.listeners.records.EvaluationRecord;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import org.nd4j.linalg.dataset.DataSet;

/*
https://javadoc.io/doc/org.deeplearning4j/deeplearning4j-nn/latest/org/deeplearning4j/nn/multilayer/package-summary.html
https://javadoc.io/doc/org.deeplearning4j/deeplearning4j-nn/1.0.0-M2/org/deeplearning4j/nn/conf/NeuralNetConfiguration.Builder.html
https://community.konduit.ai/t/lstm-regression-example/1746/10
*/

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    private final MainParameters params;
    public static Main create(MainParameters params) {
        return new Main(params);
    }
    public Main(MainParameters params) {
        this.params = params;
    }

    @SuppressWarnings("UnusedReturnValue")
    public Main run(){
        File samplesDir = new File(params.samplesDirPathName);
        File[] samplesList = samplesDir.listFiles();
        int samplesCount = 0;
        if (samplesList != null) samplesCount = samplesList.length;
        if(samplesCount <= params.minSamplesCount) {
            System.out.println("Samples count is too small.");
            return this;
        }

        File labelsDir = new File(params.labelsDirPathName);
        File[] labelsList = labelsDir.listFiles();
        int labelsCount = 0;
        if (labelsList != null) labelsCount = labelsList.length;
        if(labelsCount != samplesCount) {
            System.out.println("Labels count is invalid.");
            return this;
        }

        try {
            // TODO: zapisywanie modelu na dysku w każdej epoce
            // TODO: odczytywanie wyuczonego modelu z dysku i opcjonalnie dotrenowywanie, kontynuacja treningu
            // TODO: losowy podział na trzy zestawy próbek: uczące, sprawdzające i weryfikacyjne(?)
            // TODO: użycie próbek weryfikacyjnych w celu przerwania treningu(?)
            // TOREAD: https://deeplearning4j.konduit.ai/v/en-1.0.0-beta6/models/recurrent

            int miniBatchSize = 1;
            int numPassibleLabels = -1;
            boolean regression = true;

            int lastTrainSample = (samplesCount * 3) / 4;
            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
            trainFeatures.initialize(new NumberedFileInputSplit(samplesDir.getAbsolutePath() + "/%d.csv", 0, lastTrainSample));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, lastTrainSample));
            DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numPassibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
            testFeatures.initialize(new NumberedFileInputSplit(samplesDir.getAbsolutePath() + "/%d.csv", lastTrainSample + 1, samplesCount - 1));
            SequenceRecordReader testLabels = new CSVSequenceRecordReader();
            testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", lastTrainSample + 1, samplesCount - 1));
            DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numPassibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            trainData.next(lastTrainSample);
            log.info("The last train sample number " + String.valueOf(lastTrainSample) + ":");
            log.info(String.valueOf(trainData.next()));
            trainData.reset();

            log.info("The first test sample number " + String.valueOf(lastTrainSample + 1) + ":");
            log.info(String.valueOf(testData.next()));
            testData.reset();

            int hiddenLayerWidth = 30;
            MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                    .seed(12345)    //Random number generator seed for improved repeatability. Optional.
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nadam())
                    // .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                    // .gradientNormalizationThreshold(0.5)
                    .list()
                    .layer(new LSTM.Builder().activation(Activation.TANH).nIn(4).nOut(hiddenLayerWidth).build())
                    // MSE - Mean Squared Error, średni kwadrat błędu, http://localhost:63342/inna/nd4j-api-1.0.0-M2.1-javadoc.jar/org/nd4j/linalg/lossfunctions/impl/LossMSE.html
                    .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hiddenLayerWidth).nOut(1).build())
                    .build();

            // Inicjalizacja modelu.
            log.info("The training process begins...");
            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();

            // Nasłuchiwanie — każda próbka w zestawie treningowym to jest jedna iteracja.
            // model.setListeners(new ScoreIterationListener(1), new EvaluativeListener(testData, 1, InvocationType.EPOCH_END));
            model.setListeners(new ScoreIterationListener(50));

            // Trenowanie — cały zestaw danych treningowych to jest jedna epoka.
            int nEpochs = 1;
            for (int i = 0; i < nEpochs; i++) {
                log.info("Training epoch {}.", i);
                model.fit(trainData);

                // log.info("Evaluating epoch {}.", i);
                // RegressionEvaluation eval = model.evaluateRegression(testData);
                // eval.eval(ds.getLabels(), output);
                // log.info(eval.stats());

                DataSet ds = testData.next();
                INDArray output = model.output(ds.getFeatures(), false);
                log.info("Test data features:\n{}", ds.getFeatures());
                log.info("Model output:\n{}", output);
                log.info("Test data labels:\n{}.", ds.getLabels());
                testData.reset();
            }
            model.save(new File(params.modelFileName));
        } catch (IOException | InterruptedException e) {
            System.out.println(e.getLocalizedMessage());
        }

        return this;
    }
}
