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
import java.io.*;
import java.util.ArrayList;
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
    // Starting or continuing training.
    public Main fit() {
        File samplesDir = new File(params.samplesDirPathName);
        File[] samplesList = samplesDir.listFiles();
        int samplesCount = 0;
        if (samplesList != null) samplesCount = samplesList.length;
        if (samplesCount <= params.minSamplesCount) {
            System.out.println("Samples count is too small.");
            return this;
        }

        File labelsDir = new File(params.labelsDirPathName);
        File[] labelsList = labelsDir.listFiles();
        int labelsCount = 0;
        if (labelsList != null) labelsCount = labelsList.length;
        if (labelsCount != samplesCount) {
            System.out.println("Labels count is invalid.");
            return this;
        }

        try {
            // TODO: odczytywanie wyuczonego modelu z dysku i opcjonalnie dotrenowywanie, kontynuacja treningu
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
            log.info("The last train sample number " + lastTrainSample + ":");
            log.info(String.valueOf(trainData.next()));
            trainData.reset();

            log.info("The first test sample number " + (lastTrainSample + 1) + ":");
            log.info(String.valueOf(testData.next()));
            testData.reset();

            // Wczytywanie modelu.
            // https://deeplearning4j.konduit.ai/deeplearning4j/reference/saving-and-loading-models
            MultiLayerNetwork model = loadMultiLayerNetwork();

            // Inicjalizacja modelu.
            log.info("The training process begins...");
            if (model == null) {
                // Konfiguracja modelu.
                // MSE -> Mean Squared Error, średni kwadrat błędu -> http://localhost:63342/inna/nd4j-api-1.0.0-M2.1-javadoc.jar/org/nd4j/linalg/lossfunctions/impl/LossMSE.html
                // IDENTITY -> f(x) = x -> https://javadoc.io/doc/org.nd4j/nd4j-api/latest/org/nd4j/linalg/activations/impl/ActivationIdentity.html
                int hiddenLayerWidth = 200;
                MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nadam())
                        // .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                        // .gradientNormalizationThreshold(0.5)
                        .list()
                        .layer(new LSTM.Builder().activation(Activation.TANH).nIn(4).nOut(hiddenLayerWidth).build())
                        .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hiddenLayerWidth).nOut(1).build())
                        .build();

                model = new MultiLayerNetwork(configuration);
                log.info("The multi layer network has been created.");
            }
            model.init();

            // Nasłuchiwanie — każda próbka w zestawie treningowym to jest jedna iteracja.
            model.setListeners(new ScoreIterationListener(10));

            // Trenowanie — cały zestaw danych treningowych to jest jedna epoka.
            int nEpochs = 10;
            for (int i = 0; i < nEpochs; i++) {
                log.info("Training epoch {}.", i);
                model.fit(trainData);
                evaluate_and_save(model, testData, i);
                // Tutaj można dodać dodatkowe kryterium stopu.
            }
        } catch (IOException | InterruptedException e) {
            System.out.println(e.getLocalizedMessage());
        }

        return this;
    }

    protected void evaluate_and_save(MultiLayerNetwork model, DataSetIterator data, int epoch) {
        // Ocena
        // https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/evaluation#evaluation-for-regression
        log.info("Evaluating epoch {}.", epoch);
        log.info("Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, Relative Squared Error, and R^2 Coefficient of Determination");
        RegressionEvaluation eval = model.evaluateRegression(data);
        log.info(eval.stats());

        File evalFile = new File(params.evalFileName);
        if (evalFile.exists()) {
            Double savedEval = readSavedEvaluation(evalFile);
            // Zapisz, jeśli aktualny średni kwadrat błędu jest mniejszy niż poprzedni.
            if (Double.compare(eval.averageMeanSquaredError(), savedEval) < 0) {
                log.info("Replacing model to {} and {}.", params.modelFileName, params.evalFileName);
                try {
                    model.save(new File(params.modelFileName));
                    saveEvaluation(params.evalFileName, eval.averageMeanSquaredError());
                } catch (IOException e) {
                    System.out.println(e.getLocalizedMessage());
                }
            }
        } else {
            try {
                log.info("Saving model to {} and {}.", params.modelFileName, params.evalFileName);
                model.save(new File(params.modelFileName));
                saveEvaluation(params.evalFileName, eval.averageMeanSquaredError());
            } catch (IOException e) {
                System.out.println(e.getLocalizedMessage());
            }
        }
    }

    static Double readSavedEvaluation(File file) {
        Double answer = null;
        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(file));
            String line = reader.readLine();
            if (line != null) answer = Double.valueOf(line);
            reader.close();
        } catch (IOException e) {
            System.out.println(e.getLocalizedMessage());
        }
        return answer;
    }

    static void saveEvaluation(String file, double evaluation) throws IOException {
        BufferedWriter writer;
        writer = new BufferedWriter(new FileWriter(file, false));
        writer.append(String.valueOf(evaluation));
        writer.newLine();
        writer.close();
    }

    protected MultiLayerNetwork loadMultiLayerNetwork() {
        MultiLayerNetwork answer = null;
        File modelFile = new File(params.modelFileName);
        if (modelFile.exists()) {
            try {
                answer = MultiLayerNetwork.load(modelFile, true);
                log.info("The multi layer network has been loaded.");
            } catch (IOException e) {
                System.out.println(e.getLocalizedMessage());
            }
        }
        return answer;
    }
}
