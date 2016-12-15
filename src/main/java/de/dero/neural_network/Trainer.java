package de.dero.neural_network;

import org.apache.log4j.Logger;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

public class Trainer implements Runnable {

    private static Logger log = Logger.getLogger(Trainer.class);

    private TrainingSet trainingSet;
    private NeuralNetwork network;
    private double initialEta;
    private double minimalEta;
    private double etaDecay;
    private int decayAfterBackPropagation;
    private int maxEpochs;
    private LinkedList<Layer> layers;
    private int sampleSize = 500;

    private boolean stop;

    public void setSampleSize(int sampleSize) {
        this.sampleSize = sampleSize;
    }

    public void setTrainingSet(TrainingSet trainingSet) {
        this.trainingSet = trainingSet;
    }

    public void setNetwork(NeuralNetwork network) {
        this.network = network;
    }

    public void setInitialEta(double initialEta) {
        this.initialEta = initialEta;
    }

    public void setMinimalEta(double minimalEta) {
        this.minimalEta = minimalEta;
    }

    public void setEtaDecay(double etaDecay) {
        this.etaDecay = etaDecay;
    }

    public void setDecayAfterBackPropagation(int decayAfterBackPropagation) {
        this.decayAfterBackPropagation = decayAfterBackPropagation;
    }

    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    public void invokeStop() {
        stop = true;
    }

    @Override
    public void run() {
        layers = network.getLayers();
        stop = false;

        Random random = new Random();
        long epochStart = -1;
        long iterationsStart = -1;
        int backPropagationCount = 0;
        double eta = initialEta;

        while (!stop) {
            //new epoch stated
            int patternIndex = backPropagationCount % trainingSet.size();
            if (patternIndex == 0) {
                int epochNum = backPropagationCount / trainingSet.size() + 1;
                if (epochStart != -1) {
                    log.info("Epoch duration: " + (System.currentTimeMillis() - epochStart));
                }
                if (epochNum > maxEpochs) {
                    break;
                }
                log.info("Starting epoch " + epochNum);
                epochStart = System.currentTimeMillis();
                log.info("Shuffling data");
                TrainingSet set = new TrainingSet(trainingSet);
                set.randomize(random);
                log.info("Calculation hessian");
                calculateHessian(random);
                log.info("Starting back propagation");
            }
            if (backPropagationCount % decayAfterBackPropagation == 0) {
                eta = Math.min(eta * etaDecay, minimalEta);
            }
            if (backPropagationCount % 5000 == 0) {
                if (iterationsStart != -1) {
                    log.info("        5000 iterations duration: "
                             + (System.currentTimeMillis() - iterationsStart));
                }
                log.info("    Starting back propagation "
                         + backPropagationCount
                         + " of "
                         + trainingSet.size());
                iterationsStart = System.currentTimeMillis();
            }

            Vector<Double> pattern = trainingSet.getData(patternIndex);
            Vector<Double> desiredOutput = trainingSet.getDesiredResult(patternIndex);
            Vector<Double> actualOutput = network.calculate(pattern);

            backPropagate(actualOutput, desiredOutput, eta);

            backPropagationCount++;
        }
    }

    private void calculateHessian(Random random) {
        layers.stream().filter(layer -> layer.getWeights() != null).forEach(layer -> {
            for (Weight weight : layer.getWeights()) {
                weight.hessian = 0;
            }
        });

        for (int i = 0; i < sampleSize; i++) {
            int index = random.nextInt(trainingSet.size());
            Vector<Double> pattern = trainingSet.getData(index);
            Vector<Double> desiredOutput = trainingSet.getDesiredResult(index);
            Vector<Double> actualOutput = network.calculate(pattern);
            backPropagateSecondDerivatives(actualOutput, desiredOutput);
        }

        layers.stream().filter(layer -> layer.getWeights() != null).forEach(layer -> {
            for (Weight weight : layer.getWeights()) {
                if (weight.hessian < 0) {
                    weight.hessian = 0;
                }
                weight.hessian /= (double) sampleSize;
            }
        });
    }

    private void backPropagate(Vector<Double> actualOutput, Vector<Double> desiredOutput, double learningRate) {
        assert actualOutput != null;
        assert desiredOutput != null;
        assert layers.size() >= 2;
        assert actualOutput.size() == desiredOutput.size();

        double[][] differentials = new double[layers.size()][];
        for (int i = 0; i < layers.size(); i++) {
            differentials[i] = new double[layers.get(i).getNeurons().size()];
        }
        double[] lastLayerErrorDifferential = differentials[differentials.length - 1]; // contains d E^P / d x for
        // the last layer (equation 2)

        Layer currentLayer = layers.get(layers.size() - 1);
        List<Neuron> lastLayerNeurons = currentLayer.getNeurons();
        for (int i = 0; i < lastLayerNeurons.size(); i++) {
            lastLayerErrorDifferential[i] = actualOutput.get(i) - desiredOutput.get(i);
        }

        for (int i = layers.size() - 1; i > 0; i--) {
            currentLayer.backPropagate(differentials[i], differentials[i - 1], learningRate);
            currentLayer = currentLayer.previousLayer;
        }
    }

    private void backPropagateSecondDerivatives(Vector<Double> actualOutput, Vector<Double> desiredOutput) {
        assert actualOutput != null;
        assert desiredOutput != null;
        assert layers.size() >= 2;
        assert actualOutput.size() == desiredOutput.size();

        double[][] differentials = new double[layers.size()][];
        for (int i = 0; i < layers.size(); i++) {
            differentials[i] = new double[layers.get(i).getNeurons().size()];
        }
        double[] lastLayerErrorSecondDifferential = differentials[differentials.length - 1];
        for (int i = 0; i < lastLayerErrorSecondDifferential.length; i++) {
            lastLayerErrorSecondDifferential[i] = 1;
        }
        Layer currentLayer = layers.getLast();
        for (int i = layers.size() - 1; i > 0; i--) {
            currentLayer.backPropagateSecondDerivative(differentials[i], differentials[i - 1]);
            currentLayer = currentLayer.previousLayer;
        }
    }
}
