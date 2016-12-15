package de.dero.neural_network.sample;

import de.dero.neural_network.NeuralNetwork;
import de.dero.neural_network.activation.ActivationFunction;
import de.dero.neural_network.activation.HyperbolicTangent;
import de.dero.neural_network.factory.Convolutional2DLayerConfiguration;
import de.dero.neural_network.factory.FullyConnectedLayerConfiguration;
import de.dero.neural_network.factory.NetworkFactory;

import java.util.Random;

public class DigitRecognition33pxV2 {

    public static NeuralNetwork constructNNDigitReco33px() {
        ActivationFunction activationFunction = new HyperbolicTangent();

        int[] kernelTemplate = new int[]{0, 1, 2, 3, 4,
                                         33, 34, 35, 36, 37,
                                         66, 67, 68, 69, 70,
                                         99, 100, 101, 102, 103,
                                         132, 133, 134, 135, 136};
        Convolutional2DLayerConfiguration layer1Configuration = new Convolutional2DLayerConfiguration(
            6,
            33,
            33,
            5,
            5,
            2,
            2,
            activationFunction,
            kernelTemplate,
            null
        );

        kernelTemplate = new int[]{0, 1, 2, 3, 4,
                                   15, 16, 17, 18, 19,
                                   30, 31, 32, 34, 35,
                                   45, 46, 47, 48, 49,
                                   60, 61, 62, 63, 64};
        Convolutional2DLayerConfiguration layer2Configuration = new Convolutional2DLayerConfiguration(
            50,
            15,
            15,
            5,
            5,
            2,
            2,
            activationFunction,
            kernelTemplate,
            layer1Configuration
        );

        NetworkFactory factory = new NetworkFactory();

        factory.setInputLayer(1089, activationFunction);
        factory.addLayer(layer1Configuration);
        factory.addLayer(layer2Configuration);
        factory.addLayer(new FullyConnectedLayerConfiguration(100, activationFunction));
        factory.addLayer(new FullyConnectedLayerConfiguration(10, activationFunction));

        return factory.construct();
    }

}
