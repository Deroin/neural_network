package de.dero.neural_network.sample;

import de.dero.neural_network.Connection;
import de.dero.neural_network.Layer;
import de.dero.neural_network.NeuralNetwork;
import de.dero.neural_network.Neuron;

import java.util.Random;

public class DigitRecognition33px {

    public static NeuralNetwork constructNNDigitReco33px() {
        NeuralNetwork net = new NeuralNetwork();
        Random random = new Random();
        // create inputLayer for 33x33 pixels (= 1089 pixels)
        Layer inputLayer = new Layer();
        for (int i = 0; i < 1089; i++) {
            inputLayer.addNeuron(new Neuron());
        }

        /*
          create layer 1
          convolutional layer (6 feature maps)
          15x15 features with 5x5 convolutional kernel
          15*15*6 neurons = 1350 neurons
          (5*5+1)*6 weights = 156 weights
         */

        Layer layer1 = new Layer(inputLayer);
        for (int i = 0; i < 1350; i++) {
            layer1.addNeuron(new Neuron());
        }
        for (int i = 0; i < 156; i++) {
            layer1.addWeight(getRandomWeight(random));
        }

        /*
          Connect layer 1 to the inputLayer
          Use a 5x5 kernel which will be moved by 2 pixels
         */
        int[] kernelTemplate = new int[]{0, 1, 2, 3, 4, 33, 34, 35, 36, 37, 66, 67, 68, 69, 70, 99, 100, 101, 102,
                                         103, 132, 133, 134, 135, 136};

        for (int featureMap = 0; featureMap < 6; featureMap++) {
            int neuronOffset = 225 * featureMap;
            for (int i = 0; i < 15; i++) {
                for (int j = 0; j < 15; j++) {
                    int weightNum = featureMap * (kernelTemplate.length + 1);
                    Neuron n = layer1.getNeurons().get(15 * i + j + neuronOffset);
                    n.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++); //bias weight
                    int kernelOffset = 66 * i + 2 * j;
                    for (int kernelNode : kernelTemplate) {
                        n.addConnection(kernelOffset + kernelNode, weightNum++);
                    }
                }
            }
        }

        /*
          create layer 2
          convolutional layer (50 feature maps)
          6x6 features with 5x5 convolutional kernel of corresponding areas of the 6 feature maps of layer 1
          6*6*50 neurons = 1800 neurons
          (5*5*6+1)*50 weights = 7550 weights
         */
        Layer layer2 = new Layer(layer1);
        for (int i = 0; i < 1800; i++) {
            layer2.addNeuron(new Neuron());
        }
        for (int i = 0; i < 7550; i++) {
            layer2.addWeight(getRandomWeight(random));
        }

        /*
          Connect layer 2 to layer 1
          Use a 5x5 kernel which will be moved by 2 pixel-groups
         */
        kernelTemplate = new int[]{0, 1, 2, 3, 4,
                                   15, 16, 17, 18, 19,
                                   30, 31, 32, 34, 35,
                                   45, 46, 47, 48, 49,
                                   60, 61, 62, 63, 64};

        for (int featureMap = 0; featureMap < 50; featureMap++) {
            int neuronOffset = featureMap * 36;
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    int weightNum = featureMap * 26;
                    int weightOffset = 2 * j + 30 * i;
                    Neuron n = layer2.getNeurons().get(j + i * 6 + neuronOffset);
                    n.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++);
                    for (int k = 0; k < 25; k++) {
                        for (int l = 0; l < 6; l++) {
                            n.addConnection(l * 225 + weightOffset + kernelTemplate[k], weightNum++);
                        }
                    }
                }
            }
        }

        /*
          Create layer 3
          fully-connected layer of 100 neurons
          100 *(1800 + 1) weights = 180100 weights
         */
        Layer layer3 = new Layer(layer2);
        for (int i = 0; i < 100; i++) {
            layer3.addNeuron(new Neuron());
        }
        /*
          Connect layer 3 to layer 2
         */
        int weightNum = 0;
        for (Neuron n : layer3.getNeurons()) {
            layer3.addWeight(getRandomWeight(random));
            n.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++);
            for (int i = 0; i < 1800; i++) {
                layer3.addWeight(getRandomWeight(random));
                n.addConnection(i, weightNum++);
            }
        }

        /*
          create layer 4
          fully-connected layer of 10 neurons
          10 * (100 + 1) weights = 1010 weights
         */
        Layer layer4 = new Layer(layer3);
        for (int i = 0; i < 10; i++) {
            layer4.addNeuron(new Neuron());
        }
        /*
          Connect layer 4 to layer 3
         */
        weightNum = 0;
        for (Neuron n : layer4.getNeurons()) {
            layer4.addWeight(getRandomWeight(random));
            n.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++);
            for (int i = 0; i < 100; i++) {
                layer4.addWeight(getRandomWeight(random));
                n.addConnection(i, weightNum++);
            }
        }
        net.addLayer(inputLayer);
        net.addLayer(layer1);
        net.addLayer(layer2);
        net.addLayer(layer3);
        net.addLayer(layer4);

        return net;
    }

    private static double getRandomWeight(Random random) {
        return .05 * (2 * random.nextDouble() - 1);
    }
}
