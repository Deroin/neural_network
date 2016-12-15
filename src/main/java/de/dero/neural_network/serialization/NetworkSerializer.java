package de.dero.neural_network.serialization;

import de.dero.neural_network.NeuralNetwork;

public interface NetworkSerializer {
    String fromNetwork(NeuralNetwork network);
}
