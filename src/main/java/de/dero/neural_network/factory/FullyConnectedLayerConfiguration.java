package de.dero.neural_network.factory;

import de.dero.neural_network.activation.ActivationFunction;

public class FullyConnectedLayerConfiguration extends LayerConfiguration {

    public FullyConnectedLayerConfiguration(int neuronCount,
                                            ActivationFunction activationFunction) {
        super(neuronCount, activationFunction);
    }
}
