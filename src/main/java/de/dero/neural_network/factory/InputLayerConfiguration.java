package de.dero.neural_network.factory;

import de.dero.neural_network.activation.ActivationFunction;

public class InputLayerConfiguration extends LayerConfiguration {

    protected InputLayerConfiguration(int neuronCount,
                                      ActivationFunction activationFunction) {
        super(neuronCount, activationFunction);
    }
}
