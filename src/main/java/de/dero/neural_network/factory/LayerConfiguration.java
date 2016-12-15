package de.dero.neural_network.factory;

import de.dero.neural_network.activation.ActivationFunction;

public abstract class LayerConfiguration {

    int neuronCount;
    ActivationFunction activationFunction;

    protected LayerConfiguration(int neuronCount, ActivationFunction activationFunction) {
        this.neuronCount = neuronCount;
        this.activationFunction = activationFunction;
    }

    public int getNeuronCount() {
        return neuronCount;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
