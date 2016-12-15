package de.dero.neural_network;

public class Connection {
    public static final int BIAS_NEURON_INDEX = Integer.MAX_VALUE;
    private int neuronIndex;
    private int weightIndex;

    public Connection(int neuronIndex, int weightIndex) {
        this.neuronIndex = neuronIndex;
        this.weightIndex = weightIndex;
    }

    public int getNeuronIndex() {
        return neuronIndex;
    }

    public int getWeightIndex() {
        return weightIndex;
    }
}
