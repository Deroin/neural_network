package de.dero.neural_network;

import java.util.LinkedList;
import java.util.List;

public class Neuron {
    private LinkedList<Connection> inputConnections = new LinkedList<>();
    private double output;

    public void addConnection(int neuronIndex, int weightIndex) {
        addConnection(new Connection(neuronIndex, weightIndex));
    }

    private void addConnection(Connection connection) {
        inputConnections.add(connection);
    }

    public List<Connection> getInputConnections() {
        return inputConnections;
    }

    double getOutput() {
        return output;
    }

    void setOutput(double output) {
        this.output = output;
    }
}
