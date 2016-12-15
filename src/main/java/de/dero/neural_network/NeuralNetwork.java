package de.dero.neural_network;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 1;

    private LinkedList<Layer> layers = new LinkedList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public LinkedList<Layer> getLayers() {
        return layers;
    }

    public Vector<Double> calculate(Vector<Double> input) {
        Iterator<Layer> itr = layers.iterator();
        Layer currentLayer = itr.next();
        List<Neuron> neurons = currentLayer.getNeurons();
        assert neurons.size() == input.size();
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.setOutput(input.elementAt(i));
        }

        while (itr.hasNext()) {
            currentLayer = itr.next();
            currentLayer.calculate();
        }
        List<Neuron> outputNeurons = currentLayer.getNeurons();
        Vector<Double> output = new Vector<>(outputNeurons.size());
        output.addAll(
            outputNeurons.stream()
                         .map(Neuron::getOutput)
                         .collect(Collectors.toList())
        );
        return output;
    }

}
