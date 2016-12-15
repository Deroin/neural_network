package de.dero.neural_network.factory.layer;

import de.dero.neural_network.Layer;
import de.dero.neural_network.Neuron;
import de.dero.neural_network.factory.InputLayerConfiguration;

public class InputLayerFactory extends LayerFactory<InputLayerConfiguration> {

    @Override
    public Layer createLayer(InputLayerConfiguration configuration, Layer previousLayer) {
        Layer layer = new Layer(null, configuration.getActivationFunction());
        for (int i = 0; i < configuration.getNeuronCount(); i++) {
            layer.addNeuron(new Neuron());
        }
        return layer;
    }
}
