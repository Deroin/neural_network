package de.dero.neural_network.factory.layer;

import de.dero.neural_network.Connection;
import de.dero.neural_network.Layer;
import de.dero.neural_network.Neuron;
import de.dero.neural_network.factory.FullyConnectedLayerConfiguration;

public class FullyConnectedLayerFactory extends LayerFactory<FullyConnectedLayerConfiguration> {

    @Override
    public Layer createLayer(FullyConnectedLayerConfiguration configuration, Layer previousLayer) {
        Layer layer = new Layer(previousLayer, configuration.getActivationFunction());

        int weightNum = 0;
        for (int i = 0; i < configuration.getNeuronCount(); ++i) {
            Neuron neuron = new Neuron();
            layer.addNeuron(neuron);

            layer.addWeight(getRandomWeight());
            neuron.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++);

            for (int j = 0; j < previousLayer.getNeurons().size(); ++j) {
                layer.addWeight(getRandomWeight());
                neuron.addConnection(j, weightNum++);
            }
        }

        return layer;
    }
}
