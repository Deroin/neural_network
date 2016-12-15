package de.dero.neural_network.factory.layer;

import de.dero.neural_network.Connection;
import de.dero.neural_network.Layer;
import de.dero.neural_network.Neuron;
import de.dero.neural_network.factory.ConvolutionalLayerConfiguration;

public class ConvolutionalLayerFactory<T extends ConvolutionalLayerConfiguration> extends LayerFactory<T> {

    @Override
    public Layer createLayer(ConvolutionalLayerConfiguration configuration, Layer previousLayer) {
        Layer layer = new Layer(previousLayer, configuration.getActivationFunction());

        addNeurons(configuration, layer);
        addWeights(configuration, layer);

        connectFeatureMap(configuration, layer);

        return layer;

    }

    protected void connectFeatureMap(ConvolutionalLayerConfiguration configuration, Layer layer) {
        int featureMapCount = configuration.getFeatureMapCount();
        int featureCount = configuration.getFeatureCount();
        int[] kernel = configuration.getConnectionKernel();
        int kernelLength = kernel.length;
        int offset = configuration.getKernelOffset();

        for(int featureMap = 0; featureMap < featureMapCount; featureMap++) {
            int neuronOffset = featureMap * featureCount;
            int kernelOffset = 0;
            for (int feature = 0; feature < featureCount; feature++) {
                int weightNum = featureMap * kernelLength + 1;
                Neuron n = layer.getNeurons().get(feature + neuronOffset);
                n.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++); //bias weight
                for (int kernelNode : kernel) {
                    n.addConnection(kernelNode + offset, weightNum++);
                }
                kernelOffset += kernelOffset;
            }
        }
    }

    private void addWeights(ConvolutionalLayerConfiguration configuration, Layer layer) {
        int weightCount = (configuration.getConnectionKernel().length);
        if (configuration.getPreviousLayerConfiguration() instanceof ConvolutionalLayerConfiguration) {
            weightCount *=
                ((ConvolutionalLayerConfiguration) configuration.getPreviousLayerConfiguration()).getFeatureMapCount();
        }
        weightCount = (weightCount + 1) * configuration.getFeatureMapCount();

        for(int i = 0; i < weightCount; ++i) {
            layer.addWeight(getRandomWeight());
        }
    }

    private void addNeurons(ConvolutionalLayerConfiguration configuration, Layer layer) {
        for (int i = 0; i < configuration.getNeuronCount(); ++i) {
            Neuron neuron = new Neuron();
            layer.addNeuron(neuron);
        }
    }
}
