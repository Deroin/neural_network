package de.dero.neural_network.factory.layer;

import de.dero.neural_network.Connection;
import de.dero.neural_network.Layer;
import de.dero.neural_network.Neuron;
import de.dero.neural_network.factory.Convolutional2DLayerConfiguration;
import de.dero.neural_network.factory.ConvolutionalLayerConfiguration;

public class Convolutional2DLayerFactory extends ConvolutionalLayerFactory<Convolutional2DLayerConfiguration> {

    @Override
    protected void connectFeatureMap(ConvolutionalLayerConfiguration configuration, Layer layer) {
        if (!(configuration instanceof Convolutional2DLayerConfiguration)) {
            throw new IllegalArgumentException("Configuration must be of type Convolutional2DLayerConfiguration");
        }
        Convolutional2DLayerConfiguration config = ((Convolutional2DLayerConfiguration) configuration);
        int featureMapCount = config.getFeatureMapCount();
        int[] kernelTemplate = config.getConnectionKernel();
        int kernelSize = kernelTemplate.length;
        int kernelHeight = config.getKernelHeight();
        int kernelWidth = config.getKernelWidth();
        int horizontalKernelOffset = config.getHorizontalKernelOffset();
        int featureCount = configuration.getFeatureCount();
        int verticalKernelOffset = config.getVerticalKernelOffset();

        int inputWidth = config.getInputWidth();
        int inputHeight = config.getInputHeight();

        int horizontalFeatures = (inputHeight - kernelHeight) / horizontalKernelOffset + 1;
        int verticalFeatures = (inputWidth - kernelWidth) / verticalKernelOffset + 1;
        int inputSize = inputWidth * inputHeight;

        int inputFeatureMapCount;
        if (config.getPreviousLayerConfiguration() instanceof ConvolutionalLayerConfiguration) {
            inputFeatureMapCount =
                ((ConvolutionalLayerConfiguration) config.getPreviousLayerConfiguration()).getFeatureMapCount();
        } else {
            inputFeatureMapCount = 1;
        }

        for (int featureMap = 0; featureMap < featureMapCount; featureMap++) {
            int neuronOffset = featureMap * featureCount;
            for (int i = 0; i < horizontalFeatures; i++) {
                for (int j = 0; j < verticalFeatures; j++) {
                    int weightNum = featureMap * (kernelSize + 1);
                    Neuron n = layer.getNeurons().get(j + i * horizontalFeatures + neuronOffset);
                    n.addConnection(Connection.BIAS_NEURON_INDEX, weightNum++); //bias weight
                    int kernelOffset = horizontalKernelOffset * j + verticalKernelOffset * inputWidth * i;
                    for (int kernelNode : kernelTemplate) {
                        for (int k = 0; k < inputFeatureMapCount; k++) {
                            n.addConnection(k * inputSize + kernelOffset + kernelNode, weightNum++);
                        }
                    }
                }
            }
        }
    }
}
