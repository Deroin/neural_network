package de.dero.neural_network.factory;

import de.dero.neural_network.activation.ActivationFunction;

public class ConvolutionalLayerConfiguration extends LayerConfiguration {

    private final int featureMapCount;
    private int featureCount;
    private final int[] connectionKernel;
    private final int kernelOffset;
    private final LayerConfiguration previousLayerConfiguration;

    public ConvolutionalLayerConfiguration(int featureMapCount,
                                           int featureCount,
                                           ActivationFunction activationFunction,
                                           int[] connectionKernel,
                                           int kernelOffset,
                                           LayerConfiguration previousLayerConfiguration) {
        super(featureMapCount * featureCount, activationFunction);
        this.featureMapCount = featureMapCount;
        this.featureCount = featureCount;
        this.connectionKernel = connectionKernel;
        this.kernelOffset = kernelOffset;
        this.previousLayerConfiguration = previousLayerConfiguration;
    }

    public int getFeatureMapCount() {
        return featureMapCount;
    }

    public int getFeatureCount() {
        return featureCount;
    }

    public int[] getConnectionKernel() {
        return connectionKernel;
    }

    public int getKernelOffset() {
        return kernelOffset;
    }

    public LayerConfiguration getPreviousLayerConfiguration() {
        return previousLayerConfiguration;
    }
}
