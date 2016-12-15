package de.dero.neural_network.factory;

import de.dero.neural_network.activation.ActivationFunction;

public class Convolutional2DLayerConfiguration extends ConvolutionalLayerConfiguration {

    private final int inputWidth;
    private final int inputHeight;
    private final int kernelWidth;
    private final int kernelHeight;
    private final int horizontalKernelOffset;
    private final int verticalKernelOffset;

    public Convolutional2DLayerConfiguration(int featureMapCount,
                                             int inputWidth,
                                             int inputHeight,
                                             int kernelWidth,
                                             int kernelHeight,
                                             int horizontalKernelOffset,
                                             int verticalKernelOffset,
                                             ActivationFunction activationFunction,
                                             int[] connectionKernel,
                                             LayerConfiguration previousLayerConfiguration) {
        super(
            featureMapCount,
            ((inputHeight - kernelHeight) / horizontalKernelOffset + 1)
            * ((inputWidth - kernelWidth) / verticalKernelOffset + 1),
            activationFunction,
            connectionKernel,
            0,
            previousLayerConfiguration
        );
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.horizontalKernelOffset = horizontalKernelOffset;
        this.verticalKernelOffset = verticalKernelOffset;
    }

    public int getHorizontalKernelOffset() {
        return horizontalKernelOffset;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getKernelHeight() {
        return kernelHeight;
    }

    public int getKernelWidth() {
        return kernelWidth;
    }

    public int getVerticalKernelOffset() {
        return verticalKernelOffset;
    }
}
