package de.dero.neural_network.factory;

import de.dero.neural_network.Layer;
import de.dero.neural_network.NeuralNetwork;
import de.dero.neural_network.activation.ActivationFunction;
import de.dero.neural_network.activation.HyperbolicTangent;
import de.dero.neural_network.factory.layer.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NetworkFactory {

    private Map<Class<? extends LayerConfiguration>, LayerFactory<? extends LayerConfiguration>>
        layerFactories = new HashMap<>();
    private List<LayerConfiguration> layers;

    public NetworkFactory() {
        newNetwork();
        registerLayerFactory(InputLayerConfiguration.class, new InputLayerFactory());
        registerLayerFactory(FullyConnectedLayerConfiguration.class, new FullyConnectedLayerFactory());
        registerLayerFactory(ConvolutionalLayerConfiguration.class, new ConvolutionalLayerFactory<>());
        registerLayerFactory(Convolutional2DLayerConfiguration.class, new Convolutional2DLayerFactory());
    }

    public NetworkFactory newNetwork() {
        layers = new ArrayList<>();
        return this;
    }

    public <T extends LayerConfiguration> NetworkFactory registerLayerFactory(Class<T> layerType,
                                                                              LayerFactory<T> factory) {
        layerFactories.put(layerType, factory);
        return this;
    }

    public NetworkFactory addLayer(int neuronCount) {
        if (layers.size() == 0) {
            setInputLayer(neuronCount, new HyperbolicTangent());
            return this;
        }

        LayerConfiguration layerConfiguration = new FullyConnectedLayerConfiguration(
            neuronCount,
            new HyperbolicTangent()
        );
        layers.add(layerConfiguration);
        return this;
    }

    public NetworkFactory addLayer(LayerConfiguration layerConfiguration) {
        layers.add(layerConfiguration);
        return this;
    }

    public NeuralNetwork construct() {
        NeuralNetwork network = new NeuralNetwork();
        Layer previousLayer = null;
        for (LayerConfiguration configuration : layers) {
            LayerFactory factory = layerFactories.get(configuration.getClass());
            if (factory == null) {
                throw new RuntimeException("There is no layer factory registered for class " +
                                    configuration.getClass().getName());
            }
            //noinspection unchecked
            previousLayer = factory.createLayer(configuration, previousLayer);
            network.addLayer(previousLayer);
        }
        return network;
    }

    public NetworkFactory setInputLayer(int neuronCount, ActivationFunction activationFunction) {
        if(layers.size() == 0) {
            layers.add(new InputLayerConfiguration(neuronCount, activationFunction));
        } else {
            layers.set(0, new InputLayerConfiguration(neuronCount, activationFunction));
        }
        return this;
    }
}
