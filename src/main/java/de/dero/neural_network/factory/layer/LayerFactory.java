package de.dero.neural_network.factory.layer;

import de.dero.neural_network.Layer;
import de.dero.neural_network.factory.LayerConfiguration;

import java.util.Random;

public abstract class LayerFactory<T extends LayerConfiguration> {

    private static Random random = new Random();

    static double getRandomWeight() {
        return .05 * (2 * random.nextDouble() - 1);
    }

    public abstract Layer createLayer(T configuration, Layer previousLayer);
}
