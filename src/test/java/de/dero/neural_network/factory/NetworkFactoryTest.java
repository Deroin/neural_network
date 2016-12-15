package de.dero.neural_network.factory;

import de.dero.neural_network.NetworkStructureTester;
import de.dero.neural_network.NeuralNetwork;
import de.dero.neural_network.activation.HyperbolicTangent;
import de.dero.neural_network.serialization.StructureSerializer;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;

public class NetworkFactoryTest implements NetworkStructureTester {

    private static NetworkFactory factory;
    private StructureSerializer structureSerializer = new StructureSerializer();

    @BeforeClass
    public static void setUp() {
        factory = new NetworkFactory();
    }

    @Test
    public void convolutional2DLayers() throws IOException {
        NeuralNetwork expected = structureSerializer.toNetwork(
                getClass().getResourceAsStream("convolutional2DLayers.json")
        );

        int[] kernelTemplate = new int[]{0, 1, 2, 3, 4, 33, 34, 35, 36, 37, 66, 67, 68, 69, 70, 99, 100, 101, 102,
                                         103, 132, 133, 134, 135, 136};

        Convolutional2DLayerConfiguration layer1Configuration = new Convolutional2DLayerConfiguration(
                6, 33, 33, 5, 5, 2, 2, new HyperbolicTangent(), kernelTemplate, null
        );
        kernelTemplate = new int[]{0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 30, 31, 32, 34, 35, 45, 46, 47, 48, 49, 60, 61,
                                   62, 63, 64};

        Convolutional2DLayerConfiguration layer2Configuration = new Convolutional2DLayerConfiguration(
                50, 15, 15, 5, 5, 2, 2, new HyperbolicTangent(), kernelTemplate, layer1Configuration
        );

        NeuralNetwork actual = factory.newNetwork()
                                      .setInputLayer(1089, new HyperbolicTangent())
                                      .addLayer(layer1Configuration)
                                      .addLayer(layer2Configuration)
                                      .construct();

        assertNetworkStructure(expected, actual);
    }

    @Test
    public void fullyConnectedLayers() throws IOException {
        String fileName = "fullyConnectedLayers.json";
        NeuralNetwork expected = structureSerializer.toNetwork(
                getClass().getResourceAsStream(fileName)
        );
        NeuralNetwork actual = factory.newNetwork()
                                      .setInputLayer(10, new HyperbolicTangent())
                                      .addLayer(10)
                                      .addLayer(100)
                                      .construct();

        assertNetworkStructure(expected, actual);
    }

    @Test
    public void redeclareInputLayer() throws IOException {
        NeuralNetwork expected = structureSerializer.toNetwork(
                getClass().getResourceAsStream("redeclareInputLayer.json")
        );
        NeuralNetwork actual = factory.newNetwork()
                                      .setInputLayer(10, new HyperbolicTangent())
                                      .setInputLayer(20, new HyperbolicTangent())
                                      .construct();
        assertNetworkStructure(expected, actual);
    }

    @Test(expected = RuntimeException.class)
    public void unknownLayerType() {
        LayerConfiguration layer = new LayerConfiguration(-1, null) {
            @Override
            public int getNeuronCount() {
                return -1;
            }
        };
        factory.newNetwork()
               .addLayer(layer)
               .construct();
    }
}
