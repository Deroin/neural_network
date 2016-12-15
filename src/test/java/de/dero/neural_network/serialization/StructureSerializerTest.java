package de.dero.neural_network.serialization;

import de.dero.neural_network.NetworkStructureTester;
import de.dero.neural_network.NeuralNetwork;
import de.dero.neural_network.activation.HyperbolicTangent;
import de.dero.neural_network.factory.FullyConnectedLayerConfiguration;
import de.dero.neural_network.factory.NetworkFactory;
import org.junit.BeforeClass;
import org.junit.Test;

public class StructureSerializerTest implements NetworkStructureTester {

    private static NeuralNetwork network;
    private static StructureSerializer serializer;

    @BeforeClass
    public static void setUp() {
        NetworkFactory factory = new NetworkFactory();
        factory.addLayer(5);
        factory.addLayer(new FullyConnectedLayerConfiguration(5, new HyperbolicTangent()));
        network = factory.construct();

        serializer = new StructureSerializer();
    }

    @Test
    public void testSerializer() {
        String s = serializer.fromNetwork(network);
        NeuralNetwork actual = serializer.toNetwork(s);

        assertNetworkStructure(network, actual);
    }
}
