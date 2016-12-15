package de.dero.neural_network;

import org.junit.Assert;

import java.util.Iterator;
import java.util.List;

public interface NetworkStructureTester {
    default void assertLayerStructure(int layerIndex, Layer expected, Layer actual) {
        List<Neuron> expectedNeurons = expected.getNeurons();
        List<Neuron> actualNeurons = actual.getNeurons();
        Assert.assertEquals(
                "[L" + layerIndex + "] Mismatching neuron count",
                expectedNeurons.size(),
                actualNeurons.size()
        );
        Assert.assertEquals(
                "[L" + layerIndex + "] Mismatching weight count",
                expected.getWeights().size(),
                actual.getWeights().size()
        );

        Iterator<Neuron> expectedIterator = expectedNeurons.iterator();
        Iterator<Neuron> actualIterator = actualNeurons.iterator();

        int i = 0;
        while (expectedIterator.hasNext()) {
            assertNeuronStructure(layerIndex, i++, expectedIterator.next(), actualIterator.next());
        }
    }

    default void assertNetworkStructure(NeuralNetwork expected, NeuralNetwork actual) {
        List<Layer> expectedLayers = expected.getLayers();
        List<Layer> actualLayers = actual.getLayers();
        Assert.assertEquals("Mismatching layer count", expectedLayers.size(), actualLayers.size());

        Iterator<Layer> expectedIterator = expectedLayers.iterator();
        Iterator<Layer> actualIterator = actualLayers.iterator();

        int i = 0;
        while (expectedIterator.hasNext()) {
            assertLayerStructure(i++, expectedIterator.next(), actualIterator.next());
        }
    }

    default void assertNeuronStructure(int layerIndex, int neuronIndex, Neuron expected, Neuron actual) {
        List<Connection> expectedInputConnections = expected.getInputConnections();
        List<Connection> actualInputConnections = actual.getInputConnections();
        Assert.assertEquals(
                "[L" + layerIndex + "N" + neuronIndex + "] Mismatching connection count",
                expectedInputConnections.size(),
                actualInputConnections.size()
        );

        Iterator<Connection> expectedIterator = expectedInputConnections.iterator();
        Iterator<Connection> actualIterator = actualInputConnections.iterator();

        int i = 0;
        while (expectedIterator.hasNext()) {
            assertConnectionMatches(layerIndex, neuronIndex, i++, expectedIterator.next(), actualIterator.next());
        }
    }
    default void assertConnectionMatches(int layerIndex,
                                         int neuronIndex,
                                         int connectionIndex,
                                         Connection expected,
                                         Connection actual) {
        Assert.assertEquals(
                "[L" + layerIndex + "N" + neuronIndex + "C" + connectionIndex + "] Wrong neuron index",
                expected.getNeuronIndex(),
                actual.getNeuronIndex()
        );
        Assert.assertEquals(
                "[L" + layerIndex + "N" + neuronIndex + "C" + connectionIndex + "] Wrong weight index",
                expected.getWeightIndex(),
                actual.getWeightIndex()
        );
    }
}
