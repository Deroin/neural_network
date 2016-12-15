package de.dero.neural_network;

import java.util.*;

public class TrainingSet {

    private List<Vector<Double>> dataSets = new ArrayList<>();
    private List<Vector<Double>> desiredResults = new ArrayList<>();
    private List<Integer> order = new ArrayList<>();

    private int dataSetSize;
    private int resultSize;

    public TrainingSet() {
    }

    public TrainingSet(List<Vector<Double>> dataSets, List<Vector<Double>> desiredResults) {
        validateConstructorArguments(dataSets, desiredResults);

        this.dataSets.addAll(dataSets);
        this.desiredResults.addAll(desiredResults);
        if(dataSets.size() > 0) {
            dataSetSize = dataSets.get(0).size();
            resultSize = desiredResults.get(0).size();
        }
        for (int i = 0; i < dataSets.size(); i++) {
            order.add(i);
        }
    }

    TrainingSet(TrainingSet trainingSet) {
        this(trainingSet.dataSets, trainingSet.desiredResults);
    }

    public void add(Vector<Double> dataSet, Vector<Double> desiredResult) {
        if (dataSetSize == 0) {
            dataSetSize = dataSet.size();
            resultSize = desiredResult.size();
        } else if (dataSet.size() != dataSetSize || desiredResult.size() != resultSize) {
            throw new IllegalArgumentException("Vector sizes must match with all other elements");
        }
        dataSets.add(dataSet);
        desiredResults.add(desiredResult);
        order.add(dataSets.size());
    }

    void randomize(Random random) {
        Collections.shuffle(order, random);
    }

    int size() {
        return dataSets.size();
    }

    Vector<Double> getData(int index) {
        return dataSets.get(order.get(index));
    }

    Vector<Double> getDesiredResult(int index) {
        return desiredResults.get(order.get(index));
    }

    private void validateConstructorArguments(List<Vector<Double>> dataSets, List<Vector<Double>> desiredResults) {
        if (dataSets.size() != desiredResults.size()) {
            throw new IllegalArgumentException("Sizes of both lists must match");
        }
        if(dataSets.size() > 0) {
            validateEqualElementSize(dataSets);
            validateEqualElementSize(desiredResults);
        }
    }

    private void validateEqualElementSize(List<Vector<Double>> dataSets) {
        int elementSize = dataSets.get(0).size();
        dataSets.stream().filter(vector -> vector.size() == elementSize).findAny().ifPresent(vector -> {
            throw new IllegalArgumentException("All elements inside one list must be of the same size");
        });
    }
}
