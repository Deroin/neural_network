package de.dero.neural_network;

import de.dero.neural_network.activation.ActivationFunction;
import de.dero.neural_network.activation.HyperbolicTangent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.pow;

public class Layer implements Serializable {

    private static final long serialVersionUID = 2;
    private ActivationFunction activationFunction = new HyperbolicTangent();
    private ArrayList<Neuron> neurons = new ArrayList<>();
    Layer previousLayer;

    private List<Weight> weights = new ArrayList<>();

    public Layer() {
        this(null, null);
    }

    public Layer(Layer previousLayer) {
        this(previousLayer, null);
    }
    
    public Layer(Layer previousLayer, ActivationFunction activationFunction) {
        this.previousLayer = previousLayer;
        if (activationFunction != null) {
            this.activationFunction = activationFunction;
        }
    }

    void calculate() {
        assert previousLayer != null;
        double sum;
        List<Neuron> inputNeurons = previousLayer.getNeurons();

        for (Neuron neuron : neurons) {
            sum = 0;
            for (Connection connection : neuron.getInputConnections()) {
                int weightIndex = connection.getWeightIndex();
                int neuronIndex = connection.getNeuronIndex();
                assert neuronIndex == Connection.BIAS_NEURON_INDEX || neuronIndex < inputNeurons.size();
                assert weightIndex < weights.size();

                if (neuronIndex == Connection.BIAS_NEURON_INDEX) {
                    sum += weights.get(weightIndex).value;
                } else {
                    sum += weights.get(weightIndex).value * inputNeurons.get(neuronIndex).getOutput();
                }
            }
            neuron.setOutput(activationFunction.calculate(sum));
        }
    }

    void backPropagate(double[] errorByXn, double[] errorByXnm1, double learningRate) {
        assert errorByXn.length == neurons.size();
        assert previousLayer != null;
        assert errorByXnm1.length == previousLayer.neurons.size();

        double[] errorByY = new double[neurons.size()];
        double[] errorByW = new double[weights.size()];
        for (int i = 0; i < errorByW.length; i++) {
            errorByW[i] = 0d;
        }

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            //dE/dy = G(x)*dE/dx
            errorByY[i] = activationFunction.calculateDerivativeWithX(neuron.getOutput()) * errorByXn[i];
            for (Connection connection : neuron.getInputConnections()) {
                double xnm1;
                int neuronIndex = connection.getNeuronIndex();
                if (neuronIndex == Connection.BIAS_NEURON_INDEX) {
                    xnm1 = 1;
                } else {
                    assert neuronIndex < previousLayer.neurons.size();
                    xnm1 = previousLayer.neurons.get(neuronIndex).getOutput();

                    assert neuronIndex < errorByXnm1.length;
                    assert i < errorByY.length;
                    assert connection.getWeightIndex() < weights.size();
                    // dE/dx_n-1 = \Sigma w * dE/dy
                    errorByXnm1[neuronIndex] += weights.get(connection.getWeightIndex()).value * errorByY[i];
                }
                //dE/dw = x_n-1*dE/dy
                errorByW[connection.getWeightIndex()] += xnm1 * errorByY[i];
            }
        }
        for (int i = 0; i < weights.size(); i++) {
            double divisor = weights.get(i).hessian + .1;
            double epsilon = learningRate / divisor;
            //w_new = w_old - learningRate * dE/dw
            weights.get(i).value -= epsilon * errorByW[i];
        }
    }

    void backPropagateSecondDerivative(double[] errorByXn, double[] errorByXnm1) {
        assert errorByXn.length == neurons.size();
        assert previousLayer != null;
        assert errorByXnm1.length == previousLayer.neurons.size();
        double[] errorByY = new double[neurons.size()];
        double[] errorByW = new double[weights.size()];
        for (int i = 0; i < errorByW.length; i++) {
            errorByW[i] = 0;
        }
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            //d^2E/dy = G(x)^2*d^2E/dx
            errorByY[i] = pow(activationFunction.calculateDerivativeWithX(neuron.getOutput()), 2) * errorByXn[i];
            for (Connection connection : neuron.getInputConnections()) {
                double xnm1;
                int neuronIndex = connection.getNeuronIndex();
                if (neuronIndex == Connection.BIAS_NEURON_INDEX) {
                    xnm1 = 1;
                } else {
                    assert neuronIndex < previousLayer.neurons.size();
                    xnm1 = previousLayer.neurons.get(neuronIndex).getOutput();

                    assert neuronIndex < errorByXnm1.length;
                    assert i < errorByY.length;
                    assert connection.getWeightIndex() < weights.size();
                    //d^2E/dx_n-1 = \Sigma w^2 * d^2E/dy
                    errorByXnm1[neuronIndex] += pow(weights.get(connection.getWeightIndex()).value, 2) * errorByY[i];
                }
                //d^2E/dw = x_n-1^2*d^2E/dy
                errorByW[connection.getWeightIndex()] += pow(xnm1, 2) * errorByY[i];
            }
        }

        for (int i = 0; i < weights.size(); i++) {
            //h_new = h_old + d^2E/dw
            weights.get(i).hessian += errorByW[i];
        }
    }

    public void addNeuron(Neuron neuron) {
        neurons.add(neuron);
    }

    public void addWeight(double weight) {
        Weight w = new Weight();
        w.value = weight;
        weights.add(w);
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public List<Weight> getWeights() {
        return weights;
    }
}
