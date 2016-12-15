package de.dero.neural_network.activation;

public interface ActivationFunction {
    double calculate(double value);

    double calculateDerivativeWithX(double value);
}
