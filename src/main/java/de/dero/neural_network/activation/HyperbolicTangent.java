package de.dero.neural_network.activation;

/**
 * Uses the hyperbolic tangent for scaled sigmoid calculation
 * <p>
 * x = F(y) = 1.7159 * tanh(2/3*y)
 * F'(y) = 1.7159*2/3*(1-tanh^2(2/3*y))
 * F'(F(y)) = G(x) = (2/3)/1.7159 * (1.7159^2-x^2)
 * G(X) = 0.3885230297025856 * (2.94431281 - x^2)
 * </p>
 */
public class HyperbolicTangent implements ActivationFunction {
    @Override
    public double calculate(double value) {
        return 1.7159d * Math.tanh(2d * value / 3d);
    }

    @Override
    public double calculateDerivativeWithX(double value) {
        return 0.3885230297025856 * (2.94431281 - value * value);
    }
}
