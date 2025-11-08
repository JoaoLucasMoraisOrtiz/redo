package com.example.derivcalc;

import java.util.Arrays;

/**
 * Computes partial derivatives using configurable finite difference schemes.
 */
public final class FiniteDifferencePartialDerivativeCalculator {
    private final double defaultStep;

    /**
     * Creates a calculator with the provided default step size.
     *
     * @param defaultStep step size used when none is specified for a call
     */
    public FiniteDifferencePartialDerivativeCalculator(double defaultStep) {
        if (defaultStep <= 0.0) {
            throw new IllegalArgumentException("Step size must be positive");
        }
        this.defaultStep = defaultStep;
    }

    /**
     * Computes the partial derivative using a forward difference scheme.
     */
    public double forwardDifference(MultivariateFunction function, double[] point, int variableIndex) {
        return forwardDifference(function, point, variableIndex, defaultStep);
    }

    /**
     * Computes the partial derivative using a central difference scheme.
     */
    public double centralDifference(MultivariateFunction function, double[] point, int variableIndex) {
        return centralDifference(function, point, variableIndex, defaultStep);
    }

    public double forwardDifference(MultivariateFunction function, double[] point, int variableIndex, double step) {
        validateInputs(function, point, variableIndex, step);
        double[] shifted = Arrays.copyOf(point, point.length);
        shifted[variableIndex] += step;
        double fxStep = function.evaluate(shifted);
        double fx = function.evaluate(point);
        return (fxStep - fx) / step;
    }

    public double centralDifference(MultivariateFunction function, double[] point, int variableIndex, double step) {
        validateInputs(function, point, variableIndex, step);
        double[] forward = Arrays.copyOf(point, point.length);
        double[] backward = Arrays.copyOf(point, point.length);
        forward[variableIndex] += step;
        backward[variableIndex] -= step;
        double fxForward = function.evaluate(forward);
        double fxBackward = function.evaluate(backward);
        return (fxForward - fxBackward) / (2.0 * step);
    }

    private void validateInputs(MultivariateFunction function, double[] point, int variableIndex, double step) {
        if (function == null) {
            throw new IllegalArgumentException("Function must not be null");
        }
        if (point == null) {
            throw new IllegalArgumentException("Point must not be null");
        }
        if (variableIndex < 0 || variableIndex >= point.length) {
            throw new IllegalArgumentException("Variable index out of bounds");
        }
        if (step <= 0.0) {
            throw new IllegalArgumentException("Step size must be positive");
        }
    }
}
