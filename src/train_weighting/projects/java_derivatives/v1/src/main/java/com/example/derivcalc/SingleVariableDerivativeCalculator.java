package com.example.derivcalc;

/**
 * Computes first derivatives for single-variable functions using finite differences.
 */
public final class SingleVariableDerivativeCalculator {
    private final double defaultStep;

    public SingleVariableDerivativeCalculator(double defaultStep) {
        if (defaultStep <= 0.0) {
            throw new IllegalArgumentException("Step size must be positive");
        }
        this.defaultStep = defaultStep;
    }

    public double forwardDifference(UnivariateFunction function, double point) {
        return forwardDifference(function, point, defaultStep);
    }

    public double centralDifference(UnivariateFunction function, double point) {
        return centralDifference(function, point, defaultStep);
    }

    public double forwardDifference(UnivariateFunction function, double point, double step) {
        validate(function, step);
        double fx = function.evaluate(point);
        double fxStep = function.evaluate(point + step);
        return (fxStep - fx) / step;
    }

    public double centralDifference(UnivariateFunction function, double point, double step) {
        validate(function, step);
        double fxForward = function.evaluate(point + step);
        double fxBackward = function.evaluate(point - step);
        return (fxForward - fxBackward) / (2.0 * step);
    }

    private void validate(UnivariateFunction function, double step) {
        if (function == null) {
            throw new IllegalArgumentException("Function must not be null");
        }
        if (step <= 0.0) {
            throw new IllegalArgumentException("Step size must be positive");
        }
    }
}
