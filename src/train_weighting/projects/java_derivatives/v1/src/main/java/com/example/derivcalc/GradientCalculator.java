package com.example.derivcalc;

import java.util.Objects;

/**
 * Computes gradient vectors by delegating to a partial derivative calculator.
 */
public final class GradientCalculator {
    private final FiniteDifferencePartialDerivativeCalculator partialDerivativeCalculator;

    public GradientCalculator(FiniteDifferencePartialDerivativeCalculator partialDerivativeCalculator) {
        this.partialDerivativeCalculator = Objects.requireNonNull(partialDerivativeCalculator);
    }

    public double[] gradient(MultivariateFunction function, double[] point) {
        Objects.requireNonNull(function, "function");
        Objects.requireNonNull(point, "point");
        double[] gradient = new double[point.length];
        for (int i = 0; i < point.length; i++) {
            gradient[i] = partialDerivativeCalculator.centralDifference(function, point, i);
        }
        return gradient;
    }
}
