package com.example.derivcalc;

/**
 * Sample univariate functions for derivative and integral demonstrations.
 */
public final class UnivariateSamples {
    private UnivariateSamples() {
    }

    public static UnivariateFunction cubicPolynomial() {
        return x -> 0.5 * x * x * x - 3.0 * x * x + 2.0 * x - 1.0;
    }

    public static UnivariateFunction sineWave() {
        return Math::sin;
    }

    public static UnivariateFunction exponentialDecay() {
        return x -> Math.exp(-x);
    }
}
