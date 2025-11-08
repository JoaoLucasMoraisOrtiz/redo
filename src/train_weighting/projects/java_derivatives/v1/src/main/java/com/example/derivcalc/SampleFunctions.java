package com.example.derivcalc;

/**
 * Provides sample multivariate functions for demonstration and testing.
 */
public final class SampleFunctions {
    private SampleFunctions() {
    }

    public static MultivariateFunction quadraticSurface() {
        return point -> {
            double x = point[0];
            double y = point[1];
            return 4.0 * x * x + 3.0 * x * y + 2.0 * y * y;
        };
    }

    public static MultivariateFunction trigonometricWave() {
        return point -> {
            double x = point[0];
            double y = point[1];
            double z = point.length > 2 ? point[2] : 0.0;
            return Math.sin(x) * Math.cos(y) + Math.exp(-z);
        };
    }

    public static MultivariateFunction logisticRidge() {
        return point -> {
            double x = point[0];
            double y = point[1];
            return 1.0 / (1.0 + Math.exp(-(x - 2.0 * y)));
        };
    }
}
