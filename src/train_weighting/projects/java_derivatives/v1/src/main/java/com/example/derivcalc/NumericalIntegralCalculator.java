package com.example.derivcalc;

/**
 * Performs numerical integration for single-variable functions.
 */
public final class NumericalIntegralCalculator {
    private final int defaultSteps;

    public NumericalIntegralCalculator(int defaultSteps) {
        if (defaultSteps <= 0) {
            throw new IllegalArgumentException("Integration steps must be positive");
        }
        this.defaultSteps = defaultSteps;
    }

    public double trapezoidal(UnivariateFunction function, double start, double end) {
        return trapezoidal(function, start, end, defaultSteps);
    }

    public double simpson(UnivariateFunction function, double start, double end) {
        int steps = defaultSteps % 2 == 0 ? defaultSteps : defaultSteps + 1;
        return simpson(function, start, end, steps);
    }

    public double trapezoidal(UnivariateFunction function, double start, double end, int steps) {
        validate(function, steps);
        double h = (end - start) / steps;
        double sum = 0.5 * (function.evaluate(start) + function.evaluate(end));
        for (int i = 1; i < steps; i++) {
            double x = start + i * h;
            sum += function.evaluate(x);
        }
        return sum * h;
    }

    public double simpson(UnivariateFunction function, double start, double end, int steps) {
        if (steps <= 0 || steps % 2 != 0) {
            throw new IllegalArgumentException("Simpson's rule requires a positive even number of steps");
        }
        validate(function, steps);
        double h = (end - start) / steps;
        double sum = function.evaluate(start) + function.evaluate(end);
        for (int i = 1; i < steps; i++) {
            double x = start + i * h;
            sum += (i % 2 == 0 ? 2.0 : 4.0) * function.evaluate(x);
        }
        return sum * h / 3.0;
    }

    private void validate(UnivariateFunction function, int steps) {
        if (function == null) {
            throw new IllegalArgumentException("Function must not be null");
        }
        if (steps <= 0) {
            throw new IllegalArgumentException("Integration steps must be positive");
        }
    }
}
