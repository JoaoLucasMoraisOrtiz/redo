package com.example.derivcalc;

/**
 * Represents a differentiable multivariate function f(x_0, x_1, ..., x_n).
 */
@FunctionalInterface
public interface MultivariateFunction {
    /**
     * Evaluates the function at the provided point.
     *
     * @param point array containing the coordinate of each variable
     * @return function value at that point
     */
    double evaluate(double[] point);
}
