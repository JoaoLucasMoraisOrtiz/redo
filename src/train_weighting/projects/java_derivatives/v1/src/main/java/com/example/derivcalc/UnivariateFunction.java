package com.example.derivcalc;

/**
 * Represents a differentiable single-variable function f(x).
 */
@FunctionalInterface
public interface UnivariateFunction {
    double evaluate(double x);
}
