package com.example.derivcalc;

/**
 * Provides basic arithmetic operations for real numbers.
 */
public final class BasicArithmeticCalculator {
    public double add(double a, double b) {
        return a + b;
    }

    public double subtract(double a, double b) {
        return a - b;
    }

    public double multiply(double a, double b) {
        return a * b;
    }

    public double divide(double a, double b) {
        if (b == 0.0) {
            throw new ArithmeticException("Division by zero");
        }
        return a / b;
    }

    public double power(double base, double exponent) {
        return Math.pow(base, exponent);
    }

    public double root(double value, double degree) {
        if (degree == 0.0) {
            throw new ArithmeticException("Root degree cannot be zero");
        }
        if (value < 0.0 && degree % 2.0 == 0.0) {
            throw new ArithmeticException("Even-degree root of a negative number is not real");
        }
        return Math.pow(value, 1.0 / degree);
    }
}
