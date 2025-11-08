package com.example.derivcalc;

import java.util.Arrays;

/**
 * Simple CLI bootstrap demonstrating the derivative and integral calculators.
 */
public final class Application {
    private Application() {
    }

    public static void main(String[] args) {
        FiniteDifferencePartialDerivativeCalculator partial = new FiniteDifferencePartialDerivativeCalculator(1e-4);
        GradientCalculator gradientCalculator = new GradientCalculator(partial);
        SingleVariableDerivativeCalculator singleDerivative = new SingleVariableDerivativeCalculator(1e-4);
        NumericalIntegralCalculator integralCalculator = new NumericalIntegralCalculator(1000);
        BasicArithmeticCalculator arithmeticCalculator = new BasicArithmeticCalculator();

        MultivariateFunction function = SampleFunctions.quadraticSurface();
        double[] point = {1.5, -0.8};

        System.out.printf("Forward difference wrt x: %.6f%n",
                partial.forwardDifference(function, point, 0));
        System.out.printf("Central difference wrt y: %.6f%n",
                partial.centralDifference(function, point, 1));

        double[] gradient = gradientCalculator.gradient(function, point);
        System.out.println("Gradient: " + Arrays.toString(gradient));

        UnivariateFunction cubic = UnivariateSamples.cubicPolynomial();
        double x0 = 1.2;
        System.out.printf("Single-variable derivative f'(%.2f): %.6f%n",
            x0,
            singleDerivative.centralDifference(cubic, x0));

        double area = integralCalculator.simpson(UnivariateSamples.sineWave(), 0.0, Math.PI);
        System.out.printf("Integral of sin(x) from 0 to pi: %.6f%n", area);

        double sum = arithmeticCalculator.add(12.5, 7.4);
        double power = arithmeticCalculator.power(3.0, 4.0);
        double root = arithmeticCalculator.root(81.0, 2.0);
        System.out.printf("Sum: %.2f, Power: %.2f, Root: %.2f%n", sum, power, root);
    }
}
