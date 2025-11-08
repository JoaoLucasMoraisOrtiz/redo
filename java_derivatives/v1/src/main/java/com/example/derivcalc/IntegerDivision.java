package com.example.derivcalc;

/**
 * Provides integer division utilities returning both quotient and remainder.
 */
public final class IntegerDivision {

    private IntegerDivision() {
        // Utility class
    }

    /**
     * Performs integer division returning the quotient.
     *
     * @param dividend value to be divided
     * @param divisor value to divide by
     * @return quotient of the integer division
     * @throws ArithmeticException when divisor is zero
     */
    public static int divide(int dividend, int divisor) {
        if (divisor == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return dividend / divisor;
    }

    /**
     * Performs integer division returning both quotient and remainder.
     *
     * @param dividend value to be divided
     * @param divisor value to divide by
     * @return array where index 0 is the quotient and index 1 is the remainder
     * @throws ArithmeticException when divisor is zero
     */
    public static int[] divideWithRemainder(int dividend, int divisor) {
        if (divisor == 0) {
            throw new ArithmeticException("Division by zero");
        }
        int quotient = dividend / divisor;
        int remainder = dividend % divisor;
        return new int[] { quotient, remainder };
    }
}
