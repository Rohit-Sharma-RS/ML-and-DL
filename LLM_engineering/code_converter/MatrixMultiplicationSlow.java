import java.util.Random;

public class MatrixMultiplicationSlow {
    public static double[][] generateMatrix(int size, Random rand) {
        double[][] matrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = rand.nextDouble();
            }
        }
        return matrix;
    }

    public static void main(String[] args) {
        int size = 500; // try 200 or 300 first if you're testing
        Random rand = new Random();

        double[][] matrixA = generateMatrix(size, rand);
        double[][] matrixB = generateMatrix(size, rand);
        double[][] result = new double[size][size];

        long startTime = System.currentTimeMillis();

        // Intentionally slow matrix multiplication (no reuse, no tricks)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = 0.0;
                for (int k = 0; k < size; k++) {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        long endTime = System.currentTimeMillis();
        double executionTime = (endTime - startTime) / 1000.0;

        // Sum result matrix to confirm it's not optimized away
        double sumValue = 0.0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sumValue += result[i][j];
            }
        }

        System.out.println("Matrix multiplication of " + size + "x" + size + " matrices (Java - Slow)");
        System.out.printf("Result sum: %.6f%n", sumValue);
        System.out.printf("Execution Time: %.6f seconds%n", executionTime);
    }
}
