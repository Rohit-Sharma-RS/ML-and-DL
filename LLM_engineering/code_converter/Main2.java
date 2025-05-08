import java.util.Random;

public class Main2 {
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
        int size = 300;
        Random rand = new Random();

        double[][] matrixA = generateMatrix(size, rand);
        double[][] matrixB = generateMatrix(size, rand);
        double[][] result = new double[size][size];

        long startTime = System.currentTimeMillis();

        // Matrix multiplication
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                for (int k = 0; k < size; k++) {
                    sum += matrixA[i][k] * matrixB[k][j];
                }
                result[i][j] = sum;
            }
        }

        long endTime = System.currentTimeMillis();
        double executionTime = (endTime - startTime) / 1000.0;

        double sumValue = 0.0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sumValue += result[i][j];
            }
        }

        System.out.println("Matrix multiplication of " + size + "x" + size + " matrices (Java)");
        System.out.printf("Result sum: %.6f%n", sumValue);
        System.out.printf("Execution Time: %.6f seconds%n", executionTime);
    }
}
