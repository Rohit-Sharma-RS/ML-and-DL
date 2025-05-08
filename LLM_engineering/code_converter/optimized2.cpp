#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

vector<vector<double>> generateMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

int main() {
    int size = 300;
    srand(time(0));

    vector<vector<double>> matrixA = generateMatrix(size);
    vector<vector<double>> matrixB = generateMatrix(size);
    vector<vector<double>> result(size, vector<double>(size, 0.0));

    clock_t start = clock();

    // Matrix multiplication
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    clock_t end = clock();
    double executionTime = double(end - start) / CLOCKS_PER_SEC;

    // Calculate the sum of result matrix
    double sumValue = 0.0;
    for (const auto& row : result) {
        for (double val : row) {
            sumValue += val;
        }
    }

    cout << "Matrix multiplication of " << size << "x" << size << " matrices (C++)" << endl;
    cout << "Result sum: " << sumValue << endl;
    cout << "Execution Time: " << executionTime << " seconds" << endl;

    return 0;
}
