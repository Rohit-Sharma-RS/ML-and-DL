```cpp
#include <iostream>
#include <chrono>

double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        if (j < 0) {
            break;
        }
        result -= (1.0 / j);
        j = i * param1 + param2;
        if (j >= iterations) {
            break;
        }
        result += (1.0 / j);
    }
    return result;
}

int main() {
    const double iterations = 10_000_000;
    double param1 = 4;
    double param2 = 1;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = calculate(iterations, param1, param2) * param2;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Result: " << result << "\n";
    std::cout << "Execution Time: "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6.0) << " ms\n";

    return 0;
}
```