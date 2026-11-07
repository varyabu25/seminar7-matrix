#include "matrix_ops.h"
#include <iostream>
#include <chrono>

int main() {
    std::size_t n = 800;
    auto A = lab::create_random_matrix(n);
    auto B = lab::create_random_matrix(n);
    
    if (!A || !B) return 1;

    std::cout << "Multiplying (FINAL: atomic threads, fast order i,k,j)...\n";
    auto start = std::chrono::high_resolution_clock::now();
    lab::Matrix C = lab::multiply_matrices_threaded_atomic(*A, *B);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> dur = end - start;
    std::cout << "Total time main: " << dur.count() << " ms\n";
    std::cout << "C[0][0] check: " << C[0][0] << "\n";
    
    return 0;
}