#include "matrix_ops.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace lab {

namespace {
Matrix create_zero_matrix(std::size_t n) {
    return Matrix(n, std::vector<double>(n, 0.0));
}
} // namespace

std::optional<Matrix> create_random_matrix(std::size_t n) {
    if (n == 0) return std::nullopt;
    Matrix m = create_zero_matrix(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (auto& row : m) for (auto& val : row) val = dis(gen);
    return m;
}

Matrix multiply_matrices(const Matrix& A, const Matrix& B, LoopOrder order) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) return {};
    std::size_t n = A.size();
    
    Matrix C = create_zero_matrix(n);

    switch (order) {
        case LoopOrder::IJK:
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t k = 0; k < n; ++k)
                        C[i][j] += A[i][k] * B[k][j];
            break;
        case LoopOrder::IKJ:
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = 0; k < n; ++k) {
                    double a = A[i][k];
                    for (std::size_t j = 0; j < n; ++j)
                        C[i][j] += a * B[k][j];
                }
            break;
        case LoopOrder::JIK:
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t k = 0; k < n; ++k)
                        C[i][j] += A[i][k] * B[k][j];
            break;
        case LoopOrder::JKI:
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = 0; k < n; ++k) {
                    double b = B[k][j];
                    for (std::size_t i = 0; i < n; ++i)
                        C[i][j] += A[i][k] * b;
                }
            break;
        case LoopOrder::KIJ:
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t i = 0; i < n; ++i) {
                    double a = A[i][k];
                    for (std::size_t j = 0; j < n; ++j)
                        C[i][j] += a * B[k][j];
                }
            break;
        case LoopOrder::KJI:
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t j = 0; j < n; ++j) {
                    double b = B[k][j];
                    for (std::size_t i = 0; i < n; ++i)
                        C[i][j] += A[i][k] * b;
                }
            break;
    }
    return C;
}

Matrix multiply_matrices_threaded(const Matrix& A, const Matrix& B) {
    std::size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    std::size_t rows_per_thread = n / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        std::size_t start_row = t * rows_per_thread;
        std::size_t end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        
        threads.emplace_back([&, start_row, end_row]() {
            for (std::size_t i = start_row; i < end_row; ++i) {
                for (std::size_t k = 0; k < n; ++k) {
                    double a_ik = A[i][k];
                    for (std::size_t j = 0; j < n; ++j) {
                        C[i][j] += a_ik * B[k][j];
                    }
                }
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    return C;
}

Matrix multiply_matrices_threaded_atomic(const Matrix& A, const Matrix& B) {
    std::size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    std::size_t rows_per_thread = n / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        std::size_t start_row = t * rows_per_thread;
        std::size_t end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        
        threads.emplace_back([&, start_row, end_row]() {
            while (!start_flag.load()) {
                std::this_thread::yield();
            }
            
            // Правильный (быстрый) порядок (i, k, j)
            for (std::size_t i = start_row; i < end_row; ++i) {
                for (std::size_t k = 0; k < n; ++k) {
                    double a_ik = A[i][k];
                    for (std::size_t j = 0; j < n; ++j) {
                        C[i][j] += a_ik * B[k][j];
                    }
                }
            }
        });
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    start_flag.store(true);
    
    for (auto& th : threads) {
        th.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    std::cout << "Time (atomic threads, fast order): " << dur.count() << " ms\n";
    
    return C;
}

} // namespace lab