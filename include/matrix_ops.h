#pragma once
#include <cstddef>
#include <optional>
#include <vector>

namespace lab {

using Matrix = std::vector<std::vector<double>>;
enum class LoopOrder { IJK, IKJ, JIK, JKI, KIJ, KJI };

std::optional<Matrix> create_random_matrix(std::size_t n);
Matrix multiply_matrices(const Matrix& A, const Matrix& B, LoopOrder order);
Matrix multiply_matrices_threaded(const Matrix& A, const Matrix& B);
Matrix multiply_matrices_threaded_atomic(const Matrix& A, const Matrix& B);

} // namespace lab