#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <vector>
#include <Eigen/Dense>

#include "types.hpp"

Matrix_d reduce_f (const std::vector<Matrix_d>& vec);
Matrix_d reduce_b (const std::vector<Matrix_d>& vec);

void test_sequences (std::vector<Matrix_d>& fvec, std::vector<Matrix_d>& bvec);
void dggev (const Matrix_d &A, const Matrix_d &B, Vector_cd &alpha, Vector_d &beta);

void sort_vector (Vector_cd &v);
void reverse_vector (Vector_cd &v);

#endif // HELPERS_HPP

