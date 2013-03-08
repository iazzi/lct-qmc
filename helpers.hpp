#ifndef __HELPERS_HPP
#define __HELPERS_HPP

#include <vector>
#include <Eigen/Dense>

typedef double Real;
typedef std::complex<Real> Complex;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Matrix_d;
typedef Eigen::Array<Real, Eigen::Dynamic, 1> Array_d;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector_d;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> Vector_cd;

Matrix_d reduce_f (const std::vector<Matrix_d>& vec);
Matrix_d reduce_b (const std::vector<Matrix_d>& vec);

void test_sequences (std::vector<Matrix_d>& fvec, std::vector<Matrix_d>& bvec);
void dggev (const Matrix_d &A, const Matrix_d &B, Vector_cd &alpha, Vector_d &beta);

void sort_vector (Vector_cd &v);
void reverse_vector (Vector_cd &v);

#endif // __HELPERS_HPP

