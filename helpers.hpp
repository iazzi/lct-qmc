#ifndef __HELPERS_HPP
#define __HELPERS_HPP

#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXd reduce_f (const std::vector<Eigen::MatrixXd>& vec);
Eigen::MatrixXd reduce_b (const std::vector<Eigen::MatrixXd>& vec);

void test_sequences (std::vector<Eigen::MatrixXd>& fvec, std::vector<Eigen::MatrixXd>& bvec);
void dggev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta);

void sort_vector (Eigen::VectorXcd &v);
void reverse_vector (Eigen::VectorXcd &v);

#endif // __HELPERS_HPP

