#ifndef __HELPERS_HPP
#define __HELPERS_HPP

#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXd reduce_f (const std::vector<Eigen::MatrixXd>& vec);
Eigen::MatrixXd reduce_b (const std::vector<Eigen::MatrixXd>& vec);

void test_sequences (std::vector<Eigen::MatrixXd>& fvec, std::vector<Eigen::MatrixXd>& bvec);

#endif // __HELPERS_HPP

