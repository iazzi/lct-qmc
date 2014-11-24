#include "hubbard.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int L = 100;

int main () {
	std::mt19937_64 generator;
	MatrixXd A = MatrixXd::Random(L, L);
	MatrixXd H = A * A.transpose();
	SelfAdjointEigenSolver<MatrixXd> es(H);
	HubbardInteraction I(generator);
	I.setup(es.eigenvectors(), 4.0, 5.0);
	A.setIdentity(L, L);
	HubbardInteraction::Vertex v = I.generate();
	I.apply_vertex_on_the_left(v, A);
	I.apply_inverse_on_the_left(v, A);
	if (!A.isIdentity()) return 1;
	I.apply_vertex_on_the_right(v, A);
	I.apply_inverse_on_the_right(v, A);
	if (!A.isIdentity()) return 1;
	return 0;
}

