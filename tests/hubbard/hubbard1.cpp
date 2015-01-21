#include "hubbard.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int L = 100;

int main (int argc, char **argv) {
	std::mt19937_64 generator;
	MatrixXd A = MatrixXd::Random(L, L);
	MatrixXd H(2*L, 2*L);
	H.topLeftCorner(L, L) = A * A.transpose();
	H.bottomRightCorner(L, L) = A.transpose() * A;
	SelfAdjointEigenSolver<MatrixXd> es(H);
	HubbardInteraction I;
	I.setup(4.0, 5.0);
	I.set_lattice_eigenvectors(es.eigenvectors());
	A.setIdentity(2*L, 2*L);
	HubbardInteraction::Vertex v = I.generate(generator);
	I.apply_vertex_on_the_left(v, A);
	I.apply_inverse_on_the_left(v, A);
	if (!A.isIdentity()) {
		cerr << "multiplying a vertex and its inverse on the left does not result in the identity" << endl;
		cerr << A << endl;
		return 1;
	}
	I.apply_vertex_on_the_right(v, A);
	I.apply_inverse_on_the_right(v, A);
	if (!A.isIdentity()) {
		cerr << "multiplying a vertex and its inverse on the right does not result in the identity" << endl;
		cerr << A << endl;
		return 1;
	}
	return 0;
}

