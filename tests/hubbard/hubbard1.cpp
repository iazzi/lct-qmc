#include "hubbard.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int L = 4;

int main () {
	std::mt19937_64 generator;
	MatrixXd A = MatrixXd::Random(L, L);
	MatrixXd H = A * A.transpose();
	SelfAdjointEigenSolver<MatrixXd> es(H);
	cerr << "solved\n";
	HubbardInteraction I(generator);
	I.setup(es.eigenvectors(), 4.0, 5.0);
	cerr << "setup\n";
	A.setIdentity(L, L);
	HubbardInteraction::Vertex v = I.generate();
	cerr << "generated\n";
	I.apply_vertex_on_the_left(v, A);
	I.apply_inverse_on_the_left(v, A);
	cout << A << endl << endl;
	I.apply_vertex_on_the_right(v, A);
	I.apply_inverse_on_the_right(v, A);
	cout << A << endl << endl;
	return 0;
}

