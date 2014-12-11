#include "configuration.hpp"
#include "cubiclattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

#include <algorithm>

using namespace std;
using namespace Eigen;

const int L = 10;

double relative_error (double a, double b) {
	return fabs(a-b)/min(fabs(a), fabs(b));
}

int main () {
	std::mt19937_64 generator;
	CubicLattice lattice;
	lattice.set_size(L, L, 1);
	lattice.compute();
	HubbardInteraction interaction(generator);
	interaction.setup(lattice.eigenvectors(), 4.0, 5.0);
	auto model = make_model(lattice, interaction);
	Slice<Model<CubicLattice, HubbardInteraction>> slice(model);
	for (int i=0;i<L*L;i++) {
		slice.insert(interaction.generate(0.0, 1.0));
	}
	HubbardInteraction::Vertex v = interaction.generate(0.0, 1.0);
	MatrixXd A = slice.matrix();
	Slice<Model<CubicLattice, HubbardInteraction>>::UpdateType U = slice.matrixU(v);
	Slice<Model<CubicLattice, HubbardInteraction>>::UpdateType Vt = slice.matrixVt(v);
	slice.insert(v);
	MatrixXd B = slice.matrix();
	if (!B.isApprox(A+U*Vt.transpose()*A)) {
		cerr << A+U*Vt.transpose()*A << endl << endl;
		cerr << B << endl;
		return 1;
	}
	return 0;
}



