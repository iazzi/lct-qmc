#include "cubiclattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int L = 10;

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
		slice.insert(interaction.generate());
	}
	MatrixXd A = slice.matrix() * slice.inverse();
	if (!A.isIdentity()) {
		cerr << "slice.matrix() and slice.inverse() are not actually inverses" << endl;
		cerr << A << endl;
		return 1;
	}
	return 0;
}


