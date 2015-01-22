#include "cubiclattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"
#include "spin_one_half.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main (int argc, char **argv) {
	Parameters params(argc, argv);
	std::mt19937_64 generator;
	SpinOneHalf<CubicLattice> lattice(params);
	HubbardInteraction interaction;
	interaction.setup(params);
	interaction.set_lattice_eigenvectors(lattice.eigenvectors());
	auto model = make_model(lattice, interaction);
	Slice<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>> slice(model);
	for (size_t i=0;i<lattice.volume();i++) {
		slice.insert(interaction.generate(generator));
	}
	MatrixXd A = slice.matrix() * slice.inverse();
	if (!A.isIdentity()) {
		cerr << "slice.matrix() and slice.inverse() are not actually inverses" << endl;
		cerr << A << endl;
		return 1;
	}
	return 0;
}


