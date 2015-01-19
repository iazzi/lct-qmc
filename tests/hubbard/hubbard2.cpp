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
	std::mt19937_64 generator;
	Parameters params(argc, argv);
	SpinOneHalf<CubicLattice> lattice(params);
	lattice.compute();
	HubbardInteraction interaction(generator);
	interaction.setup(lattice.eigenvectors(), 4.0, 5.0);
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


