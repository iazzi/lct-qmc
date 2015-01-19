#include "configuration.hpp"
#include "cubiclattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"
#include "spin_one_half.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

#include <algorithm>

using namespace std;
using namespace Eigen;

double relative_error (double a, double b) {
	return fabs(a-b)/min(fabs(a), fabs(b));
}

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
		cerr << "insert" << endl;
		slice.insert(interaction.generate(0.0, 1.0, generator));
	}
	HubbardInteraction::Vertex v = interaction.generate(0.0, 1.0, generator);
	MatrixXd A = slice.matrix();
	Slice<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>>::UpdateType U = slice.matrixU(v);
	Slice<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>>::UpdateType Vt = slice.matrixVt(v);
	slice.insert(v);
	MatrixXd B = slice.matrix();
	if (!B.isApprox(A+U*Vt.transpose()*A)) {
		cerr << A+U*Vt.transpose()*A << endl << endl;
		cerr << B << endl;
		return 1;
	}
	return 0;
}



