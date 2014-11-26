#include "configuration.hpp"
#include "cubiclattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int L = 100;

int main () {
	std::mt19937_64 generator;
	CubicLattice lattice;
	lattice.set_size(4, 4, 1);
	lattice.compute();
	HubbardInteraction interaction(generator);
	interaction.setup(lattice.eigenvectors(), 4.0, 5.0);
	auto model = make_model(lattice, interaction);
	Configuration<Model<CubicLattice, HubbardInteraction>> conf(generator, model);
	conf.setup(20.0, 0.0, 40); // beta, mu (relative to half filling), slice number
	for (int i=0;i<20*L;i++) {
		conf.insert(interaction.generate());
	}
	return 0;
}



