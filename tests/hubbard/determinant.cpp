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

int N = 80;

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
	N = params.getInteger("N");
	Configuration<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>> conf(generator, model);
	conf.setup(20.0, 0.0, N); // beta, mu (relative to half filling), slice number
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<lattice.volume();j++) {
			conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start(), generator));
		}
		//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	}
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		conf.compute_B();
		if (relative_error(conf.log_abs_det(), conf.slice_log_abs_det())>1e-8 && conf.log_abs_det()>1.0e-8) {
			cerr << relative_error(conf.log_abs_det(), conf.slice_log_abs_det()) << endl;
			cerr << conf.log_abs_det() << ' ' << conf.slice_log_abs_det() << endl;
			cerr << conf.log_abs_max() << endl;
			return 1;
		}
	}
	return 0;
}



