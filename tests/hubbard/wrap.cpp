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
const int N = 80;

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
	Configuration<Model<CubicLattice, HubbardInteraction>> conf(generator, model);
	conf.setup(20.0, 0.0, N); // beta, mu (relative to half filling), slice number
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<L*L;j++) {
			conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start()));
		}
		//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	}
	for (int i=0;i<N;i++) {
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



