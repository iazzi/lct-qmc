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

const int L = 1;
const int N = 10;

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
	//for (size_t i=0;i<conf.slice_number();i++) {
	//	conf.set_index(i);
	//	for (size_t j=0;j<L*L;j++) {
	//		conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start()));
	//	}
	//	//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	//}
	for (int i=0;i<N;i++) {
                std::cerr << i << endl;
		conf.set_index(i);
                conf.compute_B();
        }
	for (int i=0;i<N;i++) {
		conf.set_index(i);
                double err = conf.check_B();
	        std::cerr << i << " " << err << endl;
		//conf.wrap_B();
	}
	return 0;
}



