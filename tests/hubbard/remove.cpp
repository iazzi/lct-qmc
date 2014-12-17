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
	std::uniform_real_distribution<double> d;
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
	for (size_t i=0;i<conf.slice_number();i++) {
		double pr = 0.0;
		conf.set_index(i);
		conf.compute_B();
		conf.compute_G();
		conf.save_G();
		double p1 = conf.probability().first;
		for (int j=0;j<2;j++) {
			size_t n = d(generator)*conf.slice_size();
			HubbardInteraction::Vertex v = conf.get_vertex(n);
			pr += std::log(std::fabs(conf.remove_probability(v)));
			cerr << "vertex " << n << '/' << conf.slice_size() << ' ' << v.tau << endl;
			cerr << conf.remove_and_update(v) << endl;
		}
		conf.compute_B();
		conf.compute_G();
		cerr << conf.check_and_save_G() << endl;
		double p2 = conf.probability().first;
		std::cerr << pr-p2+p1 << endl;
	}
	return 0;
}



