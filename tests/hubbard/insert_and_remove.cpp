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
int N = 40;

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
		for (int j=0;j<L*L;j++) {
			HubbardInteraction::Vertex v;
			v = conf.get_vertex(d(generator)*conf.slice_size());
			pr += std::log(std::fabs(conf.remove_probability(v)));
			cerr << "removed vertex " << v.tau << endl;
			conf.remove_and_update(v);
			v = interaction.generate(0.0, conf.slice_end()-conf.slice_start());
			pr += std::log(std::fabs(conf.insert_probability(v)));
			cerr << "inserted vertex " << v.tau << endl;
			conf.insert_and_update(v);
		}
		conf.compute_B();
		conf.compute_G();
		cerr << "dG = " << conf.check_and_save_G() << ", ";
		double p2 = conf.probability().first;
		std::cerr << "dp = " << pr-p2+p1 << endl;
	}
	return 0;
}



