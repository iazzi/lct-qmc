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
	Parameters params(argc, argv);
	std::mt19937_64 generator;
	std::random_device rd;
	uniform_int_distribution<unsigned int> idist(0, UINT_MAX);
	generator.seed(idist(rd));
	SpinOneHalf<CubicLattice> lattice(params);
	lattice.compute();
	HubbardInteraction interaction(generator);
	interaction.setup(lattice.eigenvectors(), 4.0, 5.0);
	auto model = make_model(lattice, interaction);
	Configuration<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>> conf(generator, model);
	conf.setup(20.0, 0.0, N); // beta, mu (relative to half filling), slice number
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<lattice.volume();j++) {
			HubbardInteraction::Vertex v = interaction.generate(0.0, conf.slice_end()-conf.slice_start());
			v.sigma = (j%2?-1:+1)*fabs(v.sigma);
			conf.insert(v);
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
		for (int j=0;j<lattice.volume();j++) {
			HubbardInteraction::Vertex v = interaction.generate(0.0, conf.slice_end()-conf.slice_start());
			v.sigma = -v.sigma;
			pr += std::log(std::fabs(conf.insert_probability(v)));
			cerr << "inserted vertex " << v.tau << ' ' << v.sigma << endl;
			conf.insert_and_update(v);
		}
		conf.commit_changes();
		cerr << conf.check_B() << endl;
		conf.compute_G();
		cerr << "dG = " << conf.check_and_save_G() << endl;
		double p2 = conf.probability().first;
		std::cerr << "dp = " << pr-p2+p1 << endl;
	}
	return 0;
}



