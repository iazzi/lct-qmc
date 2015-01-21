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
#include <cstdlib>

using namespace std;
using namespace Eigen;

int N = 80;

double relative_error (double a, double b) {
	return fabs(a-b)/min(fabs(a), fabs(b));
}

int main (int argc, char **argv) {
	Parameters params(argc, argv);
	std::mt19937_64 generator;
	std::uniform_real_distribution<double> d;
	std::exponential_distribution<double> trial;
	SpinOneHalf<CubicLattice> lattice(params);
	lattice.compute();
	HubbardInteraction interaction;
	interaction.setup(params);
	auto model = make_model(lattice, interaction);
	Configuration<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>> conf(model);
	conf.setup(20.0, 0.0, N); // beta, mu (relative to half filling), slice number
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<lattice.volume();j++) {
			conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start(), generator));
		}
		//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	}
	conf.set_index(0);
	conf.compute_B();
	conf.compute_G();
	conf.save_G();
	double p1 = conf.probability().first;
	double pr = 0.0;
	for (size_t i=0;i<conf.slice_number();i++) {
		HubbardInteraction::Vertex v;
		for (size_t j=0;j<lattice.volume();j++) {
			double dp = 0.0;
			if (d(generator)<0.5) {
				v = conf.get_vertex(d(generator)*conf.slice_size());
				dp = std::log(std::fabs(conf.remove_probability(v)));
				if (-trial(generator)<dp+conf.remove_factor()) {
					//cerr << "removed vertex " << v.tau << endl;
					conf.remove_and_update(v);
					pr += dp;
				} else {
					//cerr << "remove rejected" << endl;
				}
			} else {
				v = interaction.generate(0.0, conf.slice_end()-conf.slice_start(), generator);
				dp = std::log(std::fabs(conf.insert_probability(v)));
				if (-trial(generator)<dp+conf.insert_factor()) {
					//cerr << "inserted vertex " << v.tau << endl;
					conf.insert_and_update(v);
					pr += dp;
				} else {
					//cerr << "insert rejected" << endl;
				}
			}
		}
		conf.commit_changes();
		//conf.compute_G();
		//cerr << "dG = " << conf.check_and_save_G() << ", ";
		conf.wrap_B();
		if (0==(i+1)%9) std::cerr << i << " dB = " << conf.check_B() << endl;
		conf.compute_G();
		conf.save_G();
		//double p2 = conf.probability().first;
		//std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
	}
	cerr << "dB = " << conf.check_B() << endl;
	double p2 = conf.probability().first;
	std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
	return 0;
}

