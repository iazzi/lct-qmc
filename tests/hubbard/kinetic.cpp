#include "measurements.hpp"
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


double relative_error (double a, double b) {
	return fabs(a-b)/min(fabs(a), fabs(b));
}

int main (int argc, char **argv) {
	std::mt19937_64 generator;
	Parameters params(argc, argv);
	double beta = params.getNumber("beta", 5.0);
	size_t thermalization = params.getInteger("thermalization", 10000);
	size_t sweeps = params.getInteger("sweeps", 10000);
	std::uniform_real_distribution<double> d;
	std::exponential_distribution<double> trial;
	SpinOneHalf<CubicLattice> lattice(params);
	HubbardInteraction interaction(params);
	auto model = make_model(lattice, interaction);
	Configuration<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>> conf(model);
	conf.setup(beta, 0.0, 4*beta); // beta, mu (relative to half filling), slice number
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<lattice.volume();j++) {
			conf.insert(model.interaction().generate(0.0, conf.slice_end()-conf.slice_start(), generator));
		}
		//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	}
	conf.set_index(0);
	conf.compute_B();
	conf.compute_G();
	conf.save_G();
	double p1 = conf.probability().first;
	double pr = 0.0;
	auto sweep = [&generator, &conf, &lattice, &d, &trial, &pr, &model](size_t M) {
		for (size_t i=0;i<M;i++) {
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
					v = model.interaction().generate(0.0, conf.slice_end()-conf.slice_start(), generator);
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
			//conf.compute_G();
			//cerr << "dG = " << conf.check_and_save_G() << ", ";
			if (i+1>=M) {
				conf.advance(1);
				conf.compute_B();
			} else {
				conf.commit_changes();
				conf.wrap_B();
			}
			conf.compute_G_alt();
			//double p2 = conf.probability().first;
			//std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
		}
	};
	measurement<double> Kin;
	measurement<double> Verts;
	for (size_t i=0;i<thermalization+sweeps;i++) {
		sweep(1);
		if (i>=thermalization) {
			Kin.add(lattice.kinetic_energy(conf.green_function())/lattice.volume());
			Verts.add(conf.size());
			if (i%100==0) cerr << endl << Kin << endl << Verts << endl;
		} else if (i%100==0) {
			cerr << ' ' << (100.0*i/thermalization) << "%         \r";
			//Verts.add(conf.size());
			//cerr << endl << Verts << endl;
		}
	}
	double p2 = conf.probability().first;
	std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
	conf.show_verts();
	return 0;
}

