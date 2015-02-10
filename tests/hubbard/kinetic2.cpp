#include "configuration2.hpp"
#include "cubiclattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"
#include "spin_one_half.hpp"

#include <random>
#include <iostream>
#include <fstream>
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
	std::uniform_real_distribution<double> d;
	std::exponential_distribution<double> trial;
	Parameters params(argc, argv);
	SpinOneHalf<CubicLattice> lattice(params);
	HubbardInteraction interaction(params);
	auto model = make_model(lattice, interaction);
	N = params.getInteger("N");
	Configuration2<Model<SpinOneHalf<CubicLattice>, HubbardInteraction>> conf(model);
	conf.setup(params);
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<2*lattice.volume();j++) {
			conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start(), generator));
		}
		//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	}
	conf.set_index(0);
	conf.compute_right_side();
	conf.start();
	conf.start();
	cerr << conf.check_B_vs_last_right_side() << endl;
	ofstream diff ("diff.dat", ios::app);
	diff << "\"V=" << model.interaction().dimension() << " beta=" << conf.inverse_temperature() << "\"" << endl;
	conf.compute_B();
	conf.compute_G();
	conf.save_G();
	double p1 = conf.probability().first;
	double pr = 0.0;
	auto sweep = [&conf, &d, &trial, &pr, &model] (mt19937_64 &generator) {
		HubbardInteraction::Vertex v;
		for (size_t j=0;j<model.lattice().volume();j++) {
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
		conf.compute_right_side();
	};
	Eigen::MatrixXd G;
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		conf.compute_right_side();
		//conf.compute_B();
		//conf.compute_G();
		//conf.save_G();
		conf.compute_propagators_2();
		G = conf.green_function();
		//std::cerr << G << std::endl << std::endl;
		cerr << (double(i)/conf.slice_number()) << ' '
			<< conf.inverse_temperature() << ' '
			<< model.interaction().dimension() << ' '
			<< (conf.green_function()-G).norm() << ' '
			<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
		sweep(generator);
	}
	for (size_t i=conf.slice_number();i>0;i--) {
		conf.set_index(i-1);
		conf.compute_left_side();
		//std::cerr << G << std::endl << std::endl;
		//conf.compute_B();
		//conf.compute_G();
		//conf.save_G();
		conf.compute_propagators_2();
		G = conf.green_function();
		cerr << (double(i-1)/conf.slice_number()) << ' '
			<< conf.inverse_temperature() << ' '
			<< model.interaction().dimension() << ' '
			<< (conf.green_function()-G).norm() << ' '
			<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
		sweep(generator);
	}
	diff << endl << endl;
	conf.compute_B();
	double p2 = conf.probability().first;
	std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
	return 0;
}




