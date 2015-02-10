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
	Eigen::MatrixXd G;
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		conf.compute_propagators_2();
		G = conf.green_function();
		//std::cerr << G << std::endl << std::endl;
		conf.compute_B();
		conf.compute_G();
		conf.save_G();
		cerr << (double(i)/conf.slice_number()) << ' '
			<< conf.inverse_temperature() << ' '
			<< model.interaction().dimension() << ' '
			<< (conf.green_function()-G).norm() << ' '
			<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
	}
	diff << endl << endl;
	return 0;
}



