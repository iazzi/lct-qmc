#include "measurements.hpp"
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
	size_t thermalization = params.getInteger("thermalization", 10000);
	size_t sweeps = params.getInteger("sweeps", 10000);
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
	//ofstream diff ("diff.dat", ios::app);
	//diff << "\"V=" << model.interaction().dimension() << " beta=" << conf.inverse_temperature() << "\"" << endl;
	conf.compute_B();
	conf.compute_G();
	conf.save_G();
	double p1 = 0.0, ps = 0.0, pr = 0.0;
	std::tie(p1, ps) = conf.probability();
	auto sweep = [&p1, &conf, &d, &trial, &pr, &ps, &model] (mt19937_64 &generator, bool check) {
		HubbardInteraction::Vertex v;
		for (size_t j=0;j<model.lattice().volume();j++) {
			double dp = 0.0, s = 1.0;
			if (d(generator)<0.5) {
				v = conf.get_vertex(d(generator)*conf.slice_size());
				dp = conf.remove_probability(v);
				s = dp>0.0?1.0:-1.0;
				dp = std::log(std::fabs(dp));
				if (-trial(generator)<dp+conf.remove_factor()) {
					//cerr << "removed vertex " << v.tau << endl;
					conf.remove_and_update(v);
					pr += dp;
					ps *= s;
				} else {
					//cerr << "remove rejected" << endl;
				}
			} else {
				v = model.interaction().generate(0.0, conf.slice_end()-conf.slice_start(), generator);
				dp = conf.insert_probability(v);
				s = dp>0.0?1.0:-1.0;
				dp = std::log(std::fabs(dp));
				if (-trial(generator)<dp+conf.insert_factor()) {
					//cerr << "inserted vertex " << v.tau << endl;
					conf.insert_and_update(v);
					pr += dp;
					ps *= s;
				} else {
					//cerr << "insert rejected" << endl;
				}
			}
			if (check) {
				conf.compute_B();
				double p2 = conf.probability().first;
				std::cerr << "v = " << v.x << ',' << v.tau << " dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
			}
		}
		conf.compute_right_side();
	};
	auto full_check = [&conf, &model] () {
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
	};
	auto full_sweep = [&conf, &d, &trial, &pr, &ps, &model, &sweep] (mt19937_64 &generator, bool measure, measurement<double> &Sign, measurement<double> &Kin, measurement<double> &Int, measurement<double> &Verts, bool check) {
		Eigen::MatrixXd G;
		for (size_t i=0;i<conf.slice_number();i++) {
			conf.set_index(i);
			conf.compute_right_side();
			//conf.compute_B();
			//conf.compute_G();
			//conf.save_G();
			//G = conf.green_function();
			conf.compute_propagators_2();
			if (measure) {
				Sign.add(ps);
				Kin.add(ps*model.lattice().kinetic_energy(conf.green_function())/model.lattice().volume());
				Int.add(ps*model.interaction().interaction_energy(conf.green_function())/model.lattice().volume());
				Verts.add(conf.size());
			}
			//std::cerr << G << std::endl << std::endl;
			//cerr << (double(i)/conf.slice_number()) << ' '
				//<< conf.inverse_temperature() << ' '
				//<< model.interaction().dimension() << ' '
				//<< (conf.green_function()-G).norm() << ' '
				//<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
			sweep(generator, check);
		}
		for (size_t i=conf.slice_number();i>0;i--) {
			conf.set_index(i-1);
			conf.compute_left_side();
			//std::cerr << G << std::endl << std::endl;
			//conf.compute_B();
			//conf.compute_G();
			//conf.save_G();
			conf.compute_propagators_2();
			//G = conf.green_function();
			if (measure) {
				Sign.add(ps);
				Kin.add(ps*model.lattice().kinetic_energy(conf.green_function())/model.lattice().volume());
				Int.add(ps*model.interaction().interaction_energy(conf.green_function())/model.lattice().volume());
				Verts.add(conf.size());
			}
			//cerr << (double(i-1)/conf.slice_number()) << ' '
			//<< conf.inverse_temperature() << ' '
			//<< model.interaction().dimension() << ' '
			//<< (conf.green_function()-G).norm() << ' '
			//<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
			sweep(generator, check);
		}
		if (check) {
			conf.start();
			conf.check_all_prop();
			//conf.check_all_det(0);
			//conf.check_all_det(1);
		}
	};
	measurement<double> Sign("Sign");
	measurement<double> Dens("Density");
	measurement<double> Kin("Kinetic Energy");
	measurement<double> Int("Interaction Energy");
	measurement<double> Verts("Vertices");
	for (size_t i=0;i<thermalization+sweeps;i++) {
		full_sweep(generator, i>=thermalization, Sign, Kin, Int, Verts, false);
		if (i>=thermalization) {
			if (i%100==0) cerr << endl << Kin << endl << Int << endl << Sign << endl;
		} else if (i%100==0) {
			cerr << ' ' << (100.0*i/thermalization) << "%         \r";
		}
		conf.compute_B();
		double p2 = conf.probability().first;
		std::cerr << i << " dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl;
	}
	//diff << endl << endl;
	conf.compute_B();
	double p2 = conf.probability().first;
	cerr << endl << Kin << endl << Int << endl << Sign << endl;
	std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
	return 0;
}




