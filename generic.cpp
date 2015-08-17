#include "measurements.hpp"
#include "configuration2.hpp"
#include "genericlattice.hpp"
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

double relative_error (double a, double b) {
	return fabs(a-b)/min(fabs(a), fabs(b));
}


template <class Model>
class Measurements {
	MatrixXd cache;
	public:
	measurement<double> Sign;
	measurement<ArrayXXd> Dens;
	measurement<double> Kin;
	measurement<double> Int;
	measurement<double> Verts;
	vector<measurement<ArrayXXd>> gf;
	Measurements () : Sign("Sign"), Dens("Density"), Kin("Kinetic Energy"), Int("Interaction Energy"), Verts("Vertices") {}
	void measure (Model& model, const Configuration2<Model> &conf, double sign) {
		Sign.add(sign);
		cache = conf.green_function();
		Dens.add(sign*cache);
		Kin.add(sign*model.lattice().kinetic_energy(cache)/model.lattice().volume());
		Int.add(sign*model.interaction().interaction_energy(cache)/model.lattice().volume());
		Verts.add(conf.size());
		//const int D = 4;
		//int i = conf.current_slice();
		//gf.resize(D*conf.slice_number());
		//double dt = (conf.slice_end()-conf.slice_start())/D;
		//for (int j=0;j<D;j++) {
			//conf.gf_tau(cache, j*dt);
			//gf[D*i+j].add(cache);
		//}
	}
	void write_G (std::ostream &out) {
		for (size_t i=0;i<gf.size();i++) {
			out << gf[i].mean() << endl << endl;
		}
	}
};

class LCTSimulation {
	std::mt19937_64 generator;
	std::uniform_real_distribution<double> d;
	std::exponential_distribution<double> trial;
	SpinOneHalf<GenericLattice> lattice;
	HubbardInteraction interaction;
	Model<SpinOneHalf<GenericLattice>, HubbardInteraction> model;
	Configuration2<Model<SpinOneHalf<GenericLattice>, HubbardInteraction>> conf;
	double p1; // probability at the start of the simulation (absolute value)
	double pr; // probability ration of the current configuration wrt p1 (absolute values)
	double ps; // sign of the current configuration
	public:
	LCTSimulation (Parameters params) :
		lattice(params),
		interaction(params),
		model(lattice, interaction),
		conf(model) {
			conf.setup(params);
			for (size_t i=0;i<conf.slice_number();i++) {
				conf.set_index(i);
				for (size_t j=0;j<2*lattice.volume();j++) {
					conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start(), generator));
				}
				//std::cerr << i << " -> " << conf.slice_size() << std::endl;
			}
			conf.set_index(0);
			conf.compute_right_side(0);
			conf.start();
			conf.start();
			conf.compute_B();
			conf.compute_G();
			conf.save_G();
			p1 = 0.0, ps = 0.0, pr = 0.0;
			std::tie(p1, ps) = conf.probability();
		}

	void update (bool check = false) {
		HubbardInteraction::Vertex v;
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
		//conf.compute_right_side(conf.current_slice()+1);
	}

	void sweep (bool check = false) {
		HubbardInteraction::Vertex v;
		for (size_t j=0;j<model.lattice().volume();j++) {
			update(check);
		}
		//conf.compute_right_side(conf.current_slice()+1);
	}

	void full_sweep (bool check = false) {
		Eigen::MatrixXd G;
		for (size_t i=0;i<conf.slice_number();i++) {
			conf.set_index(i);
			conf.compute_right_side(conf.current_slice()+1);
			//conf.compute_B();
			//conf.compute_G();
			//conf.save_G();
			//G = conf.green_function();
			conf.compute_propagators_2();
			//std::cerr << G << std::endl << std::endl;
			//cerr << (double(i)/conf.slice_number()) << ' '
				//<< conf.inverse_temperature() << ' '
				//<< model.interaction().dimension() << ' '
				//<< (conf.green_function()-G).norm() << ' '
				//<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
			sweep(check);
			conf.compute_right_side(conf.current_slice()+1);
		}
		for (size_t i=conf.slice_number();i>0;i--) {
			conf.set_index(i-1);
			conf.compute_left_side(conf.current_slice()+1);
			//std::cerr << G << std::endl << std::endl;
			//conf.compute_B();
			//conf.compute_G();
			//conf.save_G();
			conf.compute_propagators_2();
			//G = conf.green_function();
			//cerr << (double(i-1)/conf.slice_number()) << ' '
			//<< conf.inverse_temperature() << ' '
			//<< model.interaction().dimension() << ' '
			//<< (conf.green_function()-G).norm() << ' '
			//<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
			sweep(check);
		}
		if (check) {
			conf.start();
			conf.check_all_prop();
			//conf.check_all_det(0);
			//conf.check_all_det(1);
		}
	}

	double probability () const { return p1+pr; }
	double sign () const { return ps; }

};

int main (int argc, char **argv) {
	std::mt19937_64 generator;
	std::uniform_real_distribution<double> d;
	std::exponential_distribution<double> trial;
	Parameters params(argc, argv);
	size_t thermalization = params.getInteger("thermalization", 1000);
	size_t sweeps = params.getInteger("sweeps", 1000);
	SpinOneHalf<GenericLattice> lattice(params);
	HubbardInteraction interaction(params);
	auto model = make_model(lattice, interaction);
	//int N = params.getInteger("N");
	Configuration2<Model<SpinOneHalf<GenericLattice>, HubbardInteraction>> conf(model);
	conf.setup(params);
	for (size_t i=0;i<conf.slice_number();i++) {
		conf.set_index(i);
		for (size_t j=0;j<2*lattice.volume();j++) {
			conf.insert(interaction.generate(0.0, conf.slice_end()-conf.slice_start(), generator));
		}
		//std::cerr << i << " -> " << conf.slice_size() << std::endl;
	}
	conf.set_index(0);
	conf.compute_right_side(0);
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
		//conf.compute_right_side(conf.current_slice()+1);
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
	Measurements<Model<SpinOneHalf<GenericLattice>, HubbardInteraction>> measurements;
	auto full_sweep = [&conf, &d, &trial, &pr, &ps, &model, &measurements, &sweep] (mt19937_64 &generator, bool measure, bool check) {
		Eigen::MatrixXd G;
		for (size_t i=0;i<conf.slice_number();i++) {
			conf.set_index(i);
			conf.compute_right_side(conf.current_slice()+1);
			//conf.compute_B();
			//conf.compute_G();
			//conf.save_G();
			//G = conf.green_function();
			conf.compute_propagators_2();
			if (measure) {
				measurements.measure(model, conf, ps);
			}
			//std::cerr << G << std::endl << std::endl;
			//cerr << (double(i)/conf.slice_number()) << ' '
				//<< conf.inverse_temperature() << ' '
				//<< model.interaction().dimension() << ' '
				//<< (conf.green_function()-G).norm() << ' '
				//<< (conf.green_function()-G).cwiseAbs().maxCoeff() << endl;
			sweep(generator, check);
			conf.compute_right_side(conf.current_slice()+1);
		}
		for (size_t i=conf.slice_number();i>0;i--) {
			conf.set_index(i-1);
			conf.compute_left_side(conf.current_slice()+1);
			//std::cerr << G << std::endl << std::endl;
			//conf.compute_B();
			//conf.compute_G();
			//conf.save_G();
			conf.compute_propagators_2();
			//G = conf.green_function();
			if (measure) {
				measurements.measure(model, conf, ps);
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
	for (size_t i=0;i<thermalization+sweeps;i++) {
		full_sweep(generator, i>=thermalization, false);
		if (i>=thermalization) {
			if (i%100==0) cerr << endl << measurements.Kin << endl << measurements.Int << endl << measurements.Sign << endl;
		} else if (i%100==0) {
			cerr << ' ' << (100.0*i/thermalization) << "%         \r";
		}
		//conf.compute_B();
		//double p2 = conf.probability().first;
		//std::cerr << i << " dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl;
	}
	//diff << endl << endl;
	conf.compute_B();
	double p2 = conf.probability().first;
	cerr << endl << measurements.Kin << endl << measurements.Int << endl << measurements.Sign << endl;
	std::cerr << "dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl << endl;
	ofstream out("gf.dat");
	measurements.write_G(out);
	return 0;
}




