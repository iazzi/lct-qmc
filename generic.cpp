#include "lctsimulation.hpp"
#include "measurements.hpp"
#include "configuration.hpp"
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
	void measure (const LCTSimulation &sim) {
		double sign = sim.sign();
		Sign.add(sign);
		cache = sim.green_function();
		Dens.add(sign*cache);
		Kin.add(sign*sim.kinetic_energy(cache)/sim.volume());
		Int.add(sign*sim.interaction_energy(cache)/sim.volume());
		Verts.add(sim.vertices());
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

int main (int argc, char **argv) {
	if (true) {
		Parameters params(argc, argv);
		size_t thermalization = params.getInteger("thermalization", 1000);
		size_t sweeps = params.getInteger("sweeps", 1000);
		LCTSimulation sim(params);
		Measurements measurements;
		for (size_t i=0;i<thermalization+sweeps;i++) {
			//sim.full_sweep(false);
			for (size_t j=0;j<sim.full_sweep_size();j++) {
				//std::cerr << "dp = " << sim.exact_probability()-sim.probability() << ' ' << sim.probability_difference() << ' ' << j << ' ' << sim.is_direction_right_to_left() << endl << endl;
				sim.prepare();
				sim.sweep();
				sim.next();
				if (i>=thermalization) {
					measurements.measure(sim);
				}
			}
			if (i>=thermalization) {
				if (i%100==0) cerr << endl << measurements.Kin << endl << measurements.Int << endl << measurements.Sign << endl;
			} else if (i%100==0) {
				cerr << ' ' << (100.0*i/thermalization) << "%         \r";
			}
			//conf.compute_B();
			//double p2 = conf.probability().first;
			//std::cerr << i << " dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << endl;
		}
		double p2 = sim.exact_probability();
		cerr << endl << measurements.Kin << endl << measurements.Int << endl << measurements.Sign << endl;
		std::cerr << "dp = " << sim.exact_probability()-sim.probability() << ' ' << sim.probability() << endl << endl;
		ofstream out("gf.dat");
		measurements.write_G(out);
	} else {
	}
	return 0;
}


