#ifndef LCTSIMULATION
#define LCTSIMULATION

#include "configuration.hpp"
#include "genericlattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"
#include "spin_one_half.hpp"

#include <random>

class LCTSimulation {
	public:
	
	typedef enum {
		right_to_left = 0,
		left_to_right = 1
	} sweep_direction_type;

	private:

	std::mt19937_64 generator;
	std::uniform_real_distribution<double> d;
	std::exponential_distribution<double> trial;
	typedef HubbardInteraction<> Interaction;
	Configuration<Model<SpinOneHalf<GenericLattice>, Interaction>> conf;
	double p1; // probability at the start of the simulation (absolute value)
	double pr; // probability ration of the current configuration wrt p1 (absolute values)
	double ps; // sign of the current configuration

	sweep_direction_type sweep_direction_;
	size_t updates_;

	public:

	LCTSimulation (Parameters params, size_t seed_offset=0) :
		generator(params.getInteger("SEED",42)+seed_offset),
		conf(params),
		sweep_direction_(right_to_left),
		updates_(0) {
			conf.setup(params);
			for (size_t i=0;i<conf.slice_number();i++) {
				conf.set_index(i);
				for (size_t j=0;j<2*conf.volume();j++) {
					conf.insert(conf.generate_vertex(generator));
				}
				//std::cerr << i << " -> " << conf.slice_size() << std::endl;
			}
			conf.set_index(0);
			conf.compute_right_side(0);
			conf.start();
			conf.start();
			conf.compute_B();
			p1 = 0.0, ps = 0.0, pr = 0.0;
			std::tie(p1, ps) = conf.probability();
			conf.set_index(0);
			conf.compute_propagators_2_right();
		}

	void update_left (bool check = false) {
		Interaction::Vertex v;
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
			v = conf.generate_vertex(generator);
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
			//conf.compute_B();
			//double p2 = conf.probability().first;
			//std::cerr << "v = " << v.x << ',' << v.tau << " dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << std::endl << endl;
		}
		//conf.compute_right_side(conf.current_slice()+1);
		updates_++;
	}

	void update_right (bool check = false) {
		Interaction::Vertex v;
		double dp = 0.0, s = 1.0;
		if (d(generator)<0.5) {
			v = conf.get_vertex(d(generator)*conf.slice_size());
			dp = conf.remove_probability_right(v);
			s = dp>0.0?1.0:-1.0;
			dp = std::log(std::fabs(dp));
			if (-trial(generator)<dp+conf.remove_factor()) {
				//cerr << "removed vertex " << v.tau << endl;
				conf.remove_and_update_right(v);
				pr += dp;
				ps *= s;
			} else {
				//cerr << "remove rejected" << endl;
			}
		} else {
			v = conf.generate_vertex(generator);
			dp = conf.insert_probability_right(v);
			s = dp>0.0?1.0:-1.0;
			dp = std::log(std::fabs(dp));
			if (-trial(generator)<dp+conf.insert_factor()) {
				//cerr << "inserted vertex " << v.tau << endl;
				conf.insert_and_update_right(v);
				pr += dp;
				ps *= s;
			} else {
				//cerr << "insert rejected" << endl;
			}
		}
		if (check) {
			//conf.compute_B();
			//double p2 = conf.probability().first;
			//std::cerr << "v = " << v.x << ',' << v.tau << " dp = " << p1+pr-p2 << ' ' << p2-p1 << ' ' << pr << std::endl << endl;
		}
		//conf.compute_right_side(conf.current_slice()+1);
		updates_++;
	}

	void prepare () {
		if (is_direction_right_to_left()) {
		} else {
		}
	}

	void sweep (bool check = false) {
		Interaction::Vertex v;
		if (is_direction_right_to_left()) {
			for (size_t j=0;j<conf.volume();j++) {
				update_right(check);
			}
		} else {
			for (size_t j=0;j<conf.volume();j++) {
				update_left(check);
			}
		}
		//conf.compute_right_side(conf.current_slice()+1);
	}

	void next () {
		if (is_direction_right_to_left()) {
			if (conf.current_slice()+1<conf.slice_number()) {
				conf.compute_right_side(conf.current_slice()+1);
				conf.set_index(conf.current_slice()+1);
				conf.compute_propagators_2_right();
			} else {
				//conf.set_index(conf.current_slice()-1);
				conf.compute_right_side(conf.current_slice()+1);
				conf.compute_propagators_2();
				set_direction_left_to_right();
			}
		} else {
			if (conf.current_slice()>0) {
				conf.compute_left_side(conf.current_slice());
				conf.set_index(conf.current_slice()-1);
				conf.compute_propagators_2();
			} else {
				//conf.set_index(conf.current_slice()+1);
				conf.compute_left_side(conf.current_slice());
				conf.compute_propagators_2_right();
				set_direction_right_to_left();
			}
		}
	}

	void full_sweep (bool check = false) {
		set_direction_right_to_left();
		for (size_t i=0;i<conf.slice_number();i++) {
			conf.set_index(i);
			conf.compute_right_side(conf.current_slice()+1);
			conf.compute_propagators_2();
			sweep(check);
			conf.compute_right_side(conf.current_slice()+1);
		}
		set_direction_left_to_right();
		for (size_t i=conf.slice_number();i>0;i--) {
			conf.set_index(i-1);
			conf.compute_left_side(conf.current_slice()+1);
			conf.compute_propagators_2();
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
	double exact_probability () { conf.compute_B(); return conf.probability().first; }
	double probability_difference () const { return pr; }

	size_t vertices () const { return conf.vertices(); }
	const Eigen::MatrixXd & green_function () const {
		return conf.green_function();
	}

	double kinetic_energy (const Eigen::MatrixXd& cache) const {
		return conf.kinetic_energy(cache);
	}

	double interaction_energy (const Eigen::MatrixXd& cache) const {
		return conf.interaction_energy(cache);
	}

	size_t volume () const { return conf.volume(); }
	size_t full_sweep_size () const { return 2*conf.slice_number(); }

	sweep_direction_type sweep_direction () const { return sweep_direction_; }
	bool is_direction_left_to_right () const { return sweep_direction_==left_to_right; }
	bool is_direction_right_to_left () const { return sweep_direction_==right_to_left; }
	void set_sweep_direction (sweep_direction_type d) { sweep_direction_ = d; }
	void set_direction_left_to_right () { sweep_direction_ = left_to_right; }
	void set_direction_right_to_left () { sweep_direction_ = right_to_left; }
};

#endif // LCTSIMULATION

