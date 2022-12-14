
#include "lctaux_sim.hpp"

#include "alps/params/convenience_params.hpp"
#include <alps/mc/api.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lexical_cast.hpp>

#include "type_save.hpp"

using namespace std;
using namespace Eigen;

Parameters convert_parameters(alps::parameters_type<lctaux_sim>::type const & parameters) {
    Parameters p(0, NULL);
    for(alps::parameters_type<lctaux_sim>::type::const_iterator it=parameters.begin();
          it!=parameters.end(); ++it) {
        if (parameters.exists(it->first)) {
            p.setString(it->first, boost::lexical_cast<std::string>(it->second));
        } else {
            std::cerr << "ERROR: Parameter " << it->first << " has not been defined!" << std::endl;
            parameters.print_help(std::cerr);
            throw std::runtime_error("Parameter not defined.");
        }
    }
    return p;
}


void lctaux_sim::define_parameters(parameters_type & parameters) {
	if (parameters.is_restored()) { return; }
	alps::mcbase::define_parameters(parameters);
	alps::define_convenience_parameters(parameters)
		.description("LCT-AUX Hubbard Model")
		.define<int>("sweeps", 1000, "maximum number of sweeps")
		.define<int>("thermalization", "number of sweeps for thermalization")
		.define<double>("beta", "inverse temperature of the system")
		.define<double>("U", 0., "local Hubbard interaction")
		.define<std::string>("H", "path to file containing the edge matrix")
		;
}


lctaux_sim::lctaux_sim(parameters_type const & parms, std::size_t seed_offset)
    : LCTSimulation(convert_parameters(parms), seed_offset)
    , alps::mcbase(parms, seed_offset)
    , sweeps_(0)
    , iteration_(0)
    , thermalization_sweeps_(int(parameters["thermalization"]))
    , total_sweeps_(int(parameters["sweeps"]))
{   
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Sign")
        << alps::accumulators::FullBinningAccumulator<std::vector<double> >("Local density") // TODO: missing hdf5 save/load
        << alps::accumulators::FullBinningAccumulator<double>("Density")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
        << alps::accumulators::FullBinningAccumulator<double>("Kinetic Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Interaction Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Vertices")
    ;
}

void lctaux_sim::update() {
    prepare();
    sweep();
    next();

    iteration_++;
    if (iteration_ >=full_sweep_size()) {
        sweeps_++;
        iteration_=0;
    }
}

void lctaux_sim::measure() {
    if (sweeps_ > thermalization_sweeps_) {
        MatrixXd cache = green_function();
        const double current_sign = sign();
        // pull in operator/ for vectors
        using alps::numeric::operator/;
        using alps::numeric::operator*;
        measurements["Sign"] << current_sign;
        measurements["Density"] << current_sign*cache.trace()/volume();
        measurements["Magnetization"] << current_sign*(cache.diagonal().head(volume())-cache.diagonal().tail(volume())).array().sum()/volume();
        measurements["Kinetic Energy"] << current_sign*kinetic_energy(cache)/volume();
        measurements["Interaction Energy"] << current_sign*interaction_energy(cache)/volume();
        measurements["Vertices"] << vertices();
        Eigen::ArrayXd d = local_density(cache);
        measurements["Local density"] << current_sign * std::vector<double>(d.data(), d.data()+d.rows()*d.cols());
	//for (int x=0;x<volume();x++) for (int y;y<volume();y++) {
	//}
    }
}

double lctaux_sim::fraction_completed() const {
    return (sweeps_ < thermalization_sweeps_ ? 0. : ( sweeps_ - thermalization_sweeps_ ) / double(total_sweeps_));
}

void lctaux_sim::save(alps::hdf5::archive & ar) const {
	mcbase::save(ar);
	ar["checkpoint/sweeps"] << sweeps_;
	ar["checkpoint/iteration"] << iteration_;
	ar["checkpoint/configuration"] << configuration();
	ar["checkpoint/current_slice"] << configuration().current_slice();
	ar["checkpoint/left_to_right"] << (is_direction_left_to_right()?true:false);
	std::ostringstream seed;
	seed << generator;
	ar["checkpoint/generator"] << seed.str();
}

void lctaux_sim::load(alps::hdf5::archive & ar) {
	mcbase::load(ar);
	ar["checkpoint/sweeps"] >> sweeps_;
	ar["checkpoint/iteration"] >> iteration_;
	ar["checkpoint/configuration"] >> configuration();
	reset();
	int cs;
	ar["checkpoint/current_slice"] >> cs;
	configuration().set_slice(cs);
	bool L2R;
	ar["checkpoint/left_to_right"] >> L2R;
	if (L2R) {
		set_direction_left_to_right();
		configuration().compute_propagators_2_right();
	} else {
		set_direction_right_to_left();
		configuration().compute_propagators_2_right();
	}
	std::string s;
	ar["checkpoint/generator"] >> s;
	std::istringstream seed(s);
	seed >> generator;
}
//
// void lctaux_sim::load(alps::hdf5::archive & ar) {
//     mcbase::load(ar);
//
//     length = int(parameters["L"]);
//     thermalization_sweeps = int(parameters["THERMALIZATION"]);
//     total_sweeps = int(parameters["SWEEPS"]);
//     beta = 1. / double(parameters["T"]);
//
//     ar["checkpoint/sweeps"] >> sweeps;
//     ar["checkpoint/spins"] >> spins;
// }

void lctaux_sim::print (std::ostream &out) const {
	configuration().print(out);
}
