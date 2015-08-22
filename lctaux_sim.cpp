
#include "lctaux_sim.hpp"

#include "alps/params/convenience_params.hpp"
#include <alps/mc/api.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lexical_cast.hpp>

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
    alps::mcbase::define_parameters(parameters);
    alps::define_convenience_parameters(parameters)
        .description("LCT-AUX Hubbard Model")
        .define<int>("sweeps", 1000, "maximum number of sweeps")
        .define<int>("thermalization", "number of sweeps for thermalization")
        .define<double>("beta", "inverse temperature of the system")
        .define<double>("U", 0., "local Hubbard interaction")
        .define<std::string>("H", "path to file containing the edge matrix")
        .define<std::string>("V", "number of vertices")
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
        // << alps::accumulators::FullBinningAccumulator<double>("Density")
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
        measurements["Sign"] << current_sign;
        // measurements["Density"] << sign*cache;
        measurements["Kinetic Energy"] << current_sign*kinetic_energy(cache)/volume();
        measurements["Interaction Energy"] << current_sign*interaction_energy(cache)/volume();
        measurements["Vertices"] << vertices();
    }
}

double lctaux_sim::fraction_completed() const {
    return (sweeps_ < thermalization_sweeps_ ? 0. : ( sweeps_ - thermalization_sweeps_ ) / double(total_sweeps_));
}

// void lctaux_sim::save(alps::hdf5::archive & ar) const {
//     mcbase::save(ar);
//     ar["checkpoint/sweeps"] << sweeps;
//     ar["checkpoint/spins"] << spins;
// }
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
