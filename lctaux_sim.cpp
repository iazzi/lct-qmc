
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
        std::cout << "..." << it->first;
        if (parameters.exists(it->first)) {
            p.setString(it->first, boost::lexical_cast<std::string>(it->second));
            std::cout << "   " << "set!" << std::endl;
        } else {
            std::cout << "   SKIP."<< std::endl;
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
    : alps::mcbase(parms, seed_offset)
    , lctaux_parameters_(convert_parameters(parms))
    , lattice_(lctaux_parameters_)
    , interaction_(lctaux_parameters_)
    , model_(make_model(lattice_, interaction_))
    , conf_(model_)
    , sweeps_(0)
    , sweep_direction_(left_to_right)
    , thermalization_sweeps_(int(parameters["thermalization"]))
    , total_sweeps_(int(parameters["sweeps"]))
    , beta_(double(parameters["beta"]))
    , generator_(std::size_t(parameters["SEED"]) + seed_offset)
{   
    lctaux_parameters_.list();
    
    conf_.setup(lctaux_parameters_);
    for (size_t i=0;i<conf_.slice_number();i++) {
        conf_.set_index(i);
        for (size_t j=0;j<2*lattice_.volume();j++) {
            conf_.insert(interaction_.generate(0.0, conf_.slice_end()-conf_.slice_start(), generator_));
        }
    }
    conf_.set_index(0);
    conf_.compute_right_side();
    conf_.start();
    conf_.start();
    cerr << conf_.check_B_vs_last_right_side() << endl;
    //ofstream diff ("diff.dat", ios::app);
    //diff << "\"V=" << model.interaction().dimension() << " beta=" << conf_.inverse_temperature() << "\"" << endl;
    conf_.compute_B();
    conf_.compute_G();
    conf_.save_G();
    pr_ = 0.0;
    std::tie(p1_, ps_) = conf_.probability();
    
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Sign")
        << alps::accumulators::FullBinningAccumulator<double>("Density")
        << alps::accumulators::FullBinningAccumulator<double>("Kinetic Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Interaction Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Vertices")
    ;
}

void lctaux_sim::full_check() {
    Eigen::MatrixXd G;
    for (size_t i=0;i<conf_.slice_number();i++) {
        conf_.set_index(i);
        conf_.compute_propagators_2();
        G = conf_.green_function();
        //std::cerr << G << std::endl << std::endl;
        conf_.compute_B();
        conf_.compute_G();
        conf_.save_G();
        cerr << (double(i)/conf_.slice_number()) << ' '
            << conf_.inverse_temperature() << ' '
            << model_.interaction().dimension() << ' '
            << (conf_.green_function()-G).norm() << ' '
            << (conf_.green_function()-G).cwiseAbs().maxCoeff() << endl;
    }
}

void lctaux_sim::single_step(bool check) {
    HubbardInteraction::Vertex v;
    for (size_t j=0;j<model_.lattice().volume();j++) {
        double dp = 0.0, s = 1.0;
        if (uniform_d_(generator_)<0.5) {
            v = conf_.get_vertex(uniform_d_(generator_)*conf_.slice_size());
            dp = conf_.remove_probability(v);
            s = dp>0.0?1.0:-1.0;
            dp = std::log(std::fabs(dp));
            if (-trial_d_(generator_)<dp+conf_.remove_factor()) {
                //cerr << "removed vertex " << v.tau << endl;
                conf_.remove_and_update(v);
                pr_ += dp;
                ps_ *= s;
            } else {
                //cerr << "remove rejected" << endl;
            }
        } else {
            v = model_.interaction().generate(0.0, conf_.slice_end()-conf_.slice_start(), generator_);
            dp = conf_.insert_probability(v);
            s = dp>0.0?1.0:-1.0;
            dp = std::log(std::fabs(dp));
            if (-trial_d_(generator_)<dp+conf_.insert_factor()) {
                //cerr << "inserted vertex " << v.tau << endl;
                conf_.insert_and_update(v);
                pr_ += dp;
                ps_ *= s;
            } else {
                //cerr << "insert rejected" << endl;
            }
        }
        if (check) {
            conf_.compute_B();
            double p2 = conf_.probability().first;
            std::cerr << "v = " << v.x << ',' << v.tau << " dp = " << p1_+pr_-p2 << ' ' << p2-p1_ << ' ' << pr_ << endl << endl;
        }
    }
    conf_.compute_right_side();
}

void lctaux_sim::update() {
    bool check = false;
    
    Eigen::MatrixXd G;
    for (size_t p=0;p<conf_.slice_number();p++) {
        if (sweep_direction_ == left_to_right) {
            size_t i = p;
            
            conf_.set_index(i);
            conf_.compute_right_side();
            conf_.compute_propagators_2();
            
            // TODO: measure at every sweep or every step?
            // measurements.measure(model, conf_, ps);
            
            single_step(check);
            
        } else {
            size_t i = conf_.slice_number() - p;
            
            conf_.set_index(i-1);
            conf_.compute_left_side();
            conf_.compute_propagators_2();
            
            // TODO: measure at every sweep or every step?
            // measurements.measure(model, conf_, ps);
            
            single_step(check);
            
        }
    }
    if (check) {
        conf_.start();
        conf_.check_all_prop();
        //conf_.check_all_det(0);
        //conf_.check_all_det(1);
    }
}

void lctaux_sim::measure() {
    if (sweep_direction_ == right_to_left) {
        sweeps_++;
        sweep_direction_ = left_to_right;
    } else {
        sweep_direction_ = right_to_left;
    }
    
    if (sweeps_ > thermalization_sweeps_) {
        MatrixXd cache = conf_.green_function();
        const double sign = ps_;
        // pull in operator/ for vectors
        using alps::numeric::operator/;
        measurements["Sign"] << sign;
        // measurements["Density"] << sign*cache;
        measurements["Kinetic Energy"] << sign*model_.lattice().kinetic_energy(cache)/model_.lattice().volume();
        measurements["Interaction Energy"] << sign*model_.interaction().interaction_energy(cache)/model_.lattice().volume();
        measurements["Vertices"] << conf_.size();
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
