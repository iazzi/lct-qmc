#pragma once

#include "configuration.hpp"
#include "genericlattice.hpp"
#include "slice.hpp"
#include "model.hpp"
#include "hubbard.hpp"
#include "spin_one_half.hpp"

#include <alps/mc/mcbase.hpp>
#include <alps/hdf5/archive.hpp>
// #include <alps/hdf5/vector.hpp>

#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>

#include <vector>
#include <string>
#include <random>

class lctaux_sim : public alps::mcbase {
    typedef Model<SpinOneHalf<GenericLattice>, HubbardInteraction> model_type;
    typedef Configuration<model_type> configuration_type;
    typedef SpinOneHalf<GenericLattice> lattice_type;
    
    
    public:
    
        lctaux_sim(parameters_type const & parms, std::size_t seed_offset = 13);

        static void define_parameters(parameters_type & parameters);

        virtual void update();
        virtual void measure();
        virtual double fraction_completed() const;

        // using alps::mcbase::save;
        // virtual void save(alps::hdf5::archive & ar) const;
        //
        // using alps::mcbase::load;
        // virtual void load(alps::hdf5::archive & ar);

    private:
        
        void full_check();
        void single_step(bool check=false);
        
        
        Parameters lctaux_parameters_;
        lattice_type lattice_;
        HubbardInteraction interaction_;
        model_type model_;
        configuration_type conf_;
        
        double p1_, pr_, ps_;
        
        int sweeps_;
        enum {left_to_right, right_to_left} sweep_direction_;
        int thermalization_sweeps_;
        int total_sweeps_;
        double beta_;
        
        std::mt19937_64 generator_;
        std::uniform_real_distribution<double> uniform_d_;
        std::exponential_distribution<double> trial_d_;
};
