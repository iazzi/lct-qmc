#pragma once

#include "type_save.hpp"
#include "lctsimulation.hpp"

#include <alps/mc/mcbase.hpp>
#include <alps/hdf5/archive.hpp>
// #include <alps/hdf5/vector.hpp>

#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>

#include <vector>
#include <string>
#include <random>

class lctaux_sim : public LCTSimulation, public alps::mcbase {
public:
    
        lctaux_sim(parameters_type const & parms, std::size_t seed_offset = 13);

        static void define_parameters(parameters_type & parameters);

        virtual void update();
        virtual void measure();
        virtual double fraction_completed() const;

         using alps::mcbase::save;
         virtual void save(alps::hdf5::archive & ar) const;

	 using alps::mcbase::load;
	 virtual void load(alps::hdf5::archive & ar);

	 void print (std::ostream &out) const;

    private:
        
        double p1_, pr_, ps_;
        
        int sweeps_, iteration_;
        int thermalization_sweeps_;
        int total_sweeps_;
};
