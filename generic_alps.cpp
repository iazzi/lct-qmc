#include "lctaux_sim.hpp"

#include <alps/mc/api.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/stop_callback.hpp>
#include "alps/utilities/remove_extensions.hpp"

#include <boost/chrono.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>

#include <string>
#include <iostream>
#include <stdexcept>

typedef lctaux_sim mysim;

int main(int argc, const char *argv[]) {

     try {
        typedef alps::parameters_type<mysim>::type params_type;
        params_type parameters(argc, argv, "/parameters"); // reads from HDF5 if need be
        
        // if parameters are restored from the archive, all definitions are already there
        if (!parameters.is_restored()) {
            mysim::define_parameters(parameters);
        }
        if (parameters.help_requested(std::cerr)) return 1; // Stop if help requested.

        if (parameters["outputfile"].as<std::string>().empty()) {
            parameters["outputfile"] = alps::remove_extensions(parameters.get_origin_name()) + ".out.h5";
        }

        mysim sim(parameters);
        
        // If needed, restore the last checkpoint
        std::string checkpoint_file = parameters["checkpoint"];
        if (parameters.is_restored()) {
            std::cout << "Restoring checkpoint from " << checkpoint_file << std::endl;
            sim.load(checkpoint_file);
        }
        
        // Run the simulation
        sim.run(alps::stop_callback(int(parameters["timelimit"])));
        
        std::cout << "Checkpointing simulation..." << std::endl;
        sim.save(checkpoint_file);

        using alps::collect_results;
        alps::results_type<mysim>::type results = collect_results(sim);

        std::cout << results << std::endl;
        double p2 = sim.exact_probability();
        std::cout << "dp = " << sim.exact_probability()-sim.probability() << ' ' << sim.probability() << std::endl << std::endl;
    
        alps::hdf5::archive ar(parameters["outputfile"], "w");
        ar["/parameters"] << parameters;
        ar["/simulation/results"] << results;

    } catch (std::exception const & e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
