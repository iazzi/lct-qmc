
#include "lctaux_sim.hpp"

#include <alps/mc/api.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/stop_callback.hpp>
#include "alps/utilities/remove_extensions.hpp"

#include <boost/chrono.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>

#include <string>
#include <iostream>
#include <stdexcept>

typedef lctaux_sim mysim;

int main(int argc, char *argv[]) {

    boost::mpi::environment env(argc, argv,false);
    try {
        boost::mpi::communicator comm;

        alps::parameters_type<mysim>::type parameters;
        if (comm.rank() == 0) {
            // on master:
            alps::parameters_type<mysim>::type p(argc, (const char**)argv, "/parameters"); // reads from HDF5 if supplied
            // if parameters are restored from the archive, all definitions are already there
            if (!p.is_restored()) {
                alps::mcmpiadapter<mysim>::define_parameters(p);
            }
            parameters=p;
        }
        broadcast(comm, parameters, 0); // all slaves get parameters from the master
        
        if (parameters.help_requested(std::cerr)) return EXIT_FAILURE; // Stop if help requested
        std::string checkpoint_file=alps::remove_extensions(parameters["checkpoint"])
            + ".clone" + boost::lexical_cast<std::string>(comm.rank()) + ".h5";

        alps::mcmpiadapter<mysim> sim(parameters, comm, alps::check_schedule(parameters["Tmin"],
                                                                                 parameters["Tmax"]));

        if (parameters.is_restored())
            sim.load(checkpoint_file);

        // TODO: how do we handle signels in mpi context? do we want to handle these in the callback or in the simulation?
        // do not use stop_callback_mpi: we do not want an bcast after every sweep!
        //  Additionally this causes a race cond and deadlocks as mcmpiadapter::run will always call the stop_callback broadcast
        //  but only sometimes all_reduce on the fraction. Timers on different procs are not synchronized so they may not agree
        //  on the mpi call.
        sim.run(alps::stop_callback(comm, parameters["timelimit"]));

        sim.save(checkpoint_file);

        using alps::collect_results;
        alps::results_type<mysim>::type results = collect_results(sim);

        if (comm.rank() == 0) {
            std::cout << results << std::endl;
            alps::hdf5::archive ar(parameters["outputfile"], "w");
            ar["/parameters"] << parameters;
            ar["/simulation/results"] << results;
        }

    } catch (std::exception const & e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
