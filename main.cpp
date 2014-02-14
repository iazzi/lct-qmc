#include "simulation.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

#include "helpers.hpp"
#include "measurements.hpp"
#include "weighted_measurements.hpp"
#include "logger.hpp"
#include "svd.hpp"

extern "C" {
#include <fftw3.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <csignal>

//#define fftw_execute (void)

using namespace std;
using namespace std::chrono;

typedef std::chrono::duration<double> seconds_type;

std::string measurement_ratio (const measurement<double, false>& x, const measurement<double, false>& y, const char *s) {
	double a, b;
	a = x.mean()/y.mean();
	b = fabs(a)*(fabs(x.error()/x.mean())+fabs(y.error()/y.mean()));
	std::ostringstream buf;
	buf << a << s << b;
	return buf.str();
}

sig_atomic_t signaled = 0;
void signal_handler (int signum) {
	    cout << "Interrupt signal (" << signum << ") received.\n";
	    if (signum==SIGINT && signaled==0) {
		    signaled = 1;
	    } else {
		    exit(signum);
	    }
}


void run_thread (int j, lua_State *L, Logger &log, std::mutex &lock, std::atomic<int> &current, std::atomic<int> &failed) {
	signal(SIGINT, signal_handler);
	steady_clock::time_point t0 = steady_clock::now();
	steady_clock::time_point t1 = steady_clock::now();
	steady_clock::time_point t2 = steady_clock::now();
	log << "thread" << j << "starting";
	while (true) {
		steady_clock::time_point t_start = steady_clock::now();
		int job = current.fetch_add(1);
		lock.lock();
		lua_rawgeti(L, -1, job);
		if (lua_isnil(L, -1)) {
			log << "thread" << j << "terminating";
			lua_pop(L, 1);
			lock.unlock();
			break;
		}
		log << "thread" << j << "running simulation" << job;
		lua_getfield(L, -1, "THERMALIZATION"); int thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
		lua_getfield(L, -1, "SWEEPS"); int total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
		lua_getfield(L, -1, "savefile"); std::string savefile = lua_isstring(L, -1)?lua_tostring(L, -1):std::string(); lua_pop(L, 1);
		Simulation simulation(L, -1);
		lua_pop(L, 1);
		if (!savefile.empty()) {
			if (luaL_dofile(L, savefile.c_str())) {
				log << "error loading savefile:" << lua_tostring(L, -1);
				lua_pop(L, 1);
			} else {
				lua_getfield(L, -1, "THERMALIZATION"); thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
				lua_getfield(L, -1, "SWEEPS"); total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
				simulation.load_checkpoint(L);
				lua_pop(L, 1);
			}
			if (thermalization_sweeps>0) simulation.discard_measurements();
		}
		//simulation.load_sigma(L, "nice.lua");
		lock.unlock();
		auto save_checkpoint = [&] (int thermalization, int sweeps) {
			lock.lock();
			simulation.save_checkpoint(L);
			lua_pushinteger(L, thermalization);
			lua_setfield(L, -2, "THERMALIZATION");
			lua_pushinteger(L, sweeps);
			lua_setfield(L, -2, "SWEEPS");
			lua_pushstring(L, getenv("LSB_JOBID"));
			lua_setfield(L, -2, "JOBID");
			lua_getglobal(L, "serialize");
			lua_insert(L, -2);
			lua_pushstring(L, savefile.c_str());
			lua_insert(L, -2);
			lua_pcall(L, 2, 0, 0);
			lock.unlock();
		};
		auto save_density = [&] (const char *n) {
			int N = simulation.timeSlices();
			int V = simulation.volume();
			ofstream dens(n);
			for (int i=0;i<V;i++) {
				dens << i << ' ' << i << ' ';
				dens << measurement_ratio(simulation.d_up[i], simulation.measured_sign, " ") << ' ';
				dens << measurement_ratio(simulation.d_dn[i], simulation.measured_sign, " ") << ' ';
				dens << measurement_ratio(simulation.spincorrelation[i], simulation.measured_sign, " ") << ' ';
				dens << endl;
			}
		};
		save_checkpoint(thermalization_sweeps, total_sweeps);
		try {
			t0 = steady_clock::now();
			t1 = steady_clock::now();
			for (int i=0;i<thermalization_sweeps;i++) {
				if (duration_cast<seconds_type>(steady_clock::now()-t2).count()>5 && !savefile.empty() && signaled>0) {
					signaled = 0;
					log << "saving checkpoint";
					t2 = steady_clock::now();
					save_checkpoint(thermalization_sweeps-i, total_sweeps);
				}
				if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
					t1 = steady_clock::now();
					int N = simulation.timeSlices();
					int V = simulation.volume();
					log << "thread" << j << "thermalizing: " << i << '/' << thermalization_sweeps << "..." << (double(simulation.steps)/duration_cast<seconds_type>(t1-t0).count()) << "steps per second (" << N*V << "sites sweep in" << (duration_cast<seconds_type>(t1-t0).count()*N*V/simulation.steps) << "seconds)";
					log << simulation.measured_sign;
					log << "Density: " << measurement_ratio(simulation.density, simulation.measured_sign, " +- ");
					log << "Magnetization: " << measurement_ratio(simulation.magnetization, simulation.measured_sign, " +- ") << '\n';
					save_density("density.dat");
				}
				simulation.update();
				simulation.measure_quick();
			}
			log << "thread" << j << "thermalized";
			simulation.steps = 0;
			simulation.discard_measurements();
			t0 = steady_clock::now();
			for (int i=0;i<total_sweeps;i++) {
				if (duration_cast<seconds_type>(steady_clock::now()-t2).count()>600 && !savefile.empty()) {
					t2 = steady_clock::now();
					save_checkpoint(0, total_sweeps-i);
				}
				if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
					t1 = steady_clock::now();
					log << "thread" << j << "running: " << i << '/' << total_sweeps << "..." << (double(simulation.steps)/duration_cast<seconds_type>(t1-t0).count()) << "steps per second";
					save_density("density.dat");
				}
				simulation.update();
				simulation.measure_quick();
				//simulation.measure_sign();
			}
			double seconds = duration_cast<seconds_type>(steady_clock::now()-t_start).count();
			log << "thread" << j << "finished simulation" << job << "in" << seconds << "seconds";
			lock.lock();
			simulation.output_results();
			lua_rawgeti(L, -1, job);
			lua_pushnumber(L, seconds);
			lua_setfield(L, -2, "elapsed_time");
			simulation.save(L, lua_gettop(L));
			lua_getglobal(L, "serialize");
			lua_insert(L, -2);
			lua_getfield(L, -1, "outfile");
			lua_insert(L, -2);
			lua_pcall(L, 2, 0, 0);
			save_density("density.dat");
			lock.unlock();
		} catch (...) {
			failed++;
			log << "thread" << j << "caught exception in simulation" << job << " with params " << simulation.params();
		}
	}
}

int main (int argc, char **argv) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	if (luaL_dofile(L, argv[1])) {
		std::cerr << "Error loading configuration file \"" << argv[1] << "\":" << std::endl;
		std::cerr << '\t' << lua_tostring(L, -1) << std::endl;
		return -1;
	}

	fftw_init_threads();
	fftw_plan_with_nthreads(1);

	int nthreads = 1;
	char *e = getenv("LSB_HOSTS");
	while (e!=NULL && *e!='\0') if (*(e++)==' ') nthreads++;

	lua_getfield(L, -1, "THREADS");
	if (lua_tointeger(L, -1)) {
		nthreads = lua_tointeger(L, -1);
	}
	lua_pop(L, 1);
	Logger log(cout);
	//log.setVerbosity(5);
	log << "using" << nthreads << "threads";

	std::vector<std::thread> threads(nthreads);
	std::mutex lock;
	std::atomic<int> failed;
	failed = 0;
	std::atomic<int> current;
	current = 1;
	for (int j=0;j<nthreads;j++) {
		threads[j] = std::thread(run_thread, j, L, std::ref(log), std::ref(lock), std::ref(current), std::ref(failed));
	}
	for (std::thread& t : threads) t.join();
	log << "joined threads";
	lua_getglobal(L, "serialize");
	lua_insert(L, -2);
	lua_pushstring(L, "stablefast_out.lua");
	lua_insert(L, -2);
	lua_pcall(L, 2, 0, 0);

	std::cout << failed << " tasks failed" << std::endl;

	lua_close(L);
	fftw_cleanup_threads();
	return 0;
}


