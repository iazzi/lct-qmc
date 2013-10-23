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

//#define fftw_execute (void)

using namespace std;
using namespace std::chrono;

typedef std::chrono::duration<double> seconds_type;

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
	Logger log;
	//log.setVerbosity(5);
	log << "using" << nthreads << "threads";

	std::vector<std::thread> threads(nthreads);
	std::mutex lock;
	std::atomic<int> failed;
	failed = 0;
	std::atomic<int> current;
	current = 1;
	for (int j=0;j<nthreads;j++) {
		threads[j] = std::thread( [=, &log, &lock, &current, &failed] () {
				steady_clock::time_point t0 = steady_clock::now();
				steady_clock::time_point t1 = steady_clock::now();
				log << "thread" << j << "starting";
				while (true) {
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
					Simulation simulation(L, -1);
					lua_pop(L, 1);
					//simulation.load_sigma(L, "nice.lua");
					lock.unlock();
					//for (int i=0;i<0;i++) simulation.update_ising();
					//log << "annealed";
					//simulation.straighten_slices();
					//simulation.set_time_shift(0);
					//do {
						//simulation.anneal_ising();
					//} while(!simulation.shift_time());
					//simulation.recheck();
					//simulation.svd.diagonalize();
					try {
						t0 = steady_clock::now();
						for (int i=0;i<thermalization_sweeps;i++) {
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
								t1 = steady_clock::now();
								log << "thread" << j << "thermalizing: " << i << '/' << thermalization_sweeps << "..." << (double(simulation.steps)/duration_cast<seconds_type>(t1-t0).count()) << "steps per second";
								//log << simulation.sign;
								//log << simulation.acceptance;
							}
							simulation.update();
							if (simulation.psign<0.0) {
								//log << "negative sign found... saving";
								//simulation.recheck();
								//throw "";
							}
						}
						log << "thread" << j << "thermalized";
						simulation.measured_sign.clear();
						simulation.steps = 0;
						t0 = steady_clock::now();
						for (int i=0;i<total_sweeps;i++) {
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
								t1 = steady_clock::now();
								log << "thread" << j << "running: " << i << '/' << total_sweeps << "..." << (double(simulation.steps)/duration_cast<seconds_type>(t1-t0).count()) << "steps per second";
							}
							simulation.update();
							//simulation.measure();
							simulation.measure_sign();
							//log << simulation.measured_sign;
							if (simulation.psign<0.0) {
								//simulation.svd.diagonalize();
								//throw -1;
							}
						}
						log << "thread" << j << "finished simulation" << job;
						lock.lock();
						//simulation.output_results();
						simulation.output_sign();
						lua_rawgeti(L, -1, job);
						simulation.save(L, lua_gettop(L));
						lua_getglobal(L, "serialize");
						lua_insert(L, -2);
						lua_getfield(L, -1, "outfile");
						lua_insert(L, -2);
						lua_pcall(L, 2, 0, 0);
						cout << lua_tostring(L, -1) << endl;
						lock.unlock();
					} catch (...) {
						failed++;
						log << "thread" << j << "caught exception in simulation" << job << " with params " << simulation.params();
					}
				}
		});
	}
	for (std::thread& t : threads) t.join();
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


