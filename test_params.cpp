#include "simulation.hpp"
#include "logger.hpp"

#include <thread>
#include <chrono>
#include <atomic>

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
	lua_len(L, -1);
	size_t njobs = lua_tonumber(L, -1);
	lua_pop(L, 1);
	for (int j=0;j<nthreads;j++) {
		threads[j] = std::thread( [=, &log, &lock, &current, &failed] () {
				steady_clock::time_point t0 = steady_clock::now();
				steady_clock::time_point t1 = steady_clock::now();
				while (true) {
					int job = current.fetch_add(1);
					double fps, mps;
					lock.lock();
					lua_rawgeti(L, -1, job);
					if (lua_isnil(L, -1)) {
						log << "thread" << j << "terminating";
						lua_pop(L, 1);
						lock.unlock();
						break;
					}
					log << "thread" << j << "running simulation" << job << '/' << njobs;
					Simulation simulation(L, -1);
					lua_pop(L, 1);
					lock.unlock();
					try {
						t0 = steady_clock::now();
						while (duration_cast<seconds_type>(steady_clock::now()-t0).count()<10) {
							simulation.update();
						}
						t1 = steady_clock::now();
						fps = (double(simulation.steps)/duration_cast<seconds_type>(t1-t0).count());
						log << fps << "flips per second";
						int measures = 0;
						t0 = steady_clock::now();
						while (duration_cast<seconds_type>(steady_clock::now()-t0).count()<10) {
							simulation.measure();
							measures++;
						}
						t1 = steady_clock::now();
						mps = (double(measures)/duration_cast<seconds_type>(t1-t0).count());
						log << mps << "measures per second";
					} catch (...) {
						failed++;
						log << "thread" << j << "caught exception in simulation" << job << " with params " << simulation.params();
					}
					log << "thread" << j << "saving simulation" << job;
					lock.lock();
					lua_rawgeti(L, -1, job);
					lua_pushnumber(L, fps);
					lua_setfield(L, -2, "fps");
					lua_pushnumber(L, mps);
					lua_setfield(L, -2, "mps");
					lua_pop(L, 1);

					//lua_getglobal(L, "serialize");
					//lua_pushstring(L, "test_out.lua");
					//lua_pushvalue(L, -3);
					//lua_pcall(L, 2, 0, 0);

					lock.unlock();
				}
		});
	}
	for (std::thread& t : threads) t.join();
	lua_getglobal(L, "serialize");
	lua_insert(L, -2);
	lua_getfield(L, -1, "outfn");
	lua_insert(L, -2);
	lua_pcall(L, 2, 0, 0);

	std::cout << failed << " tasks failed" << std::endl;

	lua_close(L);
	fftw_cleanup_threads();
	return 0;
}


