#include "ct_simulation.hpp"

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

#include <set>

#include <csignal>

//#define fftw_execute (void)

using SVDMatrix = SVDHelper;

struct Vertex {
	double tau;
	size_t x;
	double sigma;
	class Compare {
		public:
			constexpr bool operator() (const Vertex& a, const Vertex& b) {
				return (a.tau<b.tau) || (a.tau==b.tau && a.x<b.x);
			}
	};
	Vertex (double a, size_t b, double c) : tau(a), x(b), sigma(c) {}
};

class V3Simulation {
	std::set<Vertex, Vertex::Compare> verts;

	Matrix_d eigenvectors;
	Vector_d eigenvalues;

	size_t V;
	double beta, mu;

	std::vector<Matrix_d> slices_up;
	std::vector<Matrix_d> slices_dn;

	public:
	void setBeta (double b) {
		beta = b;
	}

	template <typename M>
		void setEigenvectors (const Eigen::MatrixBase<M> &U) {
			eigenvectors = U;
		}

	template <typename M>
		void setEigenvalues (const Eigen::MatrixBase<M> &E) {
			eigenvalues = E;
		}

	void make_slice (Matrix_d &G,double a, double b, double s) {
		auto first = verts.lower_bound(Vertex(a, 0, 0));
		auto last = verts.lower_bound(Vertex(b, 0, 0));
		double t = a;
		for (auto v=first;v!=last;v++) {
			if (v->tau>t) {
				G.array() *= (-(v->tau-t)*eigenvalues.array()).exp();
				t = v->tau;
			}
			G += s * v->sigma * eigenvectors.row(v->x).transpose() * eigenvectors.row(v->x) * G;
		}
		if (b>t) {
			G.array() *= (-(b-t)*eigenvalues.array()).exp();
		}
	}

	void make_slices (size_t n) {
		slices_up.resize(n);
		slices_dn.resize(n);
		std::fill(slices_up.begin(), slices_up.end(), Matrix_d::Identity(V, V));
		std::fill(slices_dn.begin(), slices_dn.end(), Matrix_d::Identity(V, V));
		for (size_t i=0;i<n;i++) {
			make_slice(slices_up[i], beta/n*i, beta/n*(i+1), +1.0);
			make_slice(slices_dn[i], beta/n*i, beta/n*(i+1), -1.0);
		}
	}

	std::pair<double, double> probability_from_scratch (size_t n) {
		make_slices(n);
		SVDMatrix svd_up, svd_dn;
		svd_up.setIdentity(V);
		svd_dn.setIdentity(V);
		for (size_t t=0;t<n;t++) {
			svd_up.U.applyOnTheLeft(slices_up[t]);
			svd_up.absorbU();
			svd_dn.U.applyOnTheLeft(slices_dn[t]);
			svd_dn.absorbU();
		}
		svd_up.add_identity(exp(beta*mu));
		svd_dn.add_identity(exp(beta*mu));
	}
};


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
		CTSimulation simulation(L, -1);
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
			simulation.discard_measurements();
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
					log << "Acceptance" << simulation.measurements.acceptance;
					log << "Last expansion order" << simulation.order();
					log << "Last probability" << simulation.svd_probability();
					log << simulation.measurements.order;
					log << simulation.measurements.sign_all_steps;
					//log << "Density:" << measurement_ratio(simulation.measurements.density, simulation.measurements.sign_all_steps, " +- ");
					//log << "Magnetization:" << measurement_ratio(simulation.measurements.magnetization, simulation.measurements.sign_all_steps, " +- ") << '\n';
					for (size_t i=0;i<simulation.histogram.size();i++) {
						log << i << simulation.histogram[i] << simulation.measurement_vector[i].sign_all_steps.samples();
					}
					ofstream dens("density.dat");
					for (int i=0;i<V;i++) {
						dens << i << ' ' << i << ' ';
						//dens << measurement_ratio(simulation.d_up[i], simulation.measurements.sign_all_steps, " ") << ' ';
						//dens << measurement_ratio(simulation.d_dn[i], simulation.measurements.sign_all_steps, " ") << ' ';
						//dens << measurement_ratio(simulation.spincorrelation[i], simulation.measurements.sign_all_steps, " ") << ' ';
						dens << endl;
					}
				}
				simulation.update();
				simulation.measure_quick();
				if (i%1000+1==1000) {
				       	//simulation.update_histogram();
					//simulation.discard_measurements();
				}
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
					for (size_t i=0;i<simulation.histogram.size();i++) {
						log << i << simulation.histogram[i] << simulation.measurement_vector[i].sign_all_steps.samples();
					}
				}
				simulation.update();
				simulation.measure_quick();
				//simulation.measure_sign();
			}
			double seconds = duration_cast<seconds_type>(steady_clock::now()-t_start).count();
			log << "thread" << j << "finished simulation" << job << "in" << seconds << "seconds";
			lock.lock();
			simulation.output_results();
			//simulation.output_sign();
			lua_rawgeti(L, -1, job);
			lua_pushnumber(L, seconds);
			lua_setfield(L, -2, "elapsed_time");
			simulation.save(L, lua_gettop(L));
			lua_getglobal(L, "serialize");
			lua_insert(L, -2);
			lua_getfield(L, -1, "outfile");
			lua_insert(L, -2);
			lua_pcall(L, 2, 0, 0);
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



