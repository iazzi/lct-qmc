#include "simulation.hpp"
#include "logger.hpp"

#include <set>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace std::chrono;

typedef std::chrono::duration<double> seconds_type;

string time_str (int t) {
	stringstream str;
	str.fill('0');
	str << std::setw(2) <<(t/3600) << ':' << std::setw(2) << (t/60%60) << ':' << std::setw(2) << (t%60);
	return str.str();
}

int main (int argc, char **argv) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	if (luaL_dofile(L, argv[1])) {
		std::cerr << "Error loading configuration file \"" << argv[1] << "\":" << std::endl;
		std::cerr << '\t' << lua_tostring(L, -1) << std::endl;
		return -1;
	}
	std::string job_name("test");
	if (argc>2) job_name = argv[2];

	int nthreads = 1;
	char *e = getenv("LSB_HOSTS");
	while (e!=NULL && *e!='\0') if (*(e++)==' ') nthreads++;
	e = getenv("SLURM_NPROCS");
	if (e) {
		string f(e);
		if (!f.empty()) {
			stringstream g(f);
			g >> nthreads;
		}
	}

	lua_getfield(L, -1, "THREADS");
	if (lua_tointeger(L, -1)) {
		nthreads = lua_tointeger(L, -1);
	}
	lua_pop(L, 1);
	Logger log;

	std::atomic<int> current;
	current = 1;
	double eta = 0.0;
	lua_len(L, -1);
	int njobs = lua_tonumber(L, -1);
	lua_pop(L, 1);
	std::vector<double> times;
	for (int job=1;job<=njobs;job++) {
		steady_clock::time_point t0 = steady_clock::now();
		steady_clock::time_point t1 = steady_clock::now();
			double ups, mps;
			lua_rawgeti(L, -1, job);
			if (lua_isnil(L, -1)) {
				log << "job" << job << "empty: terminating";
				break;
			}
			log << "running simulation" << job << '/' << njobs;
			Simulation simulation(L, -1);
			lua_getfield(L, -1, "THERMALIZATION"); int thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
			lua_getfield(L, -1, "SWEEPS"); int total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
			//lua_getfield(L, -1, "TIMES"); int repetitions = std::max(int(lua_tointeger(L, -1)), 1); lua_pop(L, 1);
			lua_pop(L, 1);
			int updates = 0;
			t0 = steady_clock::now();
			while (duration_cast<seconds_type>(steady_clock::now()-t0).count()<5) {
				simulation.update();
				updates++;
			}
			t1 = steady_clock::now();
			ups = (double(updates)/duration_cast<seconds_type>(t1-t0).count());
			log << ups << "updates per second";
			int measures = 0;
			t0 = steady_clock::now();
			while (duration_cast<seconds_type>(steady_clock::now()-t0).count()<5) {
				simulation.measure_quick();
				measures++;
			}
			t1 = steady_clock::now();
			mps = (double(measures)/duration_cast<seconds_type>(t1-t0).count());
			log << mps << "measures per second";
			double jeta = (thermalization_sweeps+total_sweeps)/ups + total_sweeps/mps;
			log << "estimated time:" << jeta;
			eta += jeta;
			//for (int k=0;k<repetitions;k++)
				times.push_back(jeta);
	}
	lua_close(L);
	struct Batch {
		double eta;
		set<int> jobs;
		bool operator< (const Batch& other) const {
			return eta > other.eta;
		}
		Batch (double t) : eta(t) {}
		Batch (double t, int j) : eta(t) { jobs.insert(j); }
		Batch (const Batch& other, double t, int j) : eta(other.eta), jobs(other.jobs) { eta += t; jobs.insert(j); }
	};
	set<Batch> batches;
	njobs = times.size();
	log << njobs << "jobs";
	for (int j=0;j<njobs;j++) {
		for (set<Batch>::iterator b = batches.begin();b!=batches.end();++b) {
			if (b->eta+times[j]<18*3600) {
				batches.insert(Batch(*b, times[j], j));
				batches.erase(b);
				times[j] = 0.0;
				break;
			}
		}
		if (times[j]>0.0) {
			set<Batch>::iterator b = batches.insert(Batch(times[j], j)).first;
			times[j] = 0.0;
		}
	}
	for (auto b : batches) {
		cout << "eta " << b.eta << "; jobs [";
		for (auto j: b.jobs) cout << j << ", ";
		cout << "]\n";
	}
	log << "total estimated time:" << time_str(eta);
	string outname(argv[1]);
	outname.append(".batch");
	ofstream out(outname);
	out <<
		"#!/usr/local/bin/bash -l\n"
		"#\n"
		"#SBATCH --job-name=\"" << job_name << "\"\n"
		"#SBATCH --partition=dphys_compute\n"
		//"#SBATCH --cpus-per-task=20\n"
		//"#SBATCH --nodes=1\n"
		//"#SBATCH --ntasks-per-node=1\n"
		"#SBATCH --output=outtest.%j.out\n"
		"#SBATCH --error=errtest.%j.err\n";
	//out <<  "#SBATCH --time=24:00:00\n";
	out <<  "#SBATCH --time=" << time_str(1.5*eta) << "\n";
	out << "srun ./main " << argv[1] << "\n";
	return 0;
}


