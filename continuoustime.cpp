#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <functional>

#include <alps/alea.h>
#include <alps/alea/mcanalyze.hpp>

#include <alps/ngs.hpp>
#include <alps/ngs/scheduler/proto/mcbase.hpp>
#include <alps/ngs/make_parameters_from_xml.hpp>

#include "ct_aux.hpp"


extern "C" {
#include <fftw3.h>
}

#include <Eigen/Dense>
#include <Eigen/QR>

extern "C" void dggev_ (const char *jobvl, const char *jobvr, const int &N,
			double *A, const int &lda, double *B, const int &ldb,
			double *alphar, double *alphai, double *beta,
			double *vl, const int &ldvl, double *vr, const int &ldvr,
			double *work, const int &lwork, int &info);

void dggev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta) {
	Eigen::MatrixXd a = A, b = B;
	int info = 0;
	int N = a.rows();
	Eigen::VectorXd alphar = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd alphai = Eigen::VectorXd::Zero(N);
	alpha = Eigen::VectorXcd::Zero(N);
	beta = Eigen::VectorXd::Zero(N);
	//Eigen::VectorXd vl = Eigen::VectorXd::Zero(1);
	//Eigen::VectorXd vr = Eigen::VectorXd::Zero(1);
	Eigen::VectorXd work = Eigen::VectorXd::Zero(8*N);
	dggev_("N", "N", N, a.data(), N, b.data(), N,
			alphar.data(), alphai.data(), beta.data(),
			NULL, 1, NULL, 1,
			work.data(), work.size(), info);
	if (info == 0) {
		alpha.real() = alphar;
		alpha.imag() = alphai;
	} else if (info<0) {
		std::cerr << "dggev_: error at argument " << -info << std::endl;
	} else if (info<=N) {
		std::cerr << "QZ iteration failed at step " << info << std::endl;
	} else {
	}
}

using namespace std;
using namespace Eigen;
using namespace alps;

typedef ctaux_sim sim_type;

int main (int argc, char **argv) {
	mcoptions options(argc, argv);
	parameters_type<sim_type>::type params = make_parameters_from_xml(options.input_file);
	sim_type sim(params);


	int L = 1;
	double beta = 10.0;
	double U = 0.1;
	double mu = -0.5;
	double B = 0.0;
	double J = 0.0;
	int qrn = 0;

	int n = 0;
	int a = 0;
	for (int i=0;i>int(params["THERMALIZATION"]);i++) {
		//if (i%100==0) { std::cout << i << "\r"; std::cout.flush(); }
		//sim.update();
		//sim.measure();
	}
	//std::cout << int(params["THERMALIZATION"]) << std::endl;
	//std::cout.flush();

	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	for (int k=0;k>int(params["SWEEPS"]);k++) {
		//sim.update();
		//sim.measure();
		//n++;
		//d_up << sim.numberUp();
		//d_dn << sim.numberDown();
		//slices << double(sim.sliceNumber());
	}

	sim.setDiscreteTime(100);
	try {
		//sim.run(boost::bind(&stop_callback, options.time_limit));
	}
	catch (const char *s) {
		std::cerr << s << std::endl;
	}
	results_type<sim_type>::type results = collect_results(sim);
	std::cout << results << std::endl;
	save_results(results, params, options.output_file, "/simulation/results");

	return 0;
}

