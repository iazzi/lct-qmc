#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

#include <alps/alea.h>
#include <alps/alea/mcanalyze.hpp>
#include <alps/ngs.hpp>
#include <alps/ngs/scheduler/proto/mcbase.hpp>
#include <alps/ngs/make_parameters_from_xml.hpp>

#include "helpers.hpp"


extern "C" {
#include <fftw3.h>
}

#include <Eigen/Dense>
#include <Eigen/QR>

static const double pi = 3.141592653589793238462643383279502884197;

void dggev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta);

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

class Configuration : public alps::mcbase_ng {
	private:
	int L; // size of the system
	int D; // dimension
	int V; // volume of the system
	int N; // number of time-steps
	double beta; // inverse temperature
	double dt; // time step 
	double g; // interaction strength
	double mu; // chemical potential
	double A; // sqrt(exp(g*dt)-1)
	double B; // magnetic field
	double t; // nearest neighbour hopping
	double J; // next-nearest neighbour hopping

	std::vector<Eigen::VectorXd> diagonals;

	std::default_random_engine generator;
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::exponential_distribution<double> trialDistribution;

	Eigen::VectorXd energies;
	Eigen::VectorXd freePropagator;
	Eigen::VectorXd freePropagator_b;

	Eigen::MatrixXd positionSpace; // current matrix in position space
	Eigen::MatrixXcd momentumSpace;

	fftw_plan x2p_col;
	fftw_plan p2x_col;

	fftw_plan x2p_row;
	fftw_plan p2x_row;

	Eigen::FullPivHouseholderQR<Eigen::MatrixXd> decomposer;

	double plog;

	public:

	double n_up;
	double n_dn;

	int qrnumber;

	public:

	void init () {
		A = sqrt(exp(g*dt)-1.0);
		if (L==1) t = 0.0;
		auto distributor = std::bind(distribution, generator);
		diagonals.insert(diagonals.begin(), N, Eigen::VectorXd::Zero(V));
		for (int i=0;i<diagonals.size();i++) {
			for (int j=0;j<V;j++) {
				diagonals[i][j] = distributor()?A:-A;
			}
		}
		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		const int size[] = { L, L, L, };
		x2p_col = fftw_plan_many_dft_r2c(D, size, V, positionSpace.data(),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_PATIENT);
		p2x_col = fftw_plan_many_dft_c2r(D, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, positionSpace.data(), NULL, 1, V, FFTW_PATIENT);
		x2p_row = fftw_plan_many_dft_r2c(D, size, V, positionSpace.data(),
				NULL, V, 1, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, V, 1, FFTW_PATIENT);
		p2x_row = fftw_plan_many_dft_c2r(D, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, V, 1, positionSpace.data(), NULL, V, 1, FFTW_PATIENT);

		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		energies = Eigen::VectorXd::Zero(V);
		freePropagator = Eigen::VectorXd::Zero(V);
		freePropagator_b = Eigen::VectorXd::Zero(V);
		for (int i=0;i<V;i++) {
			energies[i] += -2.0 * t * ( cos(2.0*(i%L)*pi/L) - cos(2.0*((i/L)%L)*pi/L) - cos(2.0*(i/L/L)*pi/L) + (3.0-D) );
			energies[i] += -2.0 * J * ( cos(4.0*(i%L)*pi/L) - cos(4.0*((i/L)%L)*pi/L) - cos(4.0*(i/L/L)*pi/L) + (3.0-D) );
			freePropagator[i] = exp(-dt*energies[i]);
			freePropagator_b[i] = exp(dt*energies[i]);
		}

		measurements << alps::ngs::RealObservable("N")
			<< alps::ngs::RealObservable("M")
			<< alps::ngs::RealObservable("acceptance");

		plog = logProbability();
	}

	Configuration (const parameters_type& params) : mcbase_ng(params), distribution(0.5), trialDistribution(1.0) {
		L = params["L"];
		beta = 1.0/double(params["T"]);
		t = double(params["t"]);
		g = -double(params["U"]);
		mu = params["mu"];
		B = params["B"];
		qrnumber = 0;

		if (params["LATTICE"].cast<std::string>()==std::string("chain lattice")) {
			D = 1;
		} else if (params["LATTICE"].cast<std::string>()==std::string("square lattice")) {
			D = 2;
		} else if (params["LATTICE"].cast<std::string>()==std::string("simple cubic lattice")) {
			D = 3;
		} else {
			throw std::string("unknown lattice type");
		}

		V = std::pow(L, D);
		N = int(beta/double(params["dTau"]));
		dt = beta/N;
		init();
	}

	Configuration (int d, int l, int n, double Beta, double interaction, double m, double b, double t_ = 1.0, double j = 0.0)
		: L(l), D(d), V(std::pow(l, D)), N(n), beta(Beta), dt(Beta/n),
		g(interaction), mu(m), B(b), t(t_), J(j), qrnumber(0), distribution(0.5), randomPosition(0, l-1),
		randomTime(0, n-1), trialDistribution(1.0),
		mcbase_ng(parameters_type())	{
		init();
	}

	void accumulate_forward (int start = 0, int end = -1) {
		double X = sqrt(1.0 - A*A);
		positionSpace.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+diagonals[i]).asDiagonal());
			fftw_execute(x2p_col);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x_col);
			positionSpace /= V*X;
		}
	}

	void accumulate_backward (int start = 0, int end = -1) {
		double X = sqrt(1.0 - A*A);
		positionSpace.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			positionSpace.applyOnTheRight((Eigen::VectorXd::Constant(V, 1.0)-diagonals[i]).asDiagonal());
			fftw_execute(x2p_row);
			momentumSpace.applyOnTheRight(freePropagator_b.asDiagonal());
			fftw_execute(p2x_row);
			positionSpace /= V*X;
		}
	}

	double logProbability_simple () {
		double X = 1.0 - A*A;
		double Y = pow(X, N);
		accumulate_forward(0, N);
		Eigen::MatrixXd U_s = positionSpace;
		accumulate_backward(0, N);
		std::cout << std::endl;
		std::cout << N << std::endl;
		std::cout << X << std::endl;
		std::cout << Y << std::endl;
		std::cout << U_s*positionSpace << std::endl << std::endl;
		std::cout << positionSpace*U_s << std::endl << std::endl;
		std::cout << U_s.eigenvalues().transpose() << std::endl;
		std::cout << positionSpace.eigenvalues().transpose() << std::endl;
		std::cout << positionSpace.eigenvalues().array().inverse().transpose() << std::endl;
		std::cout << std::endl;
	}

	double logProbability_complex () {
		const int M = 30;
		std::vector<Eigen::MatrixXd> fvec;
		std::vector<Eigen::MatrixXd> bvec;
		for (int i=0;i<N;i+=M) {
			accumulate_forward(i, i+M);
			fvec.push_back(positionSpace);
		}
		for (int i=0;i<N;i+=M) {
			accumulate_backward(i, i+M);
			bvec.push_back(positionSpace);
		}
		test_sequences(fvec, bvec);
	}

	double logProbability () {
		Eigen::HouseholderQR<Eigen::MatrixXd> qrsolver;
		Eigen::MatrixXd R = Eigen::MatrixXd::Identity(V, V);
		int qrperiod = qrnumber>0?N/qrnumber:0;

		positionSpace.setIdentity(V, V);
		R.setIdentity(V, V);
		for (int i=0;i<N;i++) {
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+diagonals[i]).asDiagonal());
			fftw_execute(x2p_col);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x_col);
			positionSpace /= V;
			if ((qrperiod>0 && (i+1)%qrperiod==0) || i==N-1) {
				qrsolver.compute(positionSpace);
				R.applyOnTheLeft(qrsolver.householderQ().inverse()*positionSpace);
				positionSpace = qrsolver.householderQ();
			}
		}
		Eigen::VectorXcd eva;
		Eigen::VectorXd evb;
		dggev(R, positionSpace.transpose(), eva, evb);

		positionSpace.applyOnTheRight(R);

		//Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
		//Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;

		std::complex<double> ret = 0.0;
		//ret += (1.0 + std::exp(+beta*B)*positionSpace.eigenvalues().array()).log().sum();
		//ret += (1.0 + std::exp(-beta*B)*positionSpace.eigenvalues().array()).log().sum();
		ret += (evb.cast<std::complex<double>>() + std::exp(+beta*B*0.5+beta*mu)*eva).array().log().sum();
		ret -= evb.array().log().sum();
		ret += (evb.cast<std::complex<double>>() + std::exp(-beta*B*0.5+beta*mu)*eva).array().log().sum();
		ret -= evb.array().log().sum();

		Eigen::ArrayXcd temp = eva.array()/evb.array().cast<std::complex<double>>();
		//std::cerr << temp.transpose() << std::endl;
		//std::cerr << positionSpace.eigenvalues().transpose() << std::endl << std::endl;

		if (std::cos(ret.imag())<0.99) {
			logProbability_complex();
			throw("wtf");
		}
		return ret.real();
	}

	void print () {
		for (int i=0;i<N;i++) {
			for (int j=0;j<V;j++) {
				std::cout << (diagonals[i][j]<0?'-':'+');
			}
			std::cout << std::endl;
		}
	}

	bool metropolis (int M = 0) {
		if (M==0) M = 0.1 * volume();
		bool ret = false;
		std::vector<int> index(M);
		for (int j=0;j<M;j++) {
			std::uniform_int_distribution<int> distr(0, V*N-j-1);
			int x = distr(generator);
			int i = j;
			// standard insertion sort would be:
			//
			// while (i>0 && x<index[i-1]) { index[i] = index[i-1]; i--;}
			//
			// if we are going to insert at place i we have to shift the value by i
			// if this number is less *or equal* than the one below in the list,
			// we cannot insert (we want unique indices) so we shift i down and move
			// up by one the top index (in the standard insertion sort equality is
			// irrelevant)
			while (i>0 && x+i<=index[i-1]) { index[i] = index[i-1]; i--; }
			index[i] = x+i;
		}
		for (int i=0;i<M;i++) {
			int t = index[i]/V;
			int x = index[i]%V;
			diagonals[t][x] = -diagonals[t][x];
		}
		double trial = logProbability();
		//logProbability_complex();
		//throw "end";
		if (-trialDistribution(generator)<trial-plog) {
			plog = trial;
			n_up = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B*0.5+beta*mu) * positionSpace).inverse() ).trace();
			n_dn = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(-beta*B*0.5+beta*mu) * positionSpace).inverse() ).trace();
			if (std::isnan(n_up) || std::isinf(n_up)) {
				std::cout << n_up << std::endl;
				std::cout << n_dn << std::endl;
				std::cout << positionSpace << std::endl << std::endl;
				std::cout << positionSpace.eigenvalues().transpose() << std::endl << std::endl;
				std::cout << (Eigen::MatrixXd::Identity(V, V) + exp(-beta*B*0.5) * positionSpace).inverse() << std::endl << std::endl;
				throw(9);
			}
			ret = true;
		} else {
			for (int i=0;i<M;i++) {
				int t = index[i]/V;
				int x = index[i]%V;
				diagonals[t][x] = -diagonals[t][x];
			}
			ret = false;
		}
		return ret;
	}

	void setQRNumber (int n) {
		qrnumber = n;
	}

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		metropolis();
	}

	void measure () {
		measurements["N"] << (n_up + n_dn) / V;
		measurements["M"] << (n_up - n_dn) / 2.0 / V;
	}

	int volume () { return V; }

	~Configuration () {
		fftw_destroy_plan(x2p_col);
		fftw_destroy_plan(p2x_col);
		fftw_destroy_plan(x2p_row);
		fftw_destroy_plan(p2x_row);
	}
	protected:
};

using namespace std;
using namespace alps;

typedef mcbase_ng sim_type;

int main (int argc, char **argv) {
	mcoptions options(argc, argv);
	parameters_type<sim_type>::type params = make_parameters_from_xml(options.input_file);

	int thermalization_sweeps = int(params["THERMALIZATION"]);
	int total_sweeps = int(params["SWEEPS"]);

	Configuration configuration(params);
	//Configuration configuration(D, L, N, beta, g, mu, B, t, 0.0);
	configuration.setQRNumber(0);

	int n = 0;
	int a = 0;
	int M = 1;
	for (int i=0;i<thermalization_sweeps;i++) {
		if (i%100==0) { std::cout << i << "\r"; std::cout.flush(); }
		if (configuration.metropolis(M)) a++;
		n++;
		if (i%200==0) {
			if (a>0.6*n && M<0.1*configuration.volume()) {
				cout << "M: " << M;
				M += 5;
				cout << " -> " << M << endl;
				n = 0;
				a = 0;
				i = 0;
			} else if (a<0.4*n) {
				cout << "M: " << M;
				M -= 5;
				cout << " -> " << M << endl;
				M = M>0?M:1;
				n = 0;
				a = 0;
				i = 0;
			}
		}
	}
	std::cout << thermalization_sweeps << "\n"; std::cout.flush();

	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	for (int i=0;i<total_sweeps;i++) {
		if (i%100==0) { std::cout << i << "\r"; std::cout.flush(); }
		if (configuration.metropolis(M)) a++;
		n++;
		configuration.measure();
	}
	std::cout << total_sweeps << "\n"; std::cout.flush();
	results_type<sim_type>::type results = collect_results(configuration);
	std::cout << results << std::endl;
	save_results(results, params, options.output_file, "/simulation/results");
	return 0;
}

