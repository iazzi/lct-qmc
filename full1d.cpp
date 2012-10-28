#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>

#include <alps/alea.h>
#include <alps/alea/mcanalyze.hpp>


extern "C" {
#include <fftw3.h>
}

#include <Eigen/Dense>
#include <Eigen/QR>

static const double pi = 3.141592653589793238462643383279502884197;

//namespace alps {
  //template<> struct type_tag< Eigen::ArrayXd > : public boost::mpl::int_<37> {};
//};

class Configuration {
	private:
	int L; // size of the system
	int V; // volume of the system
	int N; // number of time-steps
	double beta; // inverse temperature
	double dt; // time step 
	double g; // interaction strength
	double mu; // chemical potential
	double A; // sqrt(exp(g*dt)-1)
	double B; // magnetic field
	double D; // 2*cosh(beta*B)

	std::vector<Eigen::VectorXd> diagonals;

	std::default_random_engine generator;
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::exponential_distribution<double> trialDistribution;

	Eigen::VectorXd energies;
	Eigen::VectorXd freePropagator;

	Eigen::MatrixXd positionSpace; // current matrix in position space
	Eigen::MatrixXcd momentumSpace;

	fftw_plan x2p;
	fftw_plan p2x;

	Eigen::FullPivHouseholderQR<Eigen::MatrixXd> decomposer;

	//Eigen::MatrixXcd n_s; // single particle density matrix

	double plog;

	double energy;
	double number;
	std::valarray<double> density;

	public:

	double n_up;
	double n_dn;

	int qrnumber;

	public:

	Configuration (int l, int n, double Beta, double interaction, double m, double b, double t_) : L(l), V(l), N(n), beta(Beta), dt(Beta/n),
		       			g(interaction), mu(m), B(b), qrnumber(0), distribution(0.5), randomPosition(0, l-1),
					randomTime(0, n-1), trialDistribution(1.0) {
		A = sqrt(exp(g*dt)-1.0);
		D = 2.0 * cosh(beta*B);
		auto distributor = std::bind(distribution, generator);
		diagonals.insert(diagonals.begin(), N, Eigen::VectorXd::Zero(V));
		for (int i=0;i<diagonals.size();i++) {
			for (int j=0;j<V;j++) {
				diagonals[i][j] = distributor()?A:-A;
			}
		}
		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		const int size[] = { L, };
		x2p = fftw_plan_many_dft_r2c(1, size, V, positionSpace.data(),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_PATIENT);
		p2x = fftw_plan_many_dft_c2r(1, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, positionSpace.data(), NULL, 1, V, FFTW_PATIENT);

		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		energies = Eigen::VectorXd::Zero(V);
		freePropagator = Eigen::VectorXd::Zero(V);
		for (int i=0;i<V;i++) {
			energies[i] = - cos(2.0*i*pi/L) - t_ * cos(4.0*i*pi/L) - mu;
			freePropagator[i] = exp(-dt*energies[i]);
		}

		plog = logProbability();
	}

	double logProbability () {
		Eigen::HouseholderQR<Eigen::MatrixXd> qrsolver;
		Eigen::MatrixXd R = Eigen::MatrixXd::Identity(V, V);
		int qrperiod = qrnumber>0?N/qrnumber:0;

		positionSpace.setIdentity(V, V);
		R.setIdentity(V, V);
		for (int i=0;i<N;i++) {
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+diagonals[i]).asDiagonal());
			fftw_execute(x2p);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x);
			positionSpace /= V;
			if ((qrperiod>0 && (i+1)%qrperiod==0) || i==N-1) {
				qrsolver.compute(positionSpace);
				R.applyOnTheLeft(qrsolver.householderQ().inverse()*positionSpace);
				positionSpace = qrsolver.householderQ();
			}
		}
		positionSpace.applyOnTheRight(R);

		//Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
		//Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;

		std::complex<double> ret = 0.0;
		ret += (1.0 + std::exp(+beta*B)*positionSpace.eigenvalues().array()).log().sum();
		ret += (1.0 + std::exp(-beta*B)*positionSpace.eigenvalues().array()).log().sum();

		if (std::cos(ret.imag())<0.99) {
			if (qrnumber==N) {
				std::cout << positionSpace << std::endl << std::endl;
				std::cout << positionSpace.eigenvalues().transpose() << std::endl;
				//std::cout << S1.eigenvalues().transpose() << std::endl;
				//std::cout << S2.eigenvalues().transpose() << std::endl;
				throw("wtf");
			} else {
				qrnumber++;
				std::cout << "imaginary part = " << ret.imag() << " increasing qrnumber -> " << qrnumber << std::endl;
				return logProbability();
			}
			return logProbability();
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

	bool metropolis (int M) {
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
		if (-trialDistribution(generator)<trial-plog) {
			//std::cout << "accepted " << trial-plog << std::endl;
			plog = trial;
			//density = std::valarray<double>(n_s.diagonal().real().data(), V);
			n_up = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B) * positionSpace).inverse() ).trace();
			n_dn = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(-beta*B) * positionSpace).inverse() ).trace();
			//number = n_s.real().trace();
			ret = true;
		} else {
			//std::cout << "rejected " << trial-plog << std::endl;
			for (int i=0;i<M;i++) {
				int t = index[i]/V;
				int x = index[i]%V;
				diagonals[t][x] = -diagonals[t][x];
			}
			ret = false;
		}
		//std::cout << n_up << ' ' << n_dn << std::endl;
		//measuredNumber << number;
		//std::cout << measuredNumber << std::endl;
		return ret;
	}

	void setQRNumber (int n) {
		qrnumber = n;
	}

	~Configuration () { fftw_destroy_plan(x2p); fftw_destroy_plan(p2x); }
	protected:
};

int main (int argc, char **argv) {
	int L = 4;
	int N = 1000;
	int M = 500;
	double beta = 10.0;
	double g = 0.1;
	double mu = -0.5;
	double B = 0.0;
	int qrn = 0;
	double t_ = -1.0;

	alps::RealObservable density("density");
	alps::RealObservable magnetization("magnetization");

	for (int i=1;i<argc;i++) {
		if (argv[i][0]=='-') {
			switch (argv[i][1]) {
				case 'L':
					L = atoi(argv[++i]);
					break;
				case 'N':
					N = atoi(argv[++i]);
					break;
				case 'M':
					M = atoi(argv[++i]);
					break;
				case 'T':
					beta = 1.0/atof(argv[++i]);
					break;
				case 'g':
					g = atof(argv[++i]);
					break;
				case 'm':
					mu = atof(argv[++i]);
					break;
				case 'B':
					B = atof(argv[++i]);
					break;
				case 'q':
					qrn = atoi(argv[++i]);
					break;
				case 't':
					t_ = atof(argv[++i]);
					break;
			}
		}
	}
	Configuration configuration(L, N, beta, g, mu, B, t_);
	configuration.setQRNumber(qrn);

	int n = 0;
	int a = 0;
	for (int i=0;i<100000;i++) {
		if (i%100==0) { std::cout << i << "\r"; std::cout.flush(); }
		configuration.metropolis(M);
	}

	density.set_bin_size(128);
	magnetization.set_bin_size(128);
	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	for (;;) {
		if (configuration.metropolis(M)) a++;
		n++;
		density << configuration.n_up + configuration.n_dn;
		magnetization << configuration.n_up - configuration.n_dn;
		if (n%1024==0) {
			time_end = std::chrono::steady_clock::now();
			std::cout << "size = " << L << ", time steps = " << N << ", decompositions = " << configuration.qrnumber << std::endl;
			std::cout << "temperature = " << (1.0/beta) << ", interaction = " << g << std::endl;
			std::cout << "chemical potential = " << mu << ", magnetic field = " << B << std::endl;
			std::cout << "acceptance = " << (double(a)/double(n)) << " spin flips = " << M << std::endl;
			std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() << " seconds" << std::endl;
			std::cout << "steps per second = " << n/std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() << std::endl;
			std::cout << density << std::endl;
			std::cout << magnetization << std::endl;
			if (a>0.6*n) {
				M += 5;
				time_start = std::chrono::steady_clock::now();
				//configuration.measuredNumber.reset(true);
				//configuration.eigenvalues.reset(true);
				density.reset(true);
				magnetization.reset(true);
				density.set_bin_size(128);
				magnetization.set_bin_size(128);
				n = 0;
				a = 0;
			} else if (a<0.4*n) {
				density.reset(true);
				magnetization.reset(true);
				density.set_bin_size(128);
				magnetization.set_bin_size(128);
				M -= 5;
			}

		}
		//configuration.print();
	}
	return 0;
}

