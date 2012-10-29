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

extern "C" void dggev_ (const char *jobvl, const char *jobvr, const int &N,
			double *A, const int &lda, double *B, const int &ldb,
			double *alphar, double *alphai, double *beta,
			double *vl, const int &ldvl, double *vr, const int &ldvr,
			double *work, const int &lwork, int &info);

void ddgev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta) {
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

//namespace alps {
  //template<> struct type_tag< Eigen::ArrayXd > : public boost::mpl::int_<37> {};
//};

class Configuration {
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
	double J; // next-nearest neighbour hopping

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

	Configuration (int d, int l, int n, double Beta, double interaction, double m, double b, double j)
		: L(l), D(d), V(std::pow(l, D)), N(n), beta(Beta), dt(Beta/n),
		g(interaction), mu(m), B(b), J(j), qrnumber(0), distribution(0.5), randomPosition(0, l-1),
		randomTime(0, n-1), trialDistribution(1.0) {
		A = sqrt(exp(g*dt)-1.0);
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
		x2p = fftw_plan_many_dft_r2c(D, size, V, positionSpace.data(),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_PATIENT);
		p2x = fftw_plan_many_dft_c2r(D, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, positionSpace.data(), NULL, 1, V, FFTW_PATIENT);

		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		energies = Eigen::VectorXd::Zero(V);
		freePropagator = Eigen::VectorXd::Zero(V);
		for (int i=0;i<V;i++) {
			energies[i] = - cos(2.0*(i%L)*pi/L) - cos(2.0*((i/L)%L)*pi/L) - cos(2.0*(i/L/L)*pi/L) + (3-D) - mu;
			energies[i] += J * (- cos(4.0*(i%L)*pi/L) - cos(4.0*((i/L)%L)*pi/L) - cos(4.0*(i/L/L)*pi/L) + (3-D) );
			freePropagator[i] = exp(-dt*energies[i]);
		}

		plog = logProbability();
	}

	void checkConsistency () {
		Eigen::HouseholderQR<Eigen::MatrixXd> qrsolver;
		Eigen::MatrixXd R = Eigen::MatrixXd::Identity(V, V);
		int qrperiod = qrnumber>0?N/qrnumber:0;

		positionSpace.setIdentity(V, V);
		R.setIdentity(V, V);
		for (int i=0;i<N;i++) {
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+diagonals[i]).asDiagonal());
			Eigen::ArrayXcd ev = (positionSpace*R).eigenvalues();
			if (std::cos(ev.log().sum().imag())<0.999) {
				std::cout << "imaginary part in the propagator at time slice " << i << std::endl;
				std::cout << positionSpace*R << std::endl << std::endl;
				std::cout << ev.transpose() << std::endl;
				std::cout << (Eigen::VectorXd::Constant(V, 1.0)+diagonals[i]).transpose() << std::endl << std::endl;
			} else {
				std::cout << "everything fine up to " << i << std::endl;
				std::cout << positionSpace*R << std::endl << std::endl;
				std::cout << ev.transpose() << std::endl << std::endl;
			}
			fftw_execute(x2p);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x);
			positionSpace /= V;
			ev = (positionSpace*R).eigenvalues();
			if (std::cos(ev.log().sum().imag())<0.999) {
				std::cout << "imaginary part in the propagator at time slice " << i << std::endl;
				std::cout << positionSpace*R << std::endl << std::endl;
				std::cout << ev.transpose() << std::endl;
				std::cout << (Eigen::VectorXd::Constant(V, 1.0)+diagonals[i]).transpose() << std::endl << std::endl;
			} else {
				std::cout << "everything fine up to " << i << std::endl;
				std::cout << positionSpace*R << std::endl << std::endl;
				std::cout << ev.transpose() << std::endl << std::endl;
			}
			if ((qrperiod>0 && (i+1)%qrperiod==0) || i==N-1) {
				qrsolver.compute(positionSpace);
				R.applyOnTheLeft(qrsolver.householderQ().inverse()*positionSpace);
				Eigen::VectorXd sign = Eigen::VectorXd::Ones(V);
				for (int k=0;k<V;k++) if (R(k, k)<0.0) sign[k] = -1.0;
				R.applyOnTheLeft(sign.asDiagonal());
				positionSpace = qrsolver.householderQ();
				positionSpace.applyOnTheRight(sign.asDiagonal());
			}
		}
		Eigen::VectorXcd eva;
		Eigen::VectorXd evb;
		ddgev(R, positionSpace.transpose(), eva, evb);

		//positionSpace.applyOnTheRight(R);
		if (std::cos(eva.array().log().sum().imag() - evb.array().log().sum())<0.999) {
			std::cout << "imaginary part in the propagator " << std::endl;
			std::cout << positionSpace << std::endl << std::endl;
			std::cout << positionSpace.eigenvalues().transpose() << std::endl << std::endl;
		} else {
			std::cout << "generalized ev problem was ok" << std::endl;
		}
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
		Eigen::VectorXcd eva;
		Eigen::VectorXd evb;
		ddgev(R, positionSpace.transpose(), eva, evb);

		positionSpace.applyOnTheRight(R);

		//Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
		//Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;

		std::complex<double> ret = 0.0;
		//ret += (1.0 + std::exp(+beta*B)*positionSpace.eigenvalues().array()).log().sum();
		//ret += (1.0 + std::exp(-beta*B)*positionSpace.eigenvalues().array()).log().sum();
		ret += (evb.cast<std::complex<double>>() + std::exp(+beta*B)*eva).array().log().sum();
		ret -= evb.array().log().sum();
		ret += (evb.cast<std::complex<double>>() + std::exp(-beta*B)*eva).array().log().sum();
		ret -= evb.array().log().sum();

		Eigen::ArrayXcd temp = eva.array()/evb.array().cast<std::complex<double>>();
		//std::cerr << temp.transpose() << std::endl;
		//std::cerr << positionSpace.eigenvalues().transpose() << std::endl << std::endl;

		if (std::cos(ret.imag())<0.99) {
			if (qrnumber==N) {
				checkConsistency();
				throw("wtf");
			} else {
				qrnumber++;
				std::cerr << "imaginary part = " << ret.imag() << " increasing qrnumber -> " << qrnumber << std::endl;
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
	int D = 1;
	int L = 4;
	int N = 1000;
	int M = 500;
	double beta = 10.0;
	double g = 0.1;
	double mu = -0.5;
	double B = 0.0;
	double J = 0.0;
	int qrn = 0;
	alps::RealObservable density("density");
	alps::RealObservable magnetization("magnetization");

	for (int i=1;i<argc;i++) {
		if (argv[i][0]=='-') {
			switch (argv[i][1]) {
				case 'D':
					D = atoi(argv[++i]);
					break;
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
				case 'J':
					J = atoi(argv[++i]);
					break;
			}
		}
	}
	Configuration configuration(D, L, N, beta, g, mu, B, J);
	configuration.setQRNumber(qrn);

	int n = 0;
	int a = 0;
	for (int i=0;i<10000;i++) {
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
			std::cout << "dimension = " << D << ", size = " << L << std::endl;
			std::cout << "time steps = " << N << ", decompositions = " << configuration.qrnumber << std::endl;
			std::cout << "temperature = " << (1.0/beta) << ", interaction = " << g << std::endl;
			std::cout << "chemical potential = " << mu << ", magnetic field = " << B << std::endl;
			std::cout << "acceptance = " << (double(a)/double(n)) << " spin flips = " << M << std::endl;
			std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() << " seconds" << std::endl;
			std::cout << "steps per second = " << n/std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() << std::endl;
			std::cout << density << std::endl;
			std::cout << magnetization << std::endl;
			if (a>0.6*n) {
				M += 1;
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
				M -= 1;
				M = M>0?M:1;
			}

		}
		//configuration.print();
	}
	return 0;
}

