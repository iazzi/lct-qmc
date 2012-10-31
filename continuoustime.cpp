#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
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

//namespace alps {
  //template<> struct type_tag< Eigen::ArrayXd > : public boost::mpl::int_<37> {};
//};

class Configuration {
	private:
	int L; // size of the system
	int D; // dimension
	int V; // volume of the system
	double beta; // inverse temperature
	double g; // interaction strength
	double mu; // chemical potential
	double A; // sqrt(g)
	double B; // magnetic field
	double J; // next-nearest neighbour hopping

	std::map<double, Eigen::VectorXd> diagonals;

	std::default_random_engine generator;
	std::bernoulli_distribution distribution;
	std::uniform_real_distribution<double> randomDouble;
	std::uniform_real_distribution<double> randomTime;
	std::exponential_distribution<double> trialDistribution;

	Eigen::VectorXd energies;

	Eigen::MatrixXd positionSpace; // current matrix in position space
	Eigen::MatrixXcd momentumSpace;

	fftw_plan x2p;
	fftw_plan p2x;

	double plog;

	double energy;
	double number;
	std::valarray<double> density;

	public:

	double n_up;
	double n_dn;

	public:

	Configuration (int d, int l, double Beta, double interaction, double m, double b, double j)
		: L(l), D(d), V(std::pow(l, D)), beta(Beta),
		g(interaction), mu(m), B(b), J(j), distribution(0.5), randomDouble(1.0),
		randomTime(0, Beta), trialDistribution(1.0) {
		A = sqrt(g);
		auto distributor = std::bind(distribution, generator);
		//diagonals.insert(diagonals.begin(), N, Eigen::VectorXd::Zero(V));
		//for (int i=0;i<diagonals.size();i++) {
			//for (int j=0;j<V;j++) {
				//diagonals[i][j] = distributor()?A:-A;
			//}
		//}
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
		for (int i=0;i<V;i++) {
			energies[i] = - cos(2.0*(i%L)*pi/L) - cos(2.0*((i/L)%L)*pi/L) - cos(2.0*(i/L/L)*pi/L) + (3-D) - mu;
			energies[i] += J * (- cos(4.0*(i%L)*pi/L) - cos(4.0*((i/L)%L)*pi/L) - cos(4.0*(i/L/L)*pi/L) + (3-D) );
		}

		plog = logProbability();
	}

	double logProbability () {
		const double F = sqrt(2.0)/2.0;
		double t = 0.0;
		double dt;
		positionSpace.setIdentity(V, V);
		for (auto i : diagonals) {
			fftw_execute(x2p);
			dt = i.first-t;
			momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
			fftw_execute(p2x);
			positionSpace /= V;
			// FIXME needs to take into account 1/2 from the sum??
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+i.second).asDiagonal());
			t = i.first;
		}
		fftw_execute(x2p);
		dt = beta-t;
		momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
		fftw_execute(p2x);
		positionSpace /= V;

		//Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
		//Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;

		Eigen::ArrayXcd ev = positionSpace.eigenvalues();

		std::complex<double> ret = 0.0;
		ret += ( 1.0 + ev*std::exp(-beta*B) ).log().sum() + ( 1.0 + ev*std::exp(+beta*B) ).log().sum();

		if (std::cos(ret.imag())<0.99) {
			Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
			Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;
			std::cerr << positionSpace << std::endl << std::endl;
			std::cerr << S1 << std::endl << std::endl;
			std::cerr << S1.eigenvalues().transpose() << std::endl;
			std::cerr << S2.eigenvalues().transpose() << std::endl;
			std::cerr << S1.eigenvalues().array().log().sum() << std::endl;
			std::cerr << S2.eigenvalues().array().log().sum() << std::endl;
			std::cerr << ev.transpose() << std::endl;
			std::cerr << ret << std::endl;
			std::cerr << diagonals.size() << std::endl;
			throw("wtf");
		}
		return ret.real();
	}

	bool metropolisFlip (int M) {
		if (diagonals.size()==0) return false;
		bool ret = false;
		if (M>V) M = V;
		std::vector<int> index(M);
		int t = std::uniform_int_distribution<int>(0, diagonals.size()-1)(generator);
		std::map<double, Eigen::VectorXd>::iterator diter = diagonals.begin();
		while (t--) diter++;
		Eigen::VectorXd &d = (*diter).second;
		for (int j=0;j<M;j++) {
			std::uniform_int_distribution<int> distr(0, V-j-1);
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
			int x = index[i]%V;
			d[x] = -d[x];
		}
		double trial = logProbability();
		if (-trialDistribution(generator)<trial-plog) {
			plog = trial;
			n_up = ( Eigen::MatrixXd::Identity(V, V) - sqrt(2.0)*(sqrt(2)*Eigen::MatrixXd::Identity(V, V) + exp(+beta*B) * positionSpace).inverse() ).trace();
			n_dn = ( Eigen::MatrixXd::Identity(V, V) - sqrt(2.0)*(sqrt(2)*Eigen::MatrixXd::Identity(V, V) + exp(-beta*B) * positionSpace).inverse() ).trace();
			ret = true;
		} else {
			for (int i=0;i<M;i++) {
				int x = index[i]%V;
				d[x] = -d[x];
			}
			ret = false;
		}
		return ret;
	}

	bool metropolisUp () {
		bool ret = false;
		double t = randomTime(generator);
		auto distributor = std::bind(distribution, generator);
		std::map<double, Eigen::VectorXd>::iterator diter = diagonals.find(t);
		if (diter!=diagonals.end()) return false;
		diagonals[t] = Eigen::VectorXd::Zero(V);
		diter = diagonals.find(t);
		for (int i=0;i<V;i++) diter->second[i] = distributor()?A:-A;
		double trial = logProbability();
		if (randomDouble(generator)<std::exp(trial-plog)*beta/diagonals.size()) {
			//std::cerr << "accepted increase: time steps = " << diagonals.size() << std::endl;
			plog = trial;
			n_up = ( Eigen::MatrixXd::Identity(V, V) - sqrt(2.0)*(sqrt(2)*Eigen::MatrixXd::Identity(V, V) + exp(+beta*B) * positionSpace).inverse() ).trace();
			n_dn = ( Eigen::MatrixXd::Identity(V, V) - sqrt(2.0)*(sqrt(2)*Eigen::MatrixXd::Identity(V, V) + exp(-beta*B) * positionSpace).inverse() ).trace();
			ret = true;
		} else {
			diagonals.erase(diter);
			ret = false;
		}
		return ret;
	}

	bool metropolisDown () {
		bool ret = false;
		std::pair<double, Eigen::VectorXd> store;
		int t = std::uniform_int_distribution<int>(0, diagonals.size()-1)(generator);
		std::map<double, Eigen::VectorXd>::iterator diter = diagonals.begin();
		while (t--) diter++;
		store = *diter;
		diagonals.erase(diter);
		double trial = logProbability();
		if (randomDouble(generator)<std::exp(trial-plog)*(diagonals.size()+1)/beta) {
			//std::cerr << "accepted decrease: time steps = " << diagonals.size() << std::endl;
			plog = trial;
			n_up = ( Eigen::MatrixXd::Identity(V, V) - sqrt(2.0)*(sqrt(2)*Eigen::MatrixXd::Identity(V, V) + exp(+beta*B) * positionSpace).inverse() ).trace();
			n_dn = ( Eigen::MatrixXd::Identity(V, V) - sqrt(2.0)*(sqrt(2)*Eigen::MatrixXd::Identity(V, V) + exp(-beta*B) * positionSpace).inverse() ).trace();
			ret = true;
		} else {
			diagonals[store.first] = store.second;
			ret = false;
		}
		return ret;
	}

	bool metropolis (int M) {
		std::discrete_distribution<int> distribution { 0.90, 0.05, 0.05 };
		int type = distribution(generator);
		if (type == 0 || (type == 1 && diagonals.size()==0 )) {
			return metropolisFlip(M);
		} else if (type == 1) {
			//std::cerr << "proposed decrease" << std::endl;
			return metropolisDown();
		} else {
			//std::cerr << "proposed increase" << std::endl;
			return metropolisUp();
		}
	}

	int sliceNumber () { return diagonals.size(); }

	void print () {
		for (auto i : diagonals) {
			std::cout << i.first << '\t';
			for (int j=0;j<V;j++) {
				std::cout << (i.second[j]<0?'-':'+');
			}
			std::cout << std::endl;
		}
	}

	~Configuration () { fftw_destroy_plan(x2p); fftw_destroy_plan(p2x); }
	protected:
};

int main (int argc, char **argv) {
	int D = 1;
	int L = 4;
	int M = 1;
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
	Configuration configuration(D, L, beta, g, mu, B, J);

	int n = 0;
	int a = 0;
	for (int i=0;i<10000;i++) {
		if (i%100==0) { std::cout << i << "\r"; std::cout.flush(); }
		configuration.metropolis(M);
	}

	std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
	for (;;) {
		if (configuration.metropolis(M)) a++;
		n++;
		density << configuration.n_up + configuration.n_dn;
		magnetization << configuration.n_up - configuration.n_dn;
		if (n%(1<<10)==0) {
			time_end = std::chrono::steady_clock::now();
			std::cout << "dimension = " << D << ", size = " << L << std::endl;
			std::cout << "temperature = " << (1.0/beta) << ", interaction = " << g << std::endl;
			std::cout << "chemical potential = " << mu << ", magnetic field = " << B << std::endl;
			std::cout << "acceptance = " << (double(a)/double(n)) << " spin flips = " << M << std::endl;
			std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() << " seconds" << std::endl;
			std::cout << "steps per second = " << n/std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() << std::endl;
			std::cout << "slices = " << configuration.sliceNumber() << std::endl;
			std::cout << density << std::endl;
			std::cout << magnetization << std::endl;
			//configuration.print();
			if (a>0.6*n) {
				//configuration.measuredNumber.reset(true);
				//configuration.eigenvalues.reset(true);
			} else if (a<0.4*n) {
			}

		}
		//configuration.print();
	}
	return 0;
}

