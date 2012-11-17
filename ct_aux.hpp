#ifndef CT_AUX_HPP
#define CT_AUX_HPP

#include <Eigen/Dense>
#include <Eigen/QR>

extern "C" {
#include <fftw3.h>
}

#include <alps/ngs.hpp>
#include <alps/ngs/scheduler/proto/mcbase.hpp>

static const double pi = 3.141592653589793238462643383279502884197;

class ctaux_sim : public alps::mcbase_ng {
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
	//std::mt19937_64 generator;
	std::bernoulli_distribution distribution;
	std::uniform_real_distribution<double> randomDouble;
	std::uniform_real_distribution<double> randomTime;
	std::exponential_distribution<double> trialDistribution;

	Eigen::VectorXd energies;

	Eigen::MatrixXd positionSpace; // current matrix in position space
	Eigen::MatrixXcd momentumSpace;


	//alps::RealObservable n_up_obs;
	//alps::RealObservable n_dn_obs;
	//alps::RealObservable slices_obs;

	fftw_plan x2p;
	fftw_plan p2x;

	double plog;

	int sweeps;
	int thermalization_sweeps;
	int total_sweeps;

	double n_up;
	double n_dn;

	public:

	void init () {
		measurements
			<< alps::ngs::RealObservable("n_up")
			<< alps::ngs::RealObservable("n_dn")
			<< alps::ngs::RealObservable("slices");

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
			energies[i] = - cos(2.0*(i%L)*pi/L) - cos(2.0*((i/L)%L)*pi/L) - cos(2.0*(i/L/L)*pi/L) + 3.0;
			energies[i] += J * (- cos(4.0*(i%L)*pi/L) - cos(4.0*((i/L)%L)*pi/L) - cos(4.0*(i/L/L)*pi/L) + 3.0 );
			energies[i] -= mu;
		}

		plog = logProbability();
	}

	ctaux_sim (parameters_type const &params, std::size_t seed_offset = 42) :
		mcbase_ng(params, seed_offset),
		sweeps(0),
		thermalization_sweeps(int(params["THERMALIZATION"])),
		total_sweeps(int(params["SWEEPS"])),
		L(params["L"]),
		D(params["D"]),
		V(std::pow(L, D)),
		beta(1.0/double(params["T"])),
		g(params["g"]),
		mu(params["mu"]),
		A(sqrt(g)),
		B(params["B"]),
		J(params["J"]),
		distribution(0.5),
		randomDouble(1.0),
		randomTime(0, beta),
		trialDistribution(1.0)
		//n_up_obs("n_up_obs"),
		//n_dn_obs("n_dn_obs"),
		//slices_obs("slices_obs")
	{
			init();
	}

	void computeNumber () {
		n_up = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B) * positionSpace).inverse() ).trace();
		n_dn = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(-beta*B) * positionSpace).inverse() ).trace();
	}

	double logProbability (int Q = 1) {
		Eigen::MatrixXd R;
		Eigen::HouseholderQR<Eigen::MatrixXd> decomposer;
		double t = 0.0;
		double dt;
		int decomposeNumber = Q;
		double decomposeStep = beta / (decomposeNumber+1);
		positionSpace.setIdentity(V, V);
		R.setIdentity(V, V);
		for (auto i : diagonals) {
			fftw_execute(x2p);
			dt = i.first-t;
			momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
			fftw_execute(p2x);
			positionSpace /= V;
			// FIXME needs to take into account 1/2 from the sum??
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+i.second).asDiagonal());
			t = i.first;
			if (t>beta-decomposeNumber*decomposeStep) {
				decomposer.compute(positionSpace);
				R.applyOnTheLeft(decomposer.householderQ().inverse()*positionSpace);
				positionSpace = decomposer.householderQ();
				decomposeNumber--;
			}
		}
		fftw_execute(x2p);
		dt = beta-t;
		momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
		fftw_execute(p2x);
		positionSpace /= V;

		//Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
		//Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;

		Eigen::VectorXcd eva;
		Eigen::VectorXd evb;
		//dggev(R, positionSpace.inverse(), eva, evb);

		positionSpace.applyOnTheRight(R);

		Eigen::ArrayXcd ev = positionSpace.eigenvalues();

		std::complex<double> ret = ( 1.0 + ev*std::exp(-beta*B) ).log().sum() + ( 1.0 + ev*std::exp(+beta*B) ).log().sum();
		std::complex<double> other = (evb.cast<std::complex<double>>() + std::exp(+beta*B)*eva).array().log().sum() - evb.array().log().sum()
					   + (evb.cast<std::complex<double>>() + std::exp(-beta*B)*eva).array().log().sum() - evb.array().log().sum();

		if (std::norm(ret - other)>1e-9 && false) {
			std::cerr << eva.transpose() << std::endl << std::endl;
			std::cerr << evb.transpose() << std::endl << std::endl;
			std::cerr << (eva.array()/evb.array().cast<std::complex<double>>()).transpose() << std::endl << std::endl;
			std::cerr << ev.transpose()  << std::endl << std::endl;
			throw("wrong");
		}

		if (std::cos(ret.imag())<0.99 && Q<100) {
			std::cerr << "increasing number of decompositions: " << Q << " -> " << Q+1 << " (number of slices = " << diagonals.size() << ")" <<  std::endl;
			return logProbability(Q+1);
		}
		if (std::cos(ret.imag())<0.99) {
			Eigen::MatrixXd S1 = Eigen::MatrixXd::Identity(V, V) + std::exp(+beta*B)*positionSpace;
			Eigen::MatrixXd S2 = Eigen::MatrixXd::Identity(V, V) + std::exp(-beta*B)*positionSpace;
			std::cerr << positionSpace << std::endl << std::endl;
			std::cerr << S1 << std::endl << std::endl;
			std::cerr << S1.eigenvalues().transpose() << std::endl;
			std::cerr << S2.eigenvalues().transpose() << std::endl;
			std::cerr << S1.eigenvalues().array().log().sum() << std::endl;
			std::cerr << S2.eigenvalues().array().log().sum() << std::endl;
			std::cerr << eva.transpose() << std::endl << std::endl;
			std::cerr << evb.transpose() << std::endl << std::endl;
			std::cerr << (eva.array()/evb.array().cast<std::complex<double>>()).transpose() << std::endl << std::endl;
			std::cerr << ev.transpose() << std::endl << std::endl;
			std::cerr << ret << ' ' << diagonals.size() << std::endl;
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
			computeNumber();
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
		std::map<double, Eigen::VectorXd>::iterator diter = diagonals.find(t);
		if (diter!=diagonals.end()) return false;
		diagonals[t] = Eigen::VectorXd::Zero(V);
		diter = diagonals.find(t);
		for (int i=0;i<V;i++) diter->second[i] = distribution(generator)?A:-A;
		double trial = logProbability();
		//if (randomDouble(generator)<std::exp(trial-plog)*beta/diagonals.size()) {
		if (-trialDistribution(generator)<trial-plog+std::log(beta)-std::log(diagonals.size())) {
			//std::cerr << "accepted increase: time steps = " << diagonals.size() << std::endl;
			plog = trial;
			computeNumber();
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
		//if (randomDouble(generator)<std::exp(trial-plog)*(diagonals.size()+1)/beta) {
		if (-trialDistribution(generator)<trial-plog+std::log(diagonals.size()+1)-std::log(beta)) {
			plog = trial;
			computeNumber();
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

	void update () {
		metropolis(1);
	}

	void measure () {
		sweeps++;
		if (sweeps > thermalization_sweeps) {
			measurements["n_up"] << numberUp();
			measurements["n_dn"] << numberDown();
			measurements["slices"] << sliceNumber();
		}
	}

	double fraction_completed() const {
		return (sweeps<thermalization_sweeps ? 0. : (sweeps-thermalization_sweeps) / double(total_sweeps));
	}


	double sliceNumber () { return diagonals.size(); }
	double numberUp () { return n_up; }
	double numberDown () { return n_dn; }

	void printResults () {
		std::cout << measurements["n_up"] << std::endl << measurements["n_dn"] << std::endl << measurements["slices"] << std::endl;
	}

	~ctaux_sim () { fftw_destroy_plan(x2p); fftw_destroy_plan(p2x); }
	protected:
};

#endif // CT_AUX_HPP

