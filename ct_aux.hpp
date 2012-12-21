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

void dggev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta);

class ctaux_sim : public alps::mcbase_ng {
	private:
	int L; // size of the system
	int Dim; // dimension
	int V; // volume of the system
	double beta; // inverse temperature
	double U; // interaction strength
	double mu; // chemical potential
	double A; // sqrt(U)
	double B; // magnetic field
	double t; // nearest neighbour hopping
	double J; // next-nearest neighbour hopping

	int MIN_SLICES;
	int MAX_SLICES;

	std::map<double, Eigen::VectorXd> diagonals;

	//std::default_random_engine generator;
	std::mt19937_64 generator;
	std::bernoulli_distribution distribution;
	std::uniform_real_distribution<double> randomDouble;
	std::uniform_real_distribution<double> randomTime;
	std::exponential_distribution<double> trialDistribution;

	Eigen::VectorXd energies;

	Eigen::MatrixXd positionSpace; // current matrix in position space
	Eigen::MatrixXcd momentumSpace;

	Eigen::MatrixXd matrixQ; // Q in the QDT decomposition of A_s
	Eigen::VectorXd vectorD; // D in the QDT decomposition of A_s
	Eigen::MatrixXd matrixT; // T in the QDT decomposition of A_s

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
			<< alps::ngs::RealObservable("acceptance")
			<< alps::ngs::RealObservable("slices")
			<< alps::ngs::RealVectorObservable("slice_distr");

		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		const int size[] = { L, L, L, };
		x2p = fftw_plan_many_dft_r2c(Dim, size, V, positionSpace.data(),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_PATIENT);
		p2x = fftw_plan_many_dft_c2r(Dim, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, positionSpace.data(), NULL, 1, V, FFTW_PATIENT);

		positionSpace = Eigen::MatrixXd::Identity(V, V);
		momentumSpace = Eigen::MatrixXcd::Identity(V, V);

		energies = Eigen::VectorXd::Zero(V);
		for (int i=0;i<V;i++) {
			energies[i] += -2.0*t * ( cos(2.0*(i%L)*pi/L) + cos(2.0*((i/L)%L)*pi/L) + cos(2.0*(i/L/L)*pi/L) - (3.0 - Dim) );
			energies[i] += -2.0*J * ( cos(4.0*(i%L)*pi/L) + cos(4.0*((i/L)%L)*pi/L) + cos(4.0*(i/L/L)*pi/L) - (3.0 - Dim) );
		}

		plog = logProbability();
	}

	ctaux_sim (parameters_type const &params, std::size_t seed_offset = 42) :
		mcbase_ng(params, seed_offset),
		sweeps(0),
		thermalization_sweeps(int(params["THERMALIZATION"])),
		total_sweeps(int(params["SWEEPS"])),
		L(params["L"]),
		beta(1.0/double(params["T"])),
		U(-double(params["U"])),
		mu(params["mu"]),
		A(sqrt(U)),
		B(params["B"]),
		t(params["t"]),
		J(params["J"]),
		distribution(0.5),
		randomDouble(1.0),
		randomTime(0, beta),
		trialDistribution(1.0)
	{
		if (params["LATTICE"].cast<std::string>()==std::string("chain lattice")) {
			Dim = 1;
		} else if (params["LATTICE"].cast<std::string>()==std::string("square lattice")) {
			Dim = 2;
		} else if (params["LATTICE"].cast<std::string>()==std::string("simple cubic lattice")) {
			Dim = 3;
		} else {
			throw std::string("unknown lattice type");
		}
		//MIN_SLICES = int(params["MIN_SLICES"]);
		//MAX_SLICES = int(params["MAX_SLICES"]);
		MIN_SLICES = 0;
		MAX_SLICES = INT_MAX;
		if (MIN_SLICES<0) MIN_SLICES = 0;
		if (MAX_SLICES<MIN_SLICES) MAX_SLICES = MIN_SLICES;
		V = std::pow(L, Dim);
		if (L==1) {
			t = J = 0.0;
		}
		//mu -= U/sqrt(2);
		init();
	}

	std::complex<double> logProbabilityFromEigenvalues (const Eigen::VectorXcd& ev) {
		std::complex<double> ret = (1.0 + std::exp(+beta*B+beta*mu)*ev.array()).log().sum()
					 + (1.0 + std::exp(-beta*B+beta*mu)*ev.array()).log().sum();
		return ret;
	}

	void computeNumber () {
		//Eigen::MatrixXd Q_plus_DT1 = matrixQ.transpose() + exp(+beta*B+beta*mu) * vectorD.asDiagonal() * matrixT;
		//Eigen::MatrixXd G1 = Q_plus_DT1.inverse() * exp(+beta*B+beta*mu) * vectorD.asDiagonal() * matrixT;
		//Eigen::MatrixXd Q_plus_DT2 = matrixQ.transpose() + exp(-beta*B+beta*mu) * vectorD.asDiagonal() * matrixT;
		//Eigen::MatrixXd G2 = Q_plus_DT2.inverse() * exp(-beta*B+beta*mu) * vectorD.asDiagonal() * matrixT;
		//n_up = G1.trace() / V;
		//n_dn = G2.trace() / V;
		//std::cerr << "n_up " << n_up << std::endl 
			//<< positionSpace << std::endl << std::endl
			//<< G1 << std::endl << std::endl
			//<< Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B+beta*mu) * positionSpace).inverse() << std::endl << std::endl;
		n_up = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B+beta*mu) * positionSpace).inverse() ).trace();
		n_dn = ( Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(-beta*B+beta*mu) * positionSpace).inverse() ).trace();
		if (isinf(n_up) || isinf(n_dn) || isnan(n_up) || isnan(n_dn)) {
			std::cerr << "positionSpace\n" << exp(+beta*B+beta*mu) * positionSpace << std::endl << std::endl;
			std::cerr << "(1+positionSpace)^-1\n" << (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B+beta*mu) * positionSpace).inverse() << std::endl << std::endl;
			std::cerr << "n_up " << n_up << std::endl << Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(+beta*B+beta*mu) * positionSpace).inverse() << std::endl << std::endl;
			std::cerr << "n_dn " << n_dn << std::endl << Eigen::MatrixXd::Identity(V, V) - (Eigen::MatrixXd::Identity(V, V) + exp(-beta*B+beta*mu) * positionSpace).inverse() << std::endl << std::endl;
			throw "shit at computeNumber";
		}
	}

	Eigen::MatrixXd propagator (double from, double to) {
		double t = from;
		double dt;
		positionSpace.setIdentity(V, V);
		for (auto i=diagonals.lower_bound(from);i!=diagonals.upper_bound(to);i++) {
			fftw_execute(x2p);
			dt = (*i).first-t;
			momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
			fftw_execute(p2x);
			positionSpace /= V;
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+(*i).second).asDiagonal());
			t = (*i).first;
		}
		fftw_execute(x2p);
		dt = to-t;
		momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
		fftw_execute(p2x);
		positionSpace /= V;
		return positionSpace;
	}

	double logProbability2 (int nsteps = 1) {
		std::vector<Eigen::MatrixXd> steps(nsteps, Eigen::MatrixXd::Zero(V, V));
		for (int i=0;i<nsteps;i++) {
			steps[i] = propagator(beta/nsteps*i, beta/nsteps*(i+1));
			//std::cerr << "steps[" << i << "]\n" << steps[i] << std::endl << std::endl;
		}
		Eigen::HouseholderQR<Eigen::MatrixXd> decomposer;
		Eigen::MatrixXd T = Eigen::MatrixXd::Identity(V, V);
		Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(V, V);
		Eigen::VectorXd D = Eigen::VectorXd::Ones(V);
		//Eigen::MatrixXd test = Eigen::MatrixXd::Identity(V, V);
		for (int i=0;i<nsteps;i++) {
			Eigen::MatrixXd Z = steps[i]*Q;
			decomposer.compute(Z*D.asDiagonal());
			Eigen::MatrixXd R = decomposer.matrixQR().triangularView<Eigen::Upper>();
			D = R.diagonal();
			R.applyOnTheLeft(D.array().inverse().matrix().asDiagonal());
			T.applyOnTheLeft(R);
			//std::cerr << "R\n" << R << std::endl << std::endl << "T\n" << T << std::endl << std::endl;
			Q = decomposer.householderQ();
			//test.applyOnTheLeft(steps[i]);
		}
		Eigen::VectorXcd ev1 = (D.asDiagonal()*(T*Q)).eigenvalues();
		double lambda = D.array().abs().maxCoeff();
		Eigen::VectorXcd ev2 = lambda - (lambda*Eigen::MatrixXd::Identity(V, V) - Q*D.asDiagonal()*T).eigenvalues().array();
		//std::cerr << "ev1 " << ev1.transpose() << std::endl;
		//std::cerr << "ev2 " << ev2.transpose() << std::endl;
		//Eigen::VectorXcd eva;
		//Eigen::VectorXd evb;
		//dggev(test, Eigen::MatrixXd::Identity(V, V), eva, evb);
		//dggev(Q*D.asDiagonal()*T, Eigen::MatrixXd::Identity(V, V), eva, evb);
		//dggev(D.asDiagonal()*T*Q, Eigen::MatrixXd::Identity(V, V), eva, evb);
		//std::cerr << Q*D.asDiagonal()*T << std::endl << std::endl;
		//std::cerr << D.transpose() << std::endl;
		//std::cerr << eva.transpose() << std::endl;
		//std::cerr << evb.transpose() << std::endl;
		//std::cerr << (eva.array() / evb.cast<std::complex<double>>().array()).transpose() << std::endl;
		//std::complex<double> ret = (evb.cast<std::complex<double>>() + std::exp(+beta*B)*eva).array().log().sum() - evb.array().log().sum()
					 //+ (evb.cast<std::complex<double>>() + std::exp(-beta*B)*eva).array().log().sum() - evb.array().log().sum();
		std::complex<double> ret = logProbabilityFromEigenvalues(ev1);
		double check = T.sum() + Q.sum() + D.sum();
		if ( ev1.array().array().prod().real()<0 || isinf(check) || isnan(check) ) {
			//double lambda = (eva.array() / evb.cast<std::complex<double>>().array()).abs().maxCoeff();
			//dggev(-(D.asDiagonal()*T*Q), Eigen::MatrixXd::Identity(V, V), eva, evb);
			//std::cerr << (eva.array() / evb.cast<std::complex<double>>().array()).transpose() << std::endl;
			//dggev(lambda*Eigen::MatrixXd::Identity(V, V) - D.asDiagonal()*T*Q, Eigen::MatrixXd::Identity(V, V), eva, evb);
			//std::cerr << (eva.array() / evb.cast<std::complex<double>>().array() - lambda).transpose() << std::endl;
			if (nsteps < 100) {
				//std::cerr << "increasing nsteps " << nsteps << " -> " << nsteps+1 << " (" << sliceNumber() << " slices)" << std::endl;
				return logProbability2(nsteps+1);
			} else {
				throw "shit at " __FILE__ " too many recursions";
			}
		}
		if (std::cos(ret.imag())<0.99 || isnan(ret.real())) {
			std::cerr << "ev1 " << ev1.transpose() << std::endl;
			std::cerr << "ev2 " << ev2.transpose() << std::endl;
			throw "shit at " __FILE__ " sign problem appeared";
		}
		matrixQ = Q;
		vectorD = D;
		matrixT = T;
		positionSpace = Q*D.asDiagonal()*T;
		Eigen::MatrixXd Q_plus_DT = Q.transpose() + exp(+beta*B+beta*mu) * D.asDiagonal() * T;
		Eigen::MatrixXd G = Q_plus_DT.inverse() * exp(+beta*B+beta*mu) * D.asDiagonal() * T;
		//std::cerr << "positionSpace at logProbability2\n" << positionSpace << std::endl << std::endl;
		//std::cerr << "Q_plus_DT\n" << Q_plus_DT << std::endl << std::endl;
		//std::cerr << "G\n" << G << std::endl << std::endl;
		//std::cerr << "Q\n" << Q << std::endl << std::endl;
		//std::cerr << "D\n" << D.transpose() << std::endl << std::endl;
		//std::cerr << "T\n" << T << std::endl << std::endl;
		//std::cerr << "ev1 " << ev1.transpose() << std::endl;
		//std::cerr << "ev2 " << ev2.transpose() << std::endl;
		return ret.real();
	}

	double logProbability () {
		//double other = logProbability2(1);
		//return other;
		double t = 0.0;
		double dt;
		positionSpace.setIdentity(V, V);
		//std::cerr << positionSpace << std::endl;
		for (auto i : diagonals) {
			fftw_execute(x2p);
			dt = i.first-t;
			momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
			fftw_execute(p2x);
			positionSpace /= V;
			//std::cerr << " after K " << positionSpace << std::endl;
			positionSpace.applyOnTheLeft((Eigen::VectorXd::Constant(V, 1.0)+i.second).asDiagonal());
			t = i.first;
			//std::cerr << " after V " << positionSpace << std::endl;
		}
		fftw_execute(x2p);
		dt = beta-t;
		momentumSpace.applyOnTheLeft((-dt*energies).array().exp().matrix().asDiagonal());
		fftw_execute(p2x);
		positionSpace /= V;
		//std::cerr << positionSpace << std::endl;
		//std::cerr << "energies " << energies.transpose() << std::endl << std::endl;

		//Eigen::VectorXcd eva;
		//Eigen::VectorXd evb;
		//dggev(R, positionSpace.inverse(), eva, evb);

		Eigen::VectorXcd ev = positionSpace.eigenvalues();

		std::complex<double> ret = logProbabilityFromEigenvalues(ev);
		//std::complex<double> other = (evb.cast<std::complex<double>>() + std::exp(+beta*B)*eva).array().log().sum() - evb.array().log().sum()
					   //+ (evb.cast<std::complex<double>>() + std::exp(-beta*B)*eva).array().log().sum() - evb.array().log().sum();


		//std::cerr << ev.transpose() << std::endl;
		//std::cerr << "old logProbability: " << ret << " new logProbability: " << other << std::endl << std::endl;

		if (std::cos(ret.imag())<0.99) {
			std::cerr << positionSpace << std::endl << std::endl;
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
		if ( type == 0 || (type == 1 && diagonals.size()<=MIN_SLICES) || (type == 2 && diagonals.size()>=MAX_SLICES) ) {
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
		measurements["acceptance"] << (metropolis(1) ? 1.0 : 0.0);
	}

	void measure () {
		sweeps++;
		if (sweeps > thermalization_sweeps) {
			measurements["n_up"] << numberUp()/V;
			measurements["n_dn"] << numberDown()/V;
			measurements["slices"] << sliceNumber();
			//const int n_hist = 50;
			//std::vector<double> times_hist(n_hist+1, 0.0);
			//for (auto i : diagonals) {
				//int x = int(n_hist*i.first/beta);
				//times_hist[x] += 1;
			//}
			//cmeasurements["slice_distr"] << times_hist;
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

