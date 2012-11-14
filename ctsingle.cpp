#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <functional>

#include <alps/alea.h>
#include <alps/alea/mcanalyze.hpp>

static const double pi = 3.141592653589793238462643383279502884197;

class Configuration {
	private:
	double beta; // inverse temperature
	double g; // interaction strength
	double mu; // chemical potential
	double A; // sqrt(g)
	double B; // magnetic field

	std::map<double, double> diagonals;

	std::default_random_engine generator;
	std::bernoulli_distribution distribution;
	std::uniform_real_distribution<double> randomDouble;
	std::uniform_real_distribution<double> randomTime;
	std::exponential_distribution<double> trialDistribution;

	double positionSpace; // current matrix in position space

	double plog;

	public:

	double n_up;
	double n_dn;

	public:

	Configuration (double Beta, double interaction, double m, double b)
		: beta(Beta), g(interaction), mu(m), B(b), distribution(0.5), randomDouble(1.0),
		randomTime(0, Beta), trialDistribution(1.0) {
		A = sqrt(g);

		positionSpace = 1.0;

		plog = logProbability();
	}

	void computeNumber () {
		n_up = ( 1.0 - 1.0/(1.0 + exp(+beta*B+beta*mu) * positionSpace) );
		n_dn = ( 1.0 - 1.0/(1.0 + exp(-beta*B+beta*mu) * positionSpace) );
	}

	double logProbability (int Q = 1) {
		positionSpace = 1.0;
		for (auto i : diagonals) {
			positionSpace *= 1.0 + i.second;
		}
		double ev = positionSpace;

		std::complex<double> ret = std::log( 1.0 + ev*std::exp(-beta*B+beta*mu) ) + std::log( 1.0 + ev*std::exp(+beta*B+beta*mu) );

		return ret.real();
	}

	bool metropolisFlip () {
		if (diagonals.size()==0) return false;
		bool ret = false;
		int t = std::uniform_int_distribution<int>(0, diagonals.size()-1)(generator);
		std::map<double, double>::iterator diter = diagonals.begin();
		while (t--) diter++;
		(*diter).second = -(*diter).second;
		double trial = logProbability();
		if (-trialDistribution(generator)<trial-plog) {
			plog = trial;
			computeNumber();
			ret = true;
		} else {
			(*diter).second = -(*diter).second;
		}
		return ret;
	}

	bool metropolisUp () {
		bool ret = false;
		double t = randomTime(generator);
		if (diagonals.find(t)!=diagonals.end()) return false;
		std::map<double, double>::iterator diter = diagonals.insert(std::pair<double,double>(t, distribution(generator)?A:-A)).first;
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
		std::pair<double, double> store;
		int t = std::uniform_int_distribution<int>(0, diagonals.size()-1)(generator);
		std::map<double, double>::iterator diter = diagonals.begin();
		while (t--) diter++;
		store = *diter;
		diagonals.erase(diter);
		double trial = logProbability();
		//if (randomDouble(generator)<std::exp(trial-plog)*(diagonals.size()+1)/beta) {
		if (-trialDistribution(generator)<trial-plog+std::log(diagonals.size()+1)-std::log(beta)) {
			//std::cerr << "accepted decrease: time steps = " << diagonals.size() << std::endl;
			plog = trial;
			computeNumber();
			ret = true;
		} else {
			diagonals[store.first] = store.second;
			ret = false;
		}
		return ret;
	}

	bool metropolis () {
		std::discrete_distribution<int> distribution { 0.90, 0.05, 0.05 };
		int type = distribution(generator);
		if (type == 0 || (type == 1 && diagonals.size()==0 )) {
			return metropolisFlip();
		} else if (type == 1) {
			//std::cerr << "proposed decrease" << std::endl;
			return metropolisDown();
		} else if (type == 2) {
			//std::cerr << "proposed increase" << std::endl;
			return metropolisUp();
		}
	}

	int sliceNumber () { return diagonals.size(); }

	void print () {
		for (auto i : diagonals) {
			std::cout << i.first << '\t';
			for (int j=0;j<1;j++) {
				std::cout << (i.second<0?'-':'+');
			}
			std::cout << std::endl;
		}
	}

	~Configuration () {}
	protected:
};

int main (int argc, char **argv) {
	double beta = 10.0;
	double g = 0.1;
	double mu = -0.5;
	double B = 0.0;

	alps::RealObservable d_up("d_up");
	alps::RealObservable d_dn("d_dn");

	for (int i=1;i<argc;i++) {
		if (argv[i][0]=='-') {
			switch (argv[i][1]) {
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
			}
		}
	}

	double p_00 = 1;
	double p_01 = exp(-beta*(-mu+B));
	double p_10 = exp(-beta*(-mu-B));
	double p_11 = exp(-beta*(-2.0*mu-g));
	double Z = p_00 + p_01 + p_10 + p_11;
	p_00 /= Z;
	p_01 /= Z;
	p_10 /= Z;
	p_11 /= Z;

	double n_up = p_10+p_11;
	double n_dn = p_01+p_11;

	Configuration configuration(beta, g, mu, B);

	int n = 0;
	int a = 0;
	for (int i=0;i<100000;i++) {
		if (i%100==0) { std::cout << i << "\r"; std::cout.flush(); }
		configuration.metropolis();
	}

	for (;;) {
		if (configuration.metropolis()) a++;
		n++;
		d_up << configuration.n_up;
		d_dn << configuration.n_dn;
		if (n%(1<<10)==0) {
			std::cout << "temperature = " << (1.0/beta) << ", interaction = " << g << std::endl;
			std::cout << "chemical potential = " << mu << ", magnetic field = " << B << std::endl;
			std::cout << "acceptance = " << (double(a)/double(n)) << " spin flips = " << 1 << std::endl;
			std::cout << "slices = " << configuration.sliceNumber() << std::endl;
			std::cout << "n_up = " << n_up << std::endl;
			std::cout << "n_dn = " << n_dn << std::endl;
			std::cout << d_up << std::endl;
			std::cout << d_dn << std::endl;
		}
	}
	return 0;
}

