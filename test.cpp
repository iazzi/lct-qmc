#include "measurements.hpp"
#include <random>
#include <cmath>

#include <alps/alea.h>


int main (int argc, char** argv) {
	alps::RealObservable o("alps obs");
	mymeasurement<double> m;
	std::mt19937_64 generator;
	std::uniform_real_distribution<double> randomDouble;

	int N;
	double x = 0.0;
	std::cin >> N;
	for (int i=0;i<N;i++) {
		double t = randomDouble(generator);
		if (randomDouble(generator)<0.5) t = -t;
		t *= 0.01;
		//std::cout << x << ' ' << t << std::endl;
		if (randomDouble(generator)<exp( - t*t - 2*t*x) ) {
			x = x + t;
		}
		m.add(x);
		o << x;
	}
	std::cout << m << std::endl;
	std::cout << std::endl << o << std::endl;
	return 0;
}

