#ifndef HUBBARD_HPP
#define HUBBARD_HPP

#include <Eigen/Dense>
#include <random>

class HubbardInteraction {
	std::mt19937_64 &generator;
	Eigen::MatrixXd eigenvalues;
	double U;
	double K;
	double N;
	double a, b;
	std::bernoulli_distribution coin_flip;
	std::uniform_int_distribution<size_t> random_site;
	public:
	typedef struct { int x; double sigma; double tau; } Vertex;
	HubbardInteraction (std::mt19937_64 &g) : generator(g) {}
	void setup (const Eigen::MatrixXd &A, double u, double k);
	Vertex generate ();
	Vertex generate (double tau);
	Vertex generate (double t0, double t1);
};

#endif // HUBBARD_HPP

