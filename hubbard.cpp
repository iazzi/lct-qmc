#include "hubbard.hpp"

#include <cmath>

void HubbardInteraction::setup (const Eigen::MatrixXd &A, double u, double k) {
	eigenvectors = A;
	U = u;
	K = k;
	N = A.diagonal().size();
	coin_flip = std::bernoulli_distribution(0.5);
	random_site = std::uniform_int_distribution<size_t>(0, N-1);
	a = 1.0*U/2.0/K;
	b = sqrt(U/K+a*a);
}

HubbardInteraction::Vertex HubbardInteraction::generate () {
	HubbardInteraction::Vertex ret;
	ret.sigma = coin_flip(generator)?(a+b):(a-b);
	ret.x = random_site(generator);
	ret.tau = random_time(generator);
	return ret;
}

HubbardInteraction::Vertex HubbardInteraction::generate (double t0, double t1) {
	HubbardInteraction::Vertex ret;
	ret.sigma = coin_flip(generator)?(a+b):(a-b);
	ret.x = random_site(generator);
	ret.tau = t0 + random_time(generator)*(t1-t0);
	return ret;
}

