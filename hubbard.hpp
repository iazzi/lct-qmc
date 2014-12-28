#ifndef HUBBARD_HPP
#define HUBBARD_HPP

#include <Eigen/Dense>
#include <random>

// FIXME
#include <iostream>

struct HubbardVertex {
	int x;
	double sigma;
	double tau;
	struct Compare {
		bool operator() (const HubbardVertex& a, const HubbardVertex& b) {
			return (a.tau<b.tau) || (a.tau==b.tau && a.x<b.x)
				|| (a.tau==b.tau && a.x==b.x && (std::fabs(a.sigma)<std::fabs(b.sigma)))
				|| (a.tau==b.tau && a.x==b.x && std::fabs(a.sigma)==std::fabs(b.sigma) && a.sigma<b.sigma);
		}
	};
};

//
// class HubbardInteraction
//
// contains information to generate and manipulate the vertices for a Hubbard-like interaction
// of the form U n_\Up n_\Dn.
//
// The expansion is in the operator  -(K - U n_\Up n_\Dn) with K>0
//
// Assumption that the lattice matrix represents the full @V \timex @V matrix for both spin species
// with spin up in the first V elements and spin down in the second half
//
// a Vertex contains the interacting site x, the strength sigma and the time tau
// it corresponds to a rank-2 matrix of the form U V^t = sigma_\Up * u_\Up v^t_\Up + sigma_\Dn * u_Dn v^t_Dn
//
// sigma_\Up = A \pm B
// sigma_\Dn = A \mp B = 2*A - sigma_\Up
//
// The UpdateType type contains
//
class HubbardInteraction {
	std::mt19937_64 &generator;
	Eigen::MatrixXd eigenvectors;
	double U;
	double K;
	size_t N;
	size_t V;
	double a, b;
	std::bernoulli_distribution coin_flip;
	std::uniform_int_distribution<size_t> random_site;
	std::uniform_real_distribution<double> random_time;
	public:
	typedef HubbardVertex Vertex;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 2> UpdateType;
	HubbardInteraction (std::mt19937_64 &g) : generator(g), coin_flip(0.5), random_time(0.0, 1.0) {}

	inline void setup (const Eigen::MatrixXd &A, double u, double k) {
		eigenvectors = A;
		U = u;
		K = k;
		N = A.diagonal().size();
		V = N/2; // FIXME assert N even?
		coin_flip = std::bernoulli_distribution(0.5);
		random_site = std::uniform_int_distribution<size_t>(0, V-1);
		a = 1.0*U/2.0/K;
		b = sqrt(U/K+a*a);
	}

	inline Vertex generate () {
		HubbardInteraction::Vertex ret;
		ret.sigma = coin_flip(generator)?(+b):(-b);
		ret.x = random_site(generator);
		ret.tau = random_time(generator);
		return ret;
	}

	Vertex generate (double tau);

	inline Vertex generate (double t0, double t1) {
		HubbardInteraction::Vertex ret;
		ret.sigma = coin_flip(generator)?(+b):(-b);
		ret.x = random_site(generator);
		ret.tau = t0 + random_time(generator)*(t1-t0);
		return ret;
	}

	size_t volume () const { return V; }
	size_t states () const { return N; }
	size_t dimension () const { return N; }
	template <typename T>
		void apply_vertex_on_the_left (Vertex v, T &M) const {
			M += (a+v.sigma) * eigenvectors.row(v.x).transpose() * (eigenvectors.row(v.x) * M)
				+ (a-v.sigma) * eigenvectors.row(v.x+V).transpose() * (eigenvectors.row(v.x+V) * M);
		}

	template <typename T>
		void apply_vertex_on_the_right (Vertex v, T &M) const {
			M += (a+v.sigma) * (M * eigenvectors.row(v.x).transpose()) * eigenvectors.row(v.x)
				+ (a-v.sigma) * (M * eigenvectors.row(v.x+V).transpose()) * eigenvectors.row(v.x+V);
		}

	template <typename T>
		void apply_inverse_on_the_left (Vertex v, T &M) const {
			M -= (a+v.sigma)/(1.0+a+v.sigma) * eigenvectors.row(v.x).transpose() * (eigenvectors.row(v.x) * M)
				+ (a-v.sigma)/(1.0+a-v.sigma) * eigenvectors.row(v.x+V).transpose() * (eigenvectors.row(v.x+V) * M);
		}

	template <typename T>
		void apply_inverse_on_the_right (Vertex v, T &M) const {
			M -= (a+v.sigma)/(1.0+a+v.sigma) * (M * eigenvectors.row(v.x).transpose()) * eigenvectors.row(v.x)
				+ (a-v.sigma)/(1.0+a-v.sigma) * (M * eigenvectors.row(v.x+V).transpose()) * eigenvectors.row(v.x+V);
		}

	UpdateType matrixU (const Vertex v) const {
		UpdateType ret(N, 2);
		ret.col(0) = (a+v.sigma) * eigenvectors.row(v.x).transpose();
		ret.col(1) = (a-v.sigma) * eigenvectors.row(v.x+V).transpose();
		return ret;
	}

	UpdateType matrixV (const Vertex v) const {
		UpdateType ret(N, 2);
		ret.col(0) = eigenvectors.row(v.x).transpose();
		ret.col(1) = eigenvectors.row(v.x+V).transpose();
		return ret;
	}

	double scalarA () const { return a; }
	double scalarB () const { return b; }

	double log_abs_det (const Vertex v) const { return 0.0; }
	double combinatorial_factor () { return log(K*volume()); }
};

#endif // HUBBARD_HPP

