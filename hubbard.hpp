#ifndef HUBBARD_HPP
#define HUBBARD_HPP

#include <Eigen/Dense>
#include <random>

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
	typedef Eigen::VectorXd UpdateType;
	HubbardInteraction (std::mt19937_64 &g) : generator(g), coin_flip(0.5), random_time(0.0, 1.0) {}
	void setup (const Eigen::MatrixXd &A, double u, double k);
	size_t volume () const { return V; }
	size_t states () const { return N; }
	size_t dimension () const { return N; }
	Vertex generate ();
	Vertex generate (double tau);
	Vertex generate (double t0, double t1);
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

	UpdateType matrixU (const Vertex v) const { return eigenvectors.row(v.x).transpose(); }
	UpdateType matrixVt (const Vertex v) const { return eigenvectors.row(v.x).transpose(); }
	double scalarA () const { return a; }
	double scalarB () const { return b; }

	double log_abs_det (const Vertex v) const { return 0.0; }
};

#endif // HUBBARD_HPP

