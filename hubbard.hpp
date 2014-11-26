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

class HubbardInteraction {
	std::mt19937_64 &generator;
	Eigen::MatrixXd eigenvectors;
	double U;
	double K;
	size_t N;
	double a, b;
	std::bernoulli_distribution coin_flip;
	std::uniform_int_distribution<size_t> random_site;
	std::uniform_real_distribution<double> random_time;
	public:
	typedef HubbardVertex Vertex;
	typedef Eigen::VectorXd UpdateType;
	HubbardInteraction (std::mt19937_64 &g) : generator(g), coin_flip(0.5), random_time(0.0, 1.0) {}
	void setup (const Eigen::MatrixXd &A, double u, double k);
	size_t volume () const { return N; }
	Vertex generate ();
	Vertex generate (double tau);
	Vertex generate (double t0, double t1);
	template <typename T>
		void apply_vertex_on_the_left (Vertex v, T &M) const {
			M += v.sigma * eigenvectors.row(v.x).transpose() * (eigenvectors.row(v.x) * M);
		}

	template <typename T>
		void apply_vertex_on_the_right (Vertex v, T &M) const {
			M += v.sigma * (M * eigenvectors.row(v.x).transpose()) * eigenvectors.row(v.x);
		}

	template <typename T>
		void apply_inverse_on_the_left (Vertex v, T &M) const {
			M -= v.sigma/(1.0+v.sigma) * eigenvectors.row(v.x).transpose() * (eigenvectors.row(v.x) * M);
		}

	template <typename T>
		void apply_inverse_on_the_right (Vertex v, T &M) const {
			M -= v.sigma/(1.0+v.sigma) * (M * eigenvectors.row(v.x).transpose()) * eigenvectors.row(v.x);
		}

	UpdateType matrixU (const Vertex v) const { return eigenvectors.row(v.x).transpose(); }
	UpdateType matrixVt (const Vertex v) const { return eigenvectors.row(v.x).transpose(); }
};

#endif // HUBBARD_HPP

