#ifndef HUBBARD_HPP
#define HUBBARD_HPP

#include "parameters.hpp"
#include "modelbase.hpp"

#include <Eigen/Dense>
#include <random>

// FIXME
#include <iostream>
#include <fstream>

typedef Eigen::Matrix<double, Eigen::Dynamic, 2> HubbardVertexMatrix;

struct HubbardVertexUpdateData {
	HubbardVertexMatrix U, V;
	Eigen::Vector2d mat, inv;
	HubbardVertexUpdateData () noexcept {}
	~HubbardVertexUpdateData () noexcept {}
	HubbardVertexUpdateData (const HubbardVertexUpdateData &d) noexcept : U(d.U), V(d.V), mat(d.mat), inv(d.inv) {}
	HubbardVertexUpdateData (HubbardVertexUpdateData &&d) noexcept : mat(d.mat), inv(d.inv) {
		U.swap(d.U);
		V.swap(d.V);
	}
	HubbardVertexUpdateData &operator= (const HubbardVertexUpdateData &d) noexcept {
		U = d.U;
		V = d.V;
		mat = d.mat;
		inv = d.inv;
	}
	HubbardVertexUpdateData &operator= (HubbardVertexUpdateData &&d) noexcept {
		U.swap(d.U);
		V.swap(d.V);
		mat = d.mat;
		inv = d.inv;
	}
};

struct HubbardVertex {
	HubbardVertexUpdateData data;
	double sigma;
	double tau;
	int x;
	struct Compare {
		bool operator() (const HubbardVertex& a, const HubbardVertex& b) {
			return (a.tau<b.tau) || (a.tau==b.tau && a.x<b.x)
				|| (a.tau==b.tau && a.x==b.x && (std::fabs(a.sigma)<std::fabs(b.sigma)))
				|| (a.tau==b.tau && a.x==b.x && std::fabs(a.sigma)==std::fabs(b.sigma) && a.sigma<b.sigma);
		}
	};
	bool operator== (const HubbardVertex &w) const { return x==w.x && sigma==w.sigma && tau==w.tau; }
	HubbardVertex () : sigma(0.0), tau(0.0), x(0) {}
	HubbardVertex (double t) : sigma(0.0), tau(t), x(0) {}
	HubbardVertex (int y, double s, double t) : sigma(s), tau(t), x(y) {}
	HubbardVertex (const HubbardVertex &v) : data(v.data), sigma(v.sigma), tau(v.tau), x(v.x) {}
	HubbardVertex (HubbardVertex &&v) : sigma(v.sigma), tau(v.tau), x(v.x) {
		data.U.swap(v.data.U);
		data.V.swap(v.data.V);
		data.mat = v.data.mat;
		data.inv = v.data.inv;
	}
	HubbardVertex& operator= (const HubbardVertex &v) {
		sigma = v.sigma;
		tau = v.tau;
		x = v.x;
		data = v.data;
		return *this;
	}
	HubbardVertex& operator= (HubbardVertex &&v) {
		sigma = v.sigma;
		tau = v.tau;
		x = v.x;
		data.U.swap(v.data.U);
		data.V.swap(v.data.V);
		data.mat = v.data.mat;
		data.inv = v.data.inv;
		return *this;
	}
};

inline std::ostream &operator<< (std::ostream &f, HubbardVertex v) {
	f << '(' << v.tau << ", " << v.x << ", " << v.sigma << ')';
	return f;
}

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
// The MatrixType type contains
//
template <bool UseSpinBlocks = false>
class HubbardInteraction : public ModelBase {
	double U;
	double K;
	size_t N;
	size_t V;
	size_t I;
	double a, b;
	std::bernoulli_distribution coin_flip;
	std::uniform_int_distribution<size_t> random_site;
	std::uniform_real_distribution<double> random_time;

	Eigen::VectorXd cached_vec;
	HubbardVertexMatrix cached_mat;
	public:
	typedef HubbardVertex Vertex;
	typedef HubbardVertexMatrix MatrixType;
	HubbardInteraction () : coin_flip(0.5), random_time(0.0, 1.0) {}
	HubbardInteraction (const Parameters &p) : coin_flip(0.5), random_time(0.0, 1.0) { setup(p); }
	//HubbardInteraction (const HubbardInteraction &other) : coin_flip(0.5), random_time(0.0, 1.0) { setup(other.U, other.K); }

	//inline void setup (double u, double k) {
		//U = u;
		//K = k;
		//a = 1.0*U/2.0/K;
		//b = sqrt(U/K+a*a);
	//}

	inline void setup (const Parameters &p) {
		if (p.contains("H")) {
			std::string fn = p.getString("H");
			std::ifstream in(fn);
			in >> V;
			H.resize(V, V);
			for (size_t x=0;x<V;x++) {
				for (size_t y=0;y<V;y++) {
					double z;
					in >> z;
					H(x, y) = z;
				}
			}
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
			if (UseSpinBlocks) {
				eigenvectors_ = Eigen::MatrixXd::Zero(2*V, 2*V);
				eigenvectors_.block(0, 0, V, V) = solver.eigenvectors();
				eigenvectors_.block(V, V, V, V) = solver.eigenvectors();
				eigenvalues_.setZero(2*V);
				eigenvalues_.head(V) = solver.eigenvalues();
				eigenvalues_.tail(V) = solver.eigenvalues();
				V = V;
			} else {
				eigenvalues_ = solver.eigenvalues();
				eigenvectors_ = solver.eigenvectors();
				V /= 2;
			}
		} else {
		}
		N = 2*V;
		I = V;
		random_site = std::uniform_int_distribution<size_t>(0, I-1);
		U = p.getNumber("U", 4.0);
		K = p.getNumber("K", 6.0);
		a = 1.0*U/2.0/K;
		b = sqrt(U/K+a*a);
	}

	void set_lattice_eigenvectors (const Eigen::MatrixXd &A) {
		eigenvectors_ = A;
		N = A.diagonal().size();
		V = N/2; // FIXME assert N even?
		I = V;
		random_site = std::uniform_int_distribution<size_t>(0, I-1);
	}

	void set_lattice_eigenvalues (const Eigen::VectorXd &v) {
		eigenvalues_ = v;
	}

	void set_interactive_sites (size_t i) {
		I = std::min(i, V);
		random_site = std::uniform_int_distribution<size_t>(0, I-1);
	}

	template <typename G>
	inline Vertex generate (G &g) {
		return generate(0.0, 1.0, g);
	}

	Vertex generate (double t0, double t1) {
		std::mt19937_64 g;
		return generate(t0, t1, g);
	}

	template <typename G>
	inline Vertex generate (double t0, double t1, G &g) {
		HubbardInteraction::Vertex ret;
		ret.sigma = coin_flip(g)?(+b):(-b);
		ret.x = random_site(g);
		ret.tau = t0 + random_time(g)*(t1-t0);
		prepare(ret);
		return ret;
	}

	void prepare (Vertex &v) {
		cached_vec = eigenvalues_;
		cached_vec *= -v.tau;
		cached_vec = cached_vec.array().exp();
		v.data.U.resize(N, 2);
		v.data.U.col(0) = eigenvectors_.row(v.x).transpose();
		v.data.U.col(1) = eigenvectors_.row(v.x+V).transpose();
		v.data.V = v.data.U;
		v.data.U.array().colwise() /= cached_vec.array();
		v.data.V.array().colwise() *= cached_vec.array();
		v.data.mat << (a+v.sigma), (a-v.sigma);
		v.data.inv << (a+v.sigma)/(1.0+a+v.sigma), (a-v.sigma)/(1.0+a-v.sigma);
	}

	size_t volume () const { return V; }
	size_t interacting_sites () const { return I; }
	size_t states () const { return N; }
	size_t dimension () const { return N; }

	template <typename T>
	void apply_vertex_on_the_left (const Vertex &v, T &M) {
		M += (a+v.sigma) * eigenvectors_.row(v.x).transpose() * (eigenvectors_.row(v.x) * M)
			+ (a-v.sigma) * eigenvectors_.row(v.x+V).transpose() * (eigenvectors_.row(v.x+V) * M);
	}

	template <typename T>
	void apply_vertex_on_the_right (const Vertex &v, T &M) {
		M += (a+v.sigma) * (M * eigenvectors_.row(v.x).transpose()) * eigenvectors_.row(v.x)
			+ (a-v.sigma) * (M * eigenvectors_.row(v.x+V).transpose()) * eigenvectors_.row(v.x+V);
	}

	template <typename T>
	void apply_inverse_on_the_left (const Vertex &v, T &M) {
		M -= (a+v.sigma)/(1.0+a+v.sigma) * eigenvectors_.row(v.x).transpose() * (eigenvectors_.row(v.x) * M)
			+ (a-v.sigma)/(1.0+a-v.sigma) * eigenvectors_.row(v.x+V).transpose() * (eigenvectors_.row(v.x+V) * M);
	}

	template <typename T>
	void apply_inverse_on_the_right (const Vertex &v, T &M) {
		M -= (a+v.sigma)/(1.0+a+v.sigma) * (M * eigenvectors_.row(v.x).transpose()) * eigenvectors_.row(v.x)
			+ (a-v.sigma)/(1.0+a-v.sigma) * (M * eigenvectors_.row(v.x+V).transpose()) * eigenvectors_.row(v.x+V);
	}

	template <typename T>
	void apply_displaced_vertex_on_the_left (const Vertex &v, T &M) {
		cached_mat.noalias() = M.transpose() * v.data.V;
		cached_mat.col(0) *= v.data.mat(0);
		cached_mat.col(1) *= v.data.mat(1);
		M += v.data.U * cached_mat.transpose();
	}

	template <typename T>
	void apply_displaced_vertex_on_the_right (const Vertex &v, T &M) {
		cached_mat.noalias() = M * v.data.U;
		cached_mat.col(0) *= v.data.mat(0);
		cached_mat.col(1) *= v.data.mat(1);
		M += cached_mat * v.data.V.transpose();
	}

	template <typename T>
	void apply_displaced_inverse_on_the_left (const Vertex &v, T &M) {
		cached_mat.noalias() = M.transpose() * v.data.V;
		cached_mat.col(0) *= v.data.inv(0);
		cached_mat.col(1) *= v.data.inv(1);
		M -= v.data.U * cached_mat.transpose();
	}

	template <typename T>
	void apply_displaced_inverse_on_the_right (const Vertex &v, T &M) {
		cached_mat.noalias() = M * v.data.U;
		cached_mat.col(0) *= v.data.inv(0);
		cached_mat.col(1) *= v.data.inv(1);
		M -= cached_mat * v.data.V.transpose();
	}


	double scalarA () const { return a; }
	double scalarB () const { return b; }

	double log_abs_det (const Vertex &v) const { return 0.0; }
	double log_abs_det_block (const Vertex &v, size_t i) const { return std::log(std::fabs(i==0?(1.0+a+v.sigma):(1.0+a-v.sigma))); }
	double combinatorial_factor () { return log(K*interacting_sites()); }

	size_t blocks () const { return UseSpinBlocks?2:1; }
	size_t block_start (size_t i) const { return UseSpinBlocks?(i==0?0:volume()):0; }
	size_t block_size (size_t i) const { return UseSpinBlocks?volume():dimension(); }

	template <typename T>
	double kinetic_energy (const T &M) const {
		return (eigenvalues_.array() * M.diagonal().array()).sum();
	}

	template <typename T>
	Eigen::ArrayXd local_density (const T &M) const {
		return (eigenvectors_ * M * eigenvectors_.transpose()).diagonal();
	}

	template <typename T>
	double interaction_energy (const T &M) const {
		Eigen::ArrayXd d = local_density(M);
		return U * (d.head(V)*d.tail(V)).sum();
	}

	template <typename T>
	void propagate (double t, T& M) {
		cached_vec = eigenvalues_;
		cached_vec *= -t;
		cached_vec = cached_vec.array().exp();
		M.array().colwise() *= cached_vec.array(); // (-t*eigenvalues_.array()).exp(); // this causes allocation!
	}

	template <typename T>
		void propagate_on_the_right (double t, T& M) {
		cached_vec = eigenvalues_;
		cached_vec *= -t;
		cached_vec = cached_vec.array().exp();
		M.array().rowwise() *= cached_vec.transpose().array(); // (-t*eigenvalues_.array()).exp(); // this causes allocation!
	}
};

template <> template <>
inline void HubbardInteraction<true>::apply_vertex_on_the_left (const Vertex &v, Eigen::MatrixXd &M) {
	cached_vec.noalias() = M.block(0, 0, V, V).transpose() * eigenvectors_.block(0, 0, V, V).row(v.x).transpose();
	M.block(0, 0, V, V).noalias() += (a+v.sigma) * eigenvectors_.block(0, 0, V, V).row(v.x).transpose() * cached_vec.transpose();
	cached_vec.noalias() = M.block(V, V, V, V).transpose() * eigenvectors_.block(V, V, V, V).row(v.x).transpose();
	M.block(V, V, V, V).noalias() += (a-v.sigma) * eigenvectors_.block(V, V, V, V).row(v.x).transpose() * cached_vec.transpose();
}

template <> template <>
inline void HubbardInteraction<true>::apply_vertex_on_the_right (const Vertex &v, Eigen::MatrixXd &M) {
	cached_vec.noalias() = M.block(0, 0, V, V) * eigenvectors_.block(0, 0, V, V).row(v.x).transpose();
	M.block(0, 0, V, V).noalias() += (a+v.sigma) * cached_vec * eigenvectors_.block(0, 0, V, V).row(v.x);
	cached_vec.noalias() = M.block(V, V, V, V) * eigenvectors_.block(V, V, V, V).row(v.x).transpose();
	M.block(V, V, V, V).noalias() += (a-v.sigma) * cached_vec * eigenvectors_.block(V, V, V, V).row(v.x);
}

template <> template <>
inline void HubbardInteraction<true>::apply_inverse_on_the_left (const Vertex &v, Eigen::MatrixXd &M) {
	cached_vec.noalias() = M.block(0, 0, V, V).transpose() * eigenvectors_.block(0, 0, V, V).row(v.x).transpose();
	M.block(0, 0, V, V).noalias() -= (a+v.sigma)/(1.0+a+v.sigma) * eigenvectors_.block(0, 0, V, V).row(v.x).transpose() * cached_vec.transpose();
	cached_vec.noalias() = M.block(V, V, V, V).transpose() * eigenvectors_.block(V, V, V, V).row(v.x).transpose();
	M.block(V, V, V, V).noalias() -= (a-v.sigma)/(1.0+a-v.sigma) * eigenvectors_.block(V, V, V, V).row(v.x).transpose() * cached_vec.transpose();
}

template <> template <>
inline void HubbardInteraction<true>::apply_inverse_on_the_right (const Vertex &v, Eigen::MatrixXd &M) {
	cached_vec.noalias() = M.block(0, 0, V, V) * eigenvectors_.block(0, 0, V, V).row(v.x).transpose();
	M.block(0, 0, V, V).noalias() -= (a+v.sigma)/(1.0+a+v.sigma) * cached_vec * eigenvectors_.block(0, 0, V, V).row(v.x);
	cached_vec.noalias() = M.block(V, V, V, V) * eigenvectors_.block(V, V, V, V).row(v.x).transpose();
	M.block(V, V, V, V).noalias() -= (a-v.sigma)/(1.0+a-v.sigma) * cached_vec * eigenvectors_.block(V, V, V, V).row(v.x);
}

template <> template <>
inline void HubbardInteraction<true>::apply_vertex_on_the_left (const Vertex &v, HubbardInteraction<true>::MatrixType &M) {
	double C = eigenvectors_.block(0, 0, V, V).row(v.x) * M.col(0).head(V);
	M.col(0).head(V).noalias() += (a+v.sigma) * eigenvectors_.block(0, 0, V, V).row(v.x).transpose() * C;
	double D = eigenvectors_.block(V, V, V, V).row(v.x) * M.col(1).tail(V);
	M.col(1).tail(V).noalias() += (a-v.sigma) * eigenvectors_.block(V, V, V, V).row(v.x).transpose() * D;
}

template <> template <>
inline void HubbardInteraction<true>::apply_inverse_on_the_left (const Vertex &v, HubbardInteraction<true>::MatrixType &M) {
	double C = eigenvectors_.block(0, 0, V, V).row(v.x) * M.col(0).head(V);
	M.col(0).head(V).noalias() -= (a+v.sigma)/(1.0+a+v.sigma) * eigenvectors_.block(0, 0, V, V).row(v.x).transpose() * C;
	double D = eigenvectors_.block(V, V, V, V).row(v.x) * M.col(1).tail(V);
	M.col(1).tail(V).noalias() -= (a-v.sigma)/(1.0+a-v.sigma) * eigenvectors_.block(V, V, V, V).row(v.x).transpose() * D;
}

template <> template <>
inline void HubbardInteraction<true>::apply_displaced_vertex_on_the_left (const Vertex &v, Eigen::MatrixXd &M) {
	cached_vec.noalias() = M.block(0, 0, V, V).transpose() * v.data.V.block(0, 0, V, 1);
	M.block(0, 0, V, V).noalias() += v.data.mat[0] * v.data.U.block(0, 0, V, 1) * cached_vec.transpose();
	cached_vec.noalias() = M.block(V, V, V, V).transpose() * v.data.V.block(V, 1, V, 1);
	M.block(V, V, V, V).noalias() += v.data.mat[1] * v.data.U.block(V, 1, V, 1) * cached_vec.transpose();
}

template <> template <>
inline void HubbardInteraction<true>::apply_displaced_vertex_on_the_right (const Vertex &v, Eigen::MatrixXd &M) {
	cached_vec.noalias() = M.block(0, 0, V, V) * v.data.U.block(0, 0, V, 1);
	M.block(0, 0, V, V).noalias() += v.data.mat[0] * cached_vec * v.data.V.block(0, 0, V, 1).transpose();
	cached_vec.noalias() = M.block(V, V, V, V) * v.data.U.block(V, 1, V, 1);
	M.block(V, V, V, V).noalias() += v.data.mat[1] * cached_vec * v.data.V.block(V, 1, V, 1).transpose();
}

template <> template <>
inline void HubbardInteraction<true>::apply_displaced_vertex_on_the_left (const Vertex &v, HubbardInteraction<true>::MatrixType &M) {
	double C = ( v.data.V.block(0, 0, V, 1).transpose() * M.col(0).head(V) )(0, 0);
	M.col(0).head(V).noalias() += v.data.mat[0] * v.data.U.block(0, 0, V, 1) * C;
	double D = ( v.data.V.block(V, 1, V, 1).transpose() * M.col(1).tail(V) )(0, 0);
	M.col(1).tail(V).noalias() += v.data.mat[1] * v.data.U.block(V, 1, V, 1) * D;
}

template <> template <>
inline void HubbardInteraction<true>::apply_displaced_inverse_on_the_left (const Vertex &v, HubbardInteraction<true>::MatrixType &M) {
	double C = ( v.data.V.block(0, 0, V, 1).transpose() * M.col(0).head(V) )(0, 0);
	M.col(0).head(V).noalias() -= v.data.inv[0] * v.data.U.block(0, 0, V, 1) * C;
	double D = ( v.data.V.block(V, 1, V, 1).transpose() * M.col(1).tail(V) )(0, 0);
	M.col(1).tail(V).noalias() -= v.data.inv[1] * v.data.U.block(V, 1, V, 1) * D;
}

template <> template <>
inline void HubbardInteraction<true>::apply_displaced_vertex_on_the_right (const Vertex &v, Eigen::Matrix<double, 2, Eigen::Dynamic> &M) {
	double C = ( M.row(0).head(V) * v.data.U.block(0, 0, V, 1) )(0, 0);
	M.row(0).head(V).noalias() += v.data.mat[0] * v.data.V.block(0, 0, V, 1).transpose() * C;
	double D = ( M.row(1).tail(V) * v.data.U.block(V, 1, V, 1))(0, 0);
	M.row(1).tail(V).noalias() += v.data.mat[1] * v.data.V.block(V, 1, V, 1).transpose() * D;
}

template <> template <>
inline void HubbardInteraction<true>::apply_displaced_inverse_on_the_right (const Vertex &v, Eigen::Matrix<double, 2, Eigen::Dynamic> &M) {
	double C = ( M.row(0).head(V) * v.data.U.block(0, 0, V, 1) )(0, 0);
	M.row(0).head(V).noalias() -= v.data.inv[0] * v.data.V.block(0, 0, V, 1).transpose() * C;
	double D = ( M.row(1).tail(V) * v.data.U.block(V, 1, V, 1))(0, 0);
	M.row(1).tail(V).noalias() -= v.data.inv[1] * v.data.V.block(V, 1, V, 1).transpose() * D;
}


#endif // HUBBARD_HPP

