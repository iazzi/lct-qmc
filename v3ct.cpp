#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

#include "measurements.hpp"
#include "logger.hpp"
#include "svd.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <set>
#include <iterator>

#include <csignal>

//#define fftw_execute (void)

using SVDMatrix = SVDHelper;

Logger debug(std::cerr);

struct Vertex {
	double tau;
	size_t x;
	double sigma;
	class Compare {
		public:
			constexpr bool operator() (const Vertex& a, const Vertex& b) {
				return (a.tau<b.tau) || (a.tau==b.tau && a.x<b.x) || (a.tau==b.tau && a.x==b.x && a.sigma<b.sigma);
			}
	};
	Vertex (double a, size_t b, double c) : tau(a), x(b), sigma(c) {}
	constexpr Vertex (): tau(0.0), x(0), sigma(0.0) {}
	constexpr Vertex (double t): tau(t), x(0), sigma(0.0) {}
	constexpr bool is_valid () const { return sigma!=0.0; }
};

class VertexFactory {
	std::mt19937_64 &generator;

	// RNG distributions
	std::bernoulli_distribution coin_flip;
	std::uniform_int_distribution<size_t> randomPosition;
	std::uniform_real_distribution<double> randomTime;

	public:
	void setBeta (double b) {
		randomTime = std::uniform_real_distribution<double>(0.0, b);
	}

	void setVolume (size_t v) {
		randomPosition = std::uniform_int_distribution<size_t>(0, v-1);
	}

	VertexFactory (std::mt19937_64& g): generator(g) {
		coin_flip = std::bernoulli_distribution(0.5);
		randomPosition = std::uniform_int_distribution<size_t>(0, 0);
		randomTime = std::uniform_real_distribution<double>(0.0, 1.0);
	}

	Vertex generate (double a = 1.0) {
		return Vertex(randomTime(generator), randomPosition(generator), coin_flip(generator)?a:-a);
	}
};

class V3Configuration {
	std::set<Vertex, Vertex::Compare> verts;

	Eigen::MatrixXd eigenvectors;
	Eigen::VectorXd eigenvalues;

	size_t V;
	double beta, mu;

	std::vector<Eigen::MatrixXd> slices_up;
	std::vector<Eigen::MatrixXd> slices_dn;
	std::vector<int> damage;

	public:
	void setBeta (double b) {
		beta = b;
	}

	void setMu (double m) {
		mu = m;
	}

	template <typename M>
		void setEigenvectors (const Eigen::MatrixBase<M> &U) {
			eigenvectors = U;
			V = eigenvectors.rows();
		}

	template <typename M>
		void setEigenvalues (const Eigen::MatrixBase<M> &E) {
			eigenvalues = E;
			V = E.size();
		}

	size_t volume () const { return V; }

	void insertVertex (const Vertex& v) { verts.insert(v); }

	const Eigen::VectorXd eigenValues () const { return eigenvalues; }
	const Eigen::MatrixXd eigenVectors () const { return eigenvectors; }

	void computeUpdateVectors (Eigen::VectorXd &u, Eigen::VectorXd &v, const Vertex& w, double s) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		double t0 = beta/n*index, t1 = beta/n*(index+1);
		auto first = verts.lower_bound(Vertex(t0, 0, 0));
		auto last = verts.lower_bound(Vertex(t1, 0, 0));
		auto now = verts.lower_bound(Vertex(w.tau, 0, 0));
		u = eigenvectors.row(w.x).transpose();
		v = eigenvectors.row(w.x).transpose();
		Eigen::VectorXd cache;

		double t = w.tau;
		for (auto i=now;i!=last;) {
			if (i->tau==w.tau) {
				i++;
				continue;
			}
			if (i->tau>t) {
				u.array() *= (-(i->tau-t)*eigenvalues.array()).exp();
				t = i->tau;
			}
			auto j = i;
			while (++j!=last && j->tau==t) {}
			if (std::distance(i, j)==1) {
				u += s * i->sigma * eigenvectors.row(i->x).transpose() * (eigenvectors.row(i->x) * u);
			} else {
				cache.setZero(V);
				for (auto k=i;k!=j;k++) {
					cache += s * k->sigma * eigenvectors.row(k->x).transpose() * (eigenvectors.row(k->x) * u);
				}
				u += cache;
			}
			i = j;
		}
		if (t1>t) {
			u.array() *= (-(t1-t)*eigenvalues.array()).exp();
		}

		t = w.tau;
		for (auto i=now;first!=verts.end();) {
			if (i!=verts.end() && i->tau<t) {
				v.array() *= (-(t-i->tau)*eigenvalues.array()).exp();
				t = i->tau;
			}
			auto j = i;
			while (j!=first && std::prev(j)->tau==t) { j--; }
			if (std::distance(j, i)==0 && t<w.tau) {
				v += s * i->sigma * eigenvectors.row(i->x).transpose() * (eigenvectors.row(i->x) * v);
			} else if (t!=w.tau) {
				cache.setZero(V);
				for (auto k=i;std::prev(k)!=j;k--) {
					cache += s * k->sigma * eigenvectors.row(k->x).transpose() * (eigenvectors.row(k->x) * v);
				}
				v += cache;
			}
			if (j==first) {
				break;
			} else {
				i = std::prev(j);
			}
		}
		if (t0<t) {
			v.array() *= (-(t-t0)*eigenvalues.array()).exp();
		}

		u *= s * w.sigma;
	}

	void addRank1Vertex (const Vertex& w) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		Eigen::VectorXd u_up, v_up;
		Eigen::VectorXd u_dn, v_dn;
		computeUpdateVectors(u_up, v_up, w, 1.0);
		computeUpdateVectors(u_dn, v_dn, w, -1.0);

		insertVertex(w);

		double t0 = beta/n*index, t1 = beta/n*(index+1);
		Eigen::MatrixXd G = Eigen::MatrixXd::Identity(V, V);
		Eigen::MatrixXd F = Eigen::MatrixXd::Identity(V, V);
		compute_slice(G, t0, t1, +1.0);
		compute_slice_inverse(F, t0, t1, +1.0);
		if ((F-G.inverse()).norm()>1.0e-10) {
			std::cerr << damage[index] << " ==> " << (F-G.inverse()).norm() << std::endl << std::endl;
			std::cerr << F << std::endl << std::endl;
			std::cerr << G.inverse() << std::endl << std::endl;
			std::cerr << (F-G.inverse())*1.0e10 << std::endl << std::endl;
			std::cerr << (F*G-Eigen::MatrixXd::Identity(V, V)).norm() << std::endl;
			std::cerr << (G.inverse()*G-Eigen::MatrixXd::Identity(V, V)).norm() << std::endl << std::endl;
			throw -1;
		}
		if ((slices_up[index]+u_up * v_up.transpose()-G).norm()>1e-10) {
			std::cerr << u_up.transpose() << std::endl;
			std::cerr << v_up.transpose() << std::endl << std::endl;
			std::cerr << (G-slices_up[index]) << std::endl << std::endl;
			std::cerr << u_up.array().inverse().matrix().asDiagonal()*(G-slices_up[index]) << std::endl;
			std::cerr << std::endl;
			throw -1;
		}

		slices_up[index] += u_up * v_up.transpose();
		slices_dn[index] += u_dn * v_dn.transpose();
		damage[index]++;
	}

	void computeReversedVector (Eigen::VectorXd &v, const Vertex& w, double s) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		double t0 = beta/n*index, t1 = beta/n*(index+1);
		auto first = verts.lower_bound(Vertex(t0, 0, 0));
		auto last = verts.lower_bound(Vertex(t1, 0, 0));
		auto now = verts.lower_bound(Vertex(w.tau, 0, 0));
		v = eigenvectors.row(w.x).transpose();
		Eigen::VectorXd cache;

		double t = w.tau;
		for (auto i=now;i!=last;) {
			if (i->tau==w.tau) {
				i++;
				continue;
			}
			if (i->tau>t) {
				v.array() *= (+(i->tau-t)*eigenvalues.array()).exp();
				t = i->tau;
			}
			auto j = i;
			while (++j!=last && j->tau==t) {}
			if (std::distance(i, j)==1) {
				v -= s * i->sigma / (1.0+s*i->sigma) * eigenvectors.row(i->x).transpose() * (eigenvectors.row(i->x) * v);
			} else {
				cache.setZero(V);
				for (auto k=i;k!=j;k++) {
					cache -= s * k->sigma / (1.0+s*k->sigma) * eigenvectors.row(k->x).transpose() * (eigenvectors.row(k->x) * v);
				}
				v += cache;
			}
			i = j;
		}
		if (t1>t) {
			v.array() *= (+(t1-t)*eigenvalues.array()).exp();
		}
		Eigen::VectorXd u, r, z;
		computeUpdateVectors(u, r, w, +1.0);
		computeUpdateVectors(u, z, w, -1.0);
		std::cerr << "test " << index << " (" << w.tau << ", " << w.x << ", " << w.sigma << ") = " << (v-slices_up[index].inverse().transpose()*r).norm();
		std::cerr << "; " << (v-slices_dn[index].inverse().transpose()*z).norm();
		std::cerr << "; " << (slices_up[index]-slices_dn[index]).norm() << std::endl;
	}

	double testRank1Update (const Vertex &w) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		Eigen::VectorXd u_up, u_dn;
		Eigen::VectorXd v_up, v_dn;
		computeUpdateVectors(u_up, v_up, w, +1.0);
		computeUpdateVectors(u_dn, v_dn, w, -1.0);
		computeReversedVector(v_up, w, +1.0);
		computeReversedVector(v_dn, w, -1.0);
		SVDMatrix svd_up, svd_dn;
		svd_up.setIdentity(V);
		svd_dn.setIdentity(V);
		size_t m = index+1;
		for (size_t t=0;t<n;t++) {
			svd_up.U.applyOnTheLeft(slices_up[(t+m)%n]);
			svd_up.absorbU();
			svd_dn.U.applyOnTheLeft(slices_dn[(t+m)%n]);
			svd_dn.absorbU();
		}
		svd_up.invertInPlace();
		svd_dn.invertInPlace();
		svd_up.add_identity(exp(-beta*mu));
		svd_dn.add_identity(exp(-beta*mu));
		return (1.0 + v_up.transpose() * svd_up.inverse() * u_up)
			* (1.0 + v_dn.transpose() * svd_dn.inverse() * u_dn);
	}

	double testRank2Update (const Vertex &w1, const Vertex &w2) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w1.tau/dtau);
		Eigen::MatrixXd U_up, U_dn;
		Eigen::MatrixXd V_up, V_dn;
		U_up.resize(V, 2);
		U_dn.resize(V, 2);
		V_up.resize(V, 2);
		V_dn.resize(V, 2);
		Eigen::VectorXd u_up, u_dn;
		Eigen::VectorXd v_up, v_dn;
		computeUpdateVectors(u_up, v_up, w1, +1.0);
		computeUpdateVectors(u_dn, v_dn, w1, -1.0);
		//computeReversedVector(v_up, w1, +1.0);
		//computeReversedVector(v_dn, w1, -1.0);
		U_up.col(0) = u_up;
		U_dn.col(0) = u_dn;
		V_up.col(0) = v_up;
		V_dn.col(0) = v_dn;
		SVDMatrix svd_up, svd_dn;
		svd_up.setIdentity(V);
		svd_dn.setIdentity(V);
		size_t m = index+1;
		for (size_t t=0;t<n;t++) {
			svd_up.U.applyOnTheLeft(slices_up[(t+m)%n]);
			svd_up.absorbU();
			svd_dn.U.applyOnTheLeft(slices_dn[(t+m)%n]);
			svd_dn.absorbU();
		}
		svd_up.invertInPlace();
		svd_dn.invertInPlace();
		svd_up.add_identity(exp(-beta*mu));
		svd_dn.add_identity(exp(-beta*mu));
		Eigen::MatrixXd update_matrix_up = slices_up[index].inverse() * svd_up.inverse();
		Eigen::MatrixXd update_matrix_dn = slices_dn[index].inverse() * svd_dn.inverse();
		addVertex(w1);
		computeUpdateVectors(u_up, v_up, w2, +1.0);
		computeUpdateVectors(u_dn, v_dn, w2, -1.0);
		//computeReversedVector(v_up, w2, +1.0);
		//computeReversedVector(v_dn, w2, -1.0);
		U_up.col(1) = u_up;
		U_dn.col(1) = u_dn;
		V_up.col(1) = v_up;
		V_dn.col(1) = v_dn;
		return ( (Eigen::MatrixXd::Identity(2, 2) + V_up.transpose() * update_matrix_up * U_up).determinant()
			* (Eigen::MatrixXd::Identity(2, 2) + V_dn.transpose() * update_matrix_dn * U_dn).determinant() );
	}

	void addVertex (const Vertex& w, int threshold = 10) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		if (damage[index]<threshold) {
			addRank1Vertex(w);
		} else {
			insertVertex(w);
			reset_slice(index);
		}
	}

	void compute_slice (Eigen::MatrixXd &G, double a, double b, double s) {
		auto first = verts.lower_bound(Vertex(a, 0, 0));
		auto last = verts.lower_bound(Vertex(b, 0, 0));
		double t = a;
		Eigen::MatrixXd cache;
		for (auto v=first;v!=last;) {
			if (v->tau>t) {
				G.array().colwise() *= (-(v->tau-t)*eigenvalues.array()).exp();
				t = v->tau;
			}
			auto w = v;
			while (++w!=last && w->tau==t) {}
			if (std::distance(v, w)==1) {
				G += s * v->sigma * eigenvectors.row(v->x).transpose() * (eigenvectors.row(v->x) * G);
			} else {
				cache.setZero(V, V);
				for (auto u=v;u!=w;u++) {
					cache += s * u->sigma * eigenvectors.row(u->x).transpose() * (eigenvectors.row(u->x) * G);
				}
				G += cache;
			}
			v = w;
			//std::cerr << "vertex!" << std::endl;
		}
		if (b>t) {
			G.array().colwise() *= (-(b-t)*eigenvalues.array()).exp();
		}
		//std::cerr << (-(b-t)*eigenvalues.array()).exp().transpose() << std::endl << std::endl;
		//std::cerr << G << std::endl << std::endl;
	}

	void compute_slice_inverse (Eigen::MatrixXd &G, double a, double b, double s) {
		auto first = verts.lower_bound(Vertex(a, 0, 0));
		auto last = verts.lower_bound(Vertex(b, 0, 0));
		double t = a;
		Eigen::MatrixXd cache;
		for (auto v=first;v!=last;) {
			if (v->tau>t) {
				G.array().rowwise() *= (+(v->tau-t)*eigenvalues.array()).exp().transpose();
				t = v->tau;
			}
			auto w = v;
			while (++w!=last && w->tau==t) {}
			if (std::distance(v, w)==1) {
				G -= s * v->sigma / (1.0+s*v->sigma) * (G * eigenvectors.row(v->x).transpose()) * eigenvectors.row(v->x);
			} else {
				cache.setZero(V, V);
				for (auto u=v;u!=w;u++) {
					cache -= s * u->sigma / (1.0+s*u->sigma) * eigenvectors.row(u->x).transpose() * (eigenvectors.row(u->x) * G);
				}
				G += cache;
			}
			v = w;
			//std::cerr << "vertex!" << std::endl;
		}
		if (b>t) {
			G.array().rowwise() *= (+(b-t)*eigenvalues.array()).exp().transpose();
		}
		//std::cerr << (-(b-t)*eigenvalues.array()).exp().transpose() << std::endl << std::endl;
		//std::cerr << G << std::endl << std::endl;
	}

	size_t sliceSize (size_t i) const {
		size_t n = slices_up.size();
		auto first = verts.lower_bound(Vertex(beta/n*i, 0, 0));
		auto last = verts.lower_bound(Vertex(beta/n*(i+1), 0, 0));
		return std::distance(first, last);
	}

	void reset_slice (size_t index) {
		size_t n = slices_up.size();
		slices_up[index].setIdentity(V, V);
		slices_dn[index].setIdentity(V, V);
		compute_slice(slices_up[index], beta/n*index, beta/n*(index+1), +1.0);
		compute_slice(slices_dn[index], beta/n*index, beta/n*(index+1), -1.0);
		damage[index] = 0;
	}

	void recheck_slice (size_t index) {
		if (index<0 || index>=damage.size() || damage[index]==0) return;
		int d = damage[index];
		Eigen::MatrixXd A = slices_up[index];
		Eigen::MatrixXd B = slices_dn[index];
		reset_slice(index);
		std::cerr << index << " (damage=" << d << ") " <<(A - slices_up[index]).norm() << ' ' << (B - slices_dn[index]).norm() << std::endl;
	}

	void make_slices (size_t n) {
		slices_up.resize(n);
		slices_dn.resize(n);
		damage.resize(n);
		for (size_t i=0;i<n;i++) {
			reset_slice(i);
		}
	}

	std::pair<double, double> probability (size_t m) {
		size_t n = damage.size();
		SVDMatrix svd_up, svd_dn;
		svd_up.setIdentity(V);
		svd_dn.setIdentity(V);
		for (size_t t=0;t<n;t++) {
			svd_up.U.applyOnTheLeft(slices_up[(t+m)%n]);
			svd_up.absorbU();
			svd_dn.U.applyOnTheLeft(slices_dn[(t+m)%n]);
			svd_dn.absorbU();
		}
		svd_up.add_identity(exp(beta*mu));
		svd_dn.add_identity(exp(beta*mu));
		std::pair<double, double> ret;
		ret.first = svd_up.S.array().log().sum() + svd_dn.S.array().log().sum();
		ret.second = (svd_up.U*svd_up.Vt*svd_dn.U*svd_dn.Vt).determinant()>0.0?1.0:-1.0;
		return ret;
	}

	std::pair<double, double> probability_from_scratch (size_t n) {
		n = n==0?slices_up.size():n;
		make_slices(n);
		return probability(0);
	}

	size_t sliceNumber () const {
		return slices_up.size();
	}

	const Eigen::MatrixXd& slice_up (size_t i) const {
		return slices_up[i];
	}

	const Eigen::MatrixXd& slice_dn (size_t i) const {
		return slices_dn[i];
	}

	double inverseTemperature () const { return beta; }
	double chemicalPotential () const { return mu; }
	size_t verticesNumber () const { return verts.size(); }

	void removeVertex (size_t slice, size_t index) {
		size_t n = slices_up.size();
		auto first = verts.lower_bound(Vertex(beta/n*slice, 0, 0));
		auto last = verts.lower_bound(Vertex(beta/n*(slice+1), 0, 0));
		for (auto i=first;i!=last;i++) {
			if (index==0) {
				Eigen::VectorXd u_up, v_up;
				Eigen::VectorXd u_dn, v_dn;
				computeUpdateVectors(u_up, v_up, *i, 1.0);
				computeUpdateVectors(u_dn, v_dn, *i, -1.0);
				slices_up[slice] -= u_up*v_up.transpose();
				slices_dn[slice] -= u_dn*v_dn.transpose();
				verts.erase(i);
				break;
			}
			index--;
		}
	}

	Vertex pickVertex (size_t slice, size_t index) const {
		size_t n = slices_up.size();
		auto first = verts.lower_bound(Vertex(beta/n*slice, 0, 0));
		auto last = verts.lower_bound(Vertex(beta/n*(slice+1), 0, 0));
		for (auto i=first;i!=last;i++) {
			if (index==0) return *i;
			index--;
		}
	}
};


using namespace std;
using namespace std::chrono;

typedef std::chrono::duration<double> seconds_type;

std::string measurement_ratio (const measurement<double, false>& x, const measurement<double, false>& y, const char *s) {
	double a, b;
	a = x.mean()/y.mean();
	b = fabs(a)*(fabs(x.error()/x.mean())+fabs(y.error()/y.mean()));
	std::ostringstream buf;
	buf << a << s << b;
	return buf.str();
}

class SquareLattice {
	size_t Lx, Ly, Lz;
	size_t V;
	double tx, ty, tz;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;

	bool computed;

	public:

	void setSize (size_t a, size_t b, size_t c) {
		Lx = a;
		Ly = b;
		Lz = c;
		V = a*b*c;
		computed = false;
	}

	void setTunnelling (double a, double b, double c) {
		tx = a;
		ty = b;
		tz = c;
		computed = false;
	}

	void compute () {
		if (computed) return;
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(V, V);
		for (size_t x=0;x<Lx;x++) {
			for (size_t y=0;y<Ly;y++) {
				for (size_t z=0;z<Lz;z++) {
					size_t a = x*Ly*Lz + y*Lz + z;
					size_t b = ((x+1)%Lx)*Ly*Lz + y*Lz + z;
					size_t c = x*Ly*Lz + ((y+1)%Ly)*Lz + z;
					size_t d = x*Ly*Lz + y*Lz + (z+1)%Lz;
					if (Lx>1 && x!=Lx-0) H(a, b) = H(b, a) = -tx;
					if (Ly>1 && y!=Ly-0) H(a, c) = H(c, a) = -ty;
					if (Lz>1 && z!=Lz-0) H(a, d) = H(d, a) = -tz;
				}
			}
		}
		solver.compute(H);
		if (solver.info()==Eigen::Success) computed = true;
	}

	const typename Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>::RealVectorType & eigenvalues () const { return solver.eigenvalues(); }
	const typename Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>::MatrixType & eigenvectors () const { return solver.eigenvectors(); }

	SquareLattice (): Lx(2), Ly(2), Lz(1), V(4), tx(1.0), ty(1.0), tz(1.0), computed(false) {}
};


class V3Probability {
	private:
		SVDMatrix svd_up, svd_dn;
		SVDMatrix G_up, G_dn;
		Eigen::MatrixXd update_matrix_up, update_matrix_dn;
	public:
		void collectSlices (const V3Configuration &conf, size_t index) {
			size_t V = conf.volume();
			size_t n = conf.sliceNumber();
			size_t m = index+1;
			svd_up.setIdentity(V);
			svd_dn.setIdentity(V);
			for (size_t t=0;t<n;t++) {
				svd_up.U.applyOnTheLeft(conf.slice_up((t+m)%n));
				svd_up.absorbU();
				svd_dn.U.applyOnTheLeft(conf.slice_dn((t+m)%n));
				svd_dn.absorbU();
			}
		}

		void shiftLeft (const V3Configuration &conf, size_t index) {}
		void shiftRight (const V3Configuration &conf, size_t index) {}

		void makeGreenFunction (const V3Configuration &conf) {
			double beta = conf.inverseTemperature();
			double mu = conf.chemicalPotential();
			G_up = svd_up;
			G_dn = svd_dn;
			G_up.invertInPlace();
			G_dn.invertInPlace();
			G_up.add_identity(exp(-beta*mu));
			G_dn.add_identity(exp(-beta*mu));
			G_up.invertInPlace();
			G_dn.invertInPlace();
		}

		void prepareUpdateMatrices (const V3Configuration &conf, size_t index) {
			update_matrix_up = conf.slice_up(index).inverse() * G_up.matrix();
			update_matrix_dn = conf.slice_dn(index).inverse() * G_dn.matrix();
		}

		std::pair<double, double> probability (const V3Configuration &conf) {
			double beta = conf.inverseTemperature();
			double mu = conf.chemicalPotential();
			SVDMatrix A_up, A_dn;
			A_up = svd_up;
			A_dn = svd_dn;
			A_up.add_identity(exp(beta*mu));
			A_dn.add_identity(exp(beta*mu));
			std::pair<double, double> ret;
			ret.first = A_up.S.array().log().sum() + A_dn.S.array().log().sum();
			ret.second = (A_up.U*A_up.Vt*A_dn.U*A_dn.Vt).determinant()>0.0?1.0:-1.0;
			return ret;
		}

		const Eigen::MatrixXd& updateMatrixUp () const { return update_matrix_up; }
		const Eigen::MatrixXd& updateMatrixDn () const { return update_matrix_dn; }

		const Eigen::MatrixXd& greenFunctionUp () const { return G_up.matrix(); }
		const Eigen::MatrixXd& greenFunctionDn () const { return G_dn.matrix(); }
};

class V3Updater {
	private:
	std::mt19937_64 generator;
	std::bernoulli_distribution coin_flip;
	std::uniform_int_distribution<size_t> randomPosition;
	std::uniform_int_distribution<size_t> randomSlice;
	std::uniform_real_distribution<double> randomTime;
	std::uniform_real_distribution<double> random;
	std::exponential_distribution<double> trialDistribution;

	double K, U;
	double dtau;
	std::pair<double, double> p;
	std::pair<double, double> update_p;

	Eigen::MatrixXd U_up, U_dn;
	Eigen::MatrixXd V_up, V_dn;
	Eigen::VectorXd u_up, u_dn;
	Eigen::VectorXd v_up, v_dn;

	size_t updates;
	size_t slice;
	public:
	void setK (double k) { K = k; }
	void setU (double u) { U = u; }

	void setup (const V3Configuration &conf, V3Probability &prob) {
		size_t n = conf.sliceNumber();
		dtau = conf.inverseTemperature()/n;
		setBeta(conf.inverseTemperature()/n);
		setSliceNumber(n);
		setVolume(conf.volume());
		prepare(conf, prob, 0);
		p = prob.probability(conf);
	}

	void setBeta (double b) {
		randomTime = std::uniform_real_distribution<double>(0.0, b);
	}

	void setVolume (size_t v) {
		randomPosition = std::uniform_int_distribution<size_t>(0, v-1);
		U_up.resize(v, v);
		U_dn.resize(v, v);
		V_up.resize(v, v);
		V_dn.resize(v, v);
	}

	void setSliceNumber (size_t n) {
		randomPosition = std::uniform_int_distribution<size_t>(0, n-1);
	}

	V3Updater () : random(0.0, 1.0), trialDistribution(1.0) {
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
		coin_flip = std::bernoulli_distribution(0.5);
		randomPosition = std::uniform_int_distribution<size_t>(0, 0);
		randomTime = std::uniform_real_distribution<double>(0.0, 1.0);
	}

	Vertex generate () {
		return Vertex(randomTime(generator) + slice*dtau, randomPosition(generator), coin_flip(generator)?U/K:-U/K);
	}

	void prepare (const V3Configuration &conf, V3Probability &prob, size_t index) {
		prob.collectSlices(conf, index);
		prob.makeGreenFunction(conf);
		prob.prepareUpdateMatrices(conf, index);
		slice = index;
	}

	bool tryInsert (V3Configuration &conf, V3Probability &prob) {
		if (updates>=V_dn.cols()) {
			flush_updates(conf, prob);
		}
		Vertex v = generate();
		conf.computeUpdateVectors(u_up, v_up, v, +1.0);
		conf.computeUpdateVectors(u_dn, v_dn, v, -1.0);
		U_up.col(updates) = u_up;
		U_dn.col(updates) = u_dn;
		V_up.col(updates) = v_up;
		V_dn.col(updates) = v_dn;
		double d1, d2;
		d1 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_up.leftCols(updates+1).transpose() * prob.updateMatrixUp() * U_up.leftCols(updates+1)).determinant();
		d2 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_dn.leftCols(updates+1).transpose() * prob.updateMatrixDn() * U_dn.leftCols(updates+1)).determinant();
		double new_p = std::log(std::fabs(d1)) + std::log(std::fabs(d2));
		double new_s = d1*d2<0.0?-1.0:1.0;

		bool ret = -trialDistribution(generator)<new_p-update_p.first+log(conf.inverseTemperature())-log(conf.verticesNumber()+1)+log(K);
		std::cerr << new_p-update_p.first+log(conf.inverseTemperature())-log(conf.sliceSize(slice)+1)+log(K) << endl;
		if (ret) {
			conf.addVertex(v);
			update_p = std::pair<double, double>(new_p, new_s);
			updates++;
		} else {
		}
		return ret;
	}

	bool flush_updates (V3Configuration &conf, V3Probability &prob) {
		debug << "resetting updates" << updates;
		debug << p.first+update_p.first << p.second*update_p.second;
		prepare(conf, prob, slice);
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
		p = prob.probability(conf);
		debug << p.first << p.second;
	}

	double sign () const { return p.second*update_p.second; }

	bool tryRemove (V3Configuration &conf, V3Probability &prob) {
		if (updates>=V_dn.cols()) {
			flush_updates(conf, prob);
		}
		if (conf.sliceSize(slice)==0) return false;
		size_t vert_index = random(generator)*conf.sliceSize(slice);
		Vertex v = conf.pickVertex(slice, vert_index);
		conf.computeUpdateVectors(u_up, v_up, v, +1.0);
		conf.computeUpdateVectors(u_dn, v_dn, v, -1.0);
		U_up.col(updates) = -u_up;
		U_dn.col(updates) = -u_dn;
		V_up.col(updates) = v_up;
		V_dn.col(updates) = v_dn;
		double d1, d2;
		d1 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_up.leftCols(updates+1).transpose() * prob.updateMatrixUp() * U_up.leftCols(updates+1)).determinant();
		d2 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_dn.leftCols(updates+1).transpose() * prob.updateMatrixDn() * U_dn.leftCols(updates+1)).determinant();
		double new_p = std::log(std::fabs(d1)) + std::log(std::fabs(d2));
		double new_s = d1*d2<0.0?-1.0:1.0;

		bool ret = -trialDistribution(generator)<new_p-update_p.first-log(conf.inverseTemperature())+log(conf.verticesNumber()+1)-log(K);
		std::cerr << new_p-update_p.first-log(conf.inverseTemperature())+log(conf.verticesNumber()+1)-log(K) << endl;
		if (ret) {
			conf.removeVertex(slice, vert_index);
			update_p = std::pair<double, double>(new_p, new_s);
			updates++;
		} else {
		}
		return ret;
	}

	bool tryStep (V3Configuration &conf, V3Probability &prob) {
		if (random(generator)<0.005) {
			slice = conf.sliceNumber() * random(generator);
			debug << "jumping to slice" << slice;
			flush_updates(conf, prob);
		}
		if (coin_flip(generator)) {
			debug << "try insert";
			return tryInsert(conf, prob);
		} else {
			debug << "try remove";
			return tryRemove(conf, prob);
		}
	}
};

class V3Measurements {
	private:
		measurement<double> sign;
		measurement<double> order;
		measurement<double> density;
		measurement<double> magnetization;
		measurement<double> kinetic_energy;
		measurement<double> double_occupancy;
		measurement<Eigen::ArrayXd> density_distribution_up;
		measurement<Eigen::ArrayXd> density_distribution_dn;
	public:
		void measure (V3Configuration &conf, V3Probability &prob, V3Updater &updater) {
			double beta = conf.inverseTemperature();
			double mu = conf.chemicalPotential();
			double s = updater.sign();
			Eigen::MatrixXd rho_up = prob.greenFunctionUp();
			Eigen::MatrixXd rho_dn = prob.greenFunctionDn();
			double K = (rho_up.diagonal() + rho_dn.diagonal()).transpose() * conf.eigenValues();
			double n_up = rho_up.diagonal().array().sum();
			double n_dn = rho_dn.diagonal().array().sum();
			K -= (n_up+n_dn) * mu;
			rho_up = conf.eigenVectors() * rho_up * conf.eigenVectors().transpose();
			rho_dn = conf.eigenVectors() * rho_dn * conf.eigenVectors().transpose();
			double op = (rho_up.diagonal().array()-rho_dn.diagonal().array()).square().sum();
			double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
			// add to measurements
			sign.add(s);
			order.add(conf.verticesNumber());
			density.add(s*(n_up+n_dn)/conf.volume());
			magnetization.add(s*.0);
			kinetic_energy.add(s*K);
			double_occupancy.add(s*n2);
			density_distribution_up.add(s*Eigen::ArrayXd::Zero(conf.volume()));
			density_distribution_dn.add(s*Eigen::ArrayXd::Zero(conf.volume()));
		}
};

int main (int argc, char **argv) {
	Logger log(cout);

	double beta = 10.0, mu = 2.0;
	V3Configuration configuration;
	V3Probability prob;
	V3Updater updater;

	configuration.setBeta(beta);
	configuration.setMu(mu);

	updater.setU(2.0);
	updater.setK(4.0);

	SquareLattice lattice;
	lattice.setSize(4, 4, 1);
	lattice.compute();
	configuration.setEigenvectors(lattice.eigenvectors());
	configuration.setEigenvalues(lattice.eigenvalues());

	//std::mt19937_64 generator;
	//VertexFactory factory(generator);
	//factory.setVolume(configuration.volume());
	//factory.setBeta(beta);

	cerr << lattice.eigenvalues().transpose() << endl << endl << lattice.eigenvectors() << endl << endl;

	cerr << "base probability " << ((-beta*lattice.eigenvalues().array()+beta*mu).exp()+1.0).log().sum()*2.0 << endl;
	cerr << "computed probability " << configuration.probability_from_scratch(14).first << endl;

	prob.collectSlices(configuration, 0);
	prob.makeGreenFunction(configuration);
	prob.prepareUpdateMatrices(configuration, 0);
	cerr << "other probability " << prob.probability(configuration).first << endl;

	updater.setup(configuration, prob);

	for (int n=0;n<beta*configuration.volume()*500;n++) {
		cerr << configuration.verticesNumber() << " vertices" << endl;
		double p = configuration.probability(0).first;
		cerr << "vertex " << (updater.tryStep(configuration, prob)?"accepted":"rejected") << endl;
		//std::cerr << configuration.probability(0).first-p << std::endl;
		//for (int i=0;i<30;i+=5)
		//cerr << (i+1) << " svds probability " << configuration.probability_from_scratch(i+1).first << endl;
		cerr << endl;
	}
	for (int k=0;k<configuration.sliceNumber();k++) {
		configuration.recheck_slice(k);
	}
	for (int k=0;k<configuration.sliceNumber();k++) {
		debug << configuration.sliceSize(k);
	}

	return 0;
}

