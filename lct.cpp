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
#include <cstdlib>

#include "accumulator.hpp"
#include "hubbard.hpp"

//#define fftw_execute (void)

using SVDMatrix = SVDHelper;

Logger debug(std::cerr);

volatile int signalled = 0;

void my_signal_handler (int sig) {
	signalled = sig;
}

struct Vertex {
	double tau;
	size_t x;
	double sigma;
	class Compare {
		public:
			constexpr bool operator() (const Vertex& a, const Vertex& b) {
				return (a.tau<b.tau) || (a.tau==b.tau && a.x<b.x)
					|| (a.tau==b.tau && a.x==b.x && (std::fabs(a.sigma)<std::fabs(b.sigma)))
					|| (a.tau==b.tau && a.x==b.x && std::fabs(a.sigma)==std::fabs(b.sigma) && a.sigma<b.sigma);
			}
	};
	Vertex (double a, size_t b, double c) : tau(a), x(b), sigma(c) {}
	constexpr Vertex (): tau(0.0), x(0), sigma(0.0) {}
	constexpr Vertex (double t): tau(t), x(0), sigma(0.0) {}
	constexpr bool is_valid () const { return sigma!=0.0; }
};

std::ostream & operator<< (std::ostream &out, const Vertex& v) {
	out << "(" << v.tau << ", " << v.x << ", " << v.sigma << ")";
	return out;
}

class V3Slice {
	std::set<Vertex, Vertex::Compare> verts;

	Eigen::MatrixXd eigenvectors;
	Eigen::VectorXd eigenvalues;

	size_t N;
	double beta;

	Eigen::MatrixXd matrix_;
	Eigen::MatrixXd matrix_inv_;

	Eigen::VectorXd cache;

	public:

	void setup (double b, int v, const Eigen::MatrixXd& U, const Eigen::VectorXd& E) {
		beta = b;
		N = v;
		eigenvectors = U;
		eigenvalues = E;
       	}

	void insert (const Vertex &v) { verts.insert(v); }
	void clear () { verts.clear(); }

	Eigen::MatrixXd matrix () {
		matrix_.setIdentity(N, N);
		double t0 = 0.0;
		for (auto v : verts) {
			if (v.tau>t0) matrix_.array().colwise() *= (-(v.tau-t0)*eigenvalues.array()).exp();
			t0 = v.tau;
			matrix_ += v.sigma * eigenvectors.row(v.x).transpose() * (eigenvectors.row(v.x) * matrix_);
		}
		if (beta>t0) matrix_.array().colwise() *= (-(beta-t0)*eigenvalues.array()).exp();
		return matrix_;
	}

	Eigen::MatrixXd inverse () {
		matrix_inv_.setIdentity(N, N);
		double t0 = beta;
		for (auto v=verts.rbegin();v!=verts.rend();v++) {
			if (v->tau<t0) matrix_inv_.array().colwise() *= (-(v->tau-t0)*eigenvalues.array()).exp();
			t0 = v->tau;
			matrix_inv_ += -v->sigma/(1.0+v->sigma) * eigenvectors.row(v->x).transpose() * (eigenvectors.row(v->x) * matrix_inv_);
		}
		if (t0>0.0) matrix_inv_.array().colwise() *= (t0*eigenvalues.array()).exp();
		return matrix_inv_;
	}
};

class V3Configuration {
	std::set<Vertex, Vertex::Compare> verts;

	Eigen::MatrixXd eigenvectors;
	Eigen::VectorXd eigenvalues;

	size_t V;
	double beta, mu, B;

	std::vector<Eigen::MatrixXd> slices_up;
	std::vector<Eigen::MatrixXd> slices_dn;
	std::vector<int> damage;

	public:

	const std::set<Vertex, Vertex::Compare>& vertices () const { return verts; }
	std::set<Vertex, Vertex::Compare>& vertices () { return verts; }

	void setBeta (double b) { beta = b; verts.erase(verts.lower_bound(Vertex(b, 0, 0)), verts.end());}
	void setMu (double m) { mu = m; }
	void setB (double b) { B = b; }

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

	Eigen::VectorXd cache;

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
			//auto j = i;
			//while (++j!=last && j->tau==t) {}
			//if (std::distance(i, j)==1) {
				u += s * i->sigma * eigenvectors.row(i->x).transpose() * (eigenvectors.row(i->x) * u);
			//} else {
				//cache.setZero(V);
				//for (auto k=i;k!=j;k++) {
					//cache += s * k->sigma * eigenvectors.row(k->x).transpose() * (eigenvectors.row(k->x) * u);
				//}
				//u += cache;
			//}
			++i;
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

		//double t0 = beta/n*index, t1 = beta/n*(index+1);
		Eigen::MatrixXd G; // = Eigen::MatrixXd::Identity(V, V);
		Eigen::MatrixXd F; // = Eigen::MatrixXd::Identity(V, V);
		//compute_slice(G, t0, t1, +1.0);
		//compute_slice_inverse(F, t0, t1, +1.0);
		if (false && (F-G.inverse()).norm()>1.0e-10 && false) {
			std::cerr << damage[index] << " ==> " << (F-G.inverse()).norm() << std::endl << std::endl;
			std::cerr << F << std::endl << std::endl;
			std::cerr << G.inverse() << std::endl << std::endl;
			std::cerr << (F-G.inverse())*1.0e10 << std::endl << std::endl;
			std::cerr << (F*G-Eigen::MatrixXd::Identity(V, V)).norm() << std::endl;
			std::cerr << (G.inverse()*G-Eigen::MatrixXd::Identity(V, V)).norm() << std::endl << std::endl;
			throw -1;
		}
		if (false && (slices_up[index]+u_up * v_up.transpose()-G).norm()>1e-10) {
			std::cerr << "error updating slice\n";
			std::cerr << (slices_up[index]+u_up * v_up.transpose()-G).norm() << '\n';
			std::cerr << u_up.transpose() << std::endl;
			std::cerr << v_up.transpose() << std::endl << std::endl;
			std::cerr << (G-slices_up[index]) << std::endl << std::endl;
			std::cerr << u_up.array().inverse().matrix().asDiagonal()*(G-slices_up[index]) << std::endl;
			std::cerr << std::endl;
			//throw -1;
		}

		slices_up[index] += u_up * v_up.transpose();
		slices_dn[index] += u_dn * v_dn.transpose();
		damage[index]++;
	}

	void computeReversedVector (Eigen::VectorXd &v, const Vertex& w, double s) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		double t0 = beta/n*index;
		//double t1 = beta/n*(index+1);
		auto first = verts.lower_bound(Vertex(t0, 0, 0));
		//auto last = verts.lower_bound(Vertex(t1, 0, 0));
		auto now = verts.lower_bound(Vertex(w.tau, 0, 0));
		v = eigenvectors.row(w.x).transpose();
		Eigen::VectorXd cache;

		double t = w.tau;
		for (auto i=now;i!=first&&std::prev(i)!=first;) {
			i = std::prev(i);
			if (i->tau==w.tau) {
				continue;
			}
			if (i->tau<t) {
				v.array() *= (+(t-i->tau)*eigenvalues.array()).exp();
				t = i->tau;
			}
			v -= s * i->sigma / (1.0+s*i->sigma) * eigenvectors.row(i->x).transpose() * (eigenvectors.row(i->x) * v);
		}
		if (t0<t) {
			v.array() *= (+(t-t0)*eigenvalues.array()).exp();
		}
		v *= s * w.sigma;
		Eigen::VectorXd u, r, z;
		computeUpdateVectors(u, r, w, +1.0);
		//computeUpdateVectors(u, z, w, -1.0);
		std::cerr << "test " << index << " (" << w.tau << ", " << w.x << ", " << w.sigma << ") = " << (v-slices_up[index].inverse()*u).norm() << '\n';
		//std::cerr << v.transpose() << std::endl << (slices_up[index].inverse()*u).transpose() << std::endl;
		//std::cerr << u.transpose() << std::endl;
		//std::cerr << slices_up[index].diagonal().transpose() << std::endl;
		//std::cerr << slices_up[index].inverse().diagonal().transpose() << std::endl;
		//std::cerr << "; " << (v-slices_dn[index].inverse().transpose()*z).norm();
		//std::cerr << (+dtau*eigenvalues.array()).exp().transpose() << std::endl;
		//std::cerr << (+(t1-w.tau)*eigenvalues.array()).exp().transpose() << std::endl;
		//std::cerr << (+(t0-w.tau)*eigenvalues.array()).exp().transpose() << std::endl;
		//throw;
	}

	void addVertex (const Vertex& w, int threshold = 10) {
		size_t n = slices_up.size();
		double dtau = beta/n;
		size_t index = size_t(w.tau/dtau);
		//addRank1Vertex(w);
		insertVertex(w);
		return;
		if (damage[index]<threshold) {
			addRank1Vertex(w);
		} else {
			insertVertex(w);
			reset_slice(index);
		}
	}

	void compute_slice (Eigen::MatrixXd &G, double a, double b, double s) const {
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
	double magneticField () const { return B; }
	double mu_up () const { return mu+0.5*B; }
	double mu_dn () const { return mu-0.5*B; }
	size_t verticesNumber () const { return verts.size(); }

	std::set<Vertex>::iterator pickVertexIterator (size_t slice, size_t index) const {
		size_t n = slices_up.size();
		auto first = verts.lower_bound(Vertex(beta/n*slice, 0, 0));
		auto last = verts.lower_bound(Vertex(beta/n*(slice+1), 0, 0));
		for (auto i=first;i!=last;i++) {
			if (index==0) return i;
			index--;
		}
		return last;
	}

	void removeVertex (size_t slice, size_t index) {
		auto i = pickVertexIterator(slice, index);
		//Eigen::VectorXd u_up, v_up;
		//Eigen::VectorXd u_dn, v_dn;
		//computeUpdateVectors(u_up, v_up, *i, 1.0);
		//computeUpdateVectors(u_dn, v_dn, *i, -1.0);
		//slices_up[slice] -= u_up*v_up.transpose();
		//slices_dn[slice] -= u_dn*v_dn.transpose();
		verts.erase(i);
	}

	Vertex pickVertex (size_t slice, size_t index) const {
		return *pickVertexIterator(slice, index);
	}

	void show_verts () const {
		show_verts(std::cerr);
	}

	void show_verts (std::ostream& out) const {
		for (auto v : verts) {
			out << v << ' ';
		}
		out << std::endl;
	}

	void printout (const std::string &fn) const {
		std::ofstream f(fn);
		for (auto v : verts) {
			f << v << '\n';
		}
		f.close();
	}

	double logDet () const {
		double ret = 0.0;
		for (auto v : verts) {
			ret += std::log(1.0+v.sigma);
		}
		return ret;
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

	size_t volume () const { return V; }

	SquareLattice (): Lx(2), Ly(2), Lz(1), V(4), tx(1.0), ty(1.0), tz(1.0), computed(false) {}
};


class V3Probability {
	private:
		SVDMatrix svd_up, svd_dn;
		SVDMatrix A_up, A_dn;
		SVDMatrix G_up, G_dn;
		Eigen::MatrixXd update_matrix_up, update_matrix_dn;

		Accumulator acc_up, acc_dn;

		Eigen::ArrayXd Rd;
		Eigen::MatrixXd R, R_inverse;
	public:
		void prepare_random_matrix (const V3Configuration& conf) {
			Rd = 1.0*Eigen::ArrayXd::Random(conf.volume());
			Rd -= Rd.sum()/Rd.size();
			R = conf.eigenVectors().transpose() * Rd.exp().matrix().asDiagonal() * conf.eigenVectors();
			R_inverse = conf.eigenVectors().transpose() * (-Rd).exp().matrix().asDiagonal() * conf.eigenVectors();
		}

		void collectSlices (const V3Configuration &conf, size_t index) {
			size_t V = conf.volume();
			size_t n = conf.sliceNumber();
			size_t m = index;
			svd_up.setIdentity(V);
			svd_dn.setIdentity(V);
			for (size_t t=0;t<n;t++) {
				svd_up.U.applyOnTheLeft(conf.slice_up((t+m)%n));
				svd_up.absorbU();
				//svd_dn.U.applyOnTheLeft(conf.slice_dn((t+m)%n));
				//svd_dn.absorbU();
			}
			svd_dn = svd_up;
		}

		void shiftLeft (const V3Configuration &conf, size_t index) {}
		void shiftRight (const V3Configuration &conf, size_t index) {}

		void makeGreenFunction (const V3Configuration &conf) {
			double beta = conf.inverseTemperature();
			//double mu = conf.chemicalPotential();
			//double B = conf.magneticField();
			G_up = svd_up;
			G_dn = svd_dn;
			G_up.invertInPlace();
			G_dn.invertInPlace();
			G_up.add_identity(exp(-beta*conf.mu_up()));
			G_dn.add_identity(exp(-beta*conf.mu_dn()));
			G_up.invertInPlace();
			G_dn.invertInPlace();
		}

		void makeGreenFunction_alt (const V3Configuration &conf) {
			makeGreenFunction(conf);
			G_up.U.applyOnTheLeft(R);
			G_up.Vt.applyOnTheRight(R_inverse);
			G_dn.U.applyOnTheLeft(R);
			G_dn.Vt.applyOnTheRight(R_inverse);
		}

		void prepareUpdateMatrices (V3Configuration &conf, size_t index) {
			conf.reset_slice(index);
			update_matrix_up = G_up.matrix(); // * conf.slice_up(index).inverse();
			update_matrix_up.applyOnTheRight(conf.slice_up(index).inverse());
			update_matrix_dn = G_dn.matrix(); // * conf.slice_up(index).inverse();
			update_matrix_dn.applyOnTheRight(conf.slice_up(index).inverse());
		}

		std::pair<double, double> probability (const V3Configuration &conf) {
			double beta = conf.inverseTemperature();
			A_up = svd_up;
			A_dn = svd_dn;
			A_up.add_identity(exp(beta*conf.mu_up()));
			A_dn.add_identity(exp(beta*conf.mu_dn()));
			std::pair<double, double> ret;
			ret.first = A_up.S.array().log().sum() + A_dn.S.array().log().sum();
			ret.second = (A_up.U*A_up.Vt*A_dn.U*A_dn.Vt).determinant()>0.0?1.0:-1.0;
			if ((A_up.U*A_up.Vt).determinant()<0.0 || (A_dn.U*A_dn.Vt).determinant()<0.0) {
				debug << "WTF?" << ret.second;
				throw -1;
			}
			return ret;
		}

		void evolve (Accumulator &acc, const V3Configuration &conf, double t, double dtau) {
			while (t>0.0) {
				double step = std::min(t, dtau-acc.distance());
				acc.matrixU().array().colwise() *= (-step*conf.eigenValues().array()).exp();
				acc.increase_distance(step);
				t -= step;
				if (acc.distance()>=dtau) acc.decomposeU();
			}
		}

		void accumulate (Accumulator &acc, const V3Configuration &conf, double t0, double s, int nt = -1) {
			if (nt<0) nt = 1.5*conf.volume();
			//acc.start(R);
			//R = conf.eigenVectors().transpose() * Rd.exp().matrix().asDiagonal() * conf.eigenVectors();
			acc.reset(conf.volume());
			acc.SVD().U = conf.eigenVectors().transpose();
			acc.SVD().S = Rd.exp();
			acc.SVD().Vt = conf.eigenVectors();
			double dtau = 1.0/3.0;
			auto first = conf.vertices().lower_bound(Vertex(t0, 0, 0));
			auto last = conf.vertices().lower_bound(Vertex(conf.inverseTemperature(), 0, 0));
			int nv = 0;
			double t = t0;
			for (auto v=first;v!=last;v++) {
				evolve(acc, conf, v->tau-t, dtau);
				t = v->tau;
				acc.matrixU() += s * v->sigma * conf.eigenVectors().row(v->x).transpose() * (conf.eigenVectors().row(v->x) * acc.matrixU());
				acc.increase_logdet(std::log(std::fabs(1.0+s*v->sigma)));
				nv++;
				//if (std::fabs(acc.logdet())>15.0) {
				if (nv>nt) {
					acc.decomposeU();
					acc.assertLogDet();
					nv = 0;
				}
			}
			evolve(acc, conf, conf.inverseTemperature()-t, dtau);
			// wrap around
			last = first;
			first = conf.vertices().begin();
			t = 0;
			for (auto v=first;v!=last;v++) {
				evolve(acc, conf, v->tau-t, dtau);
				t = v->tau;
				acc.matrixU() += s * v->sigma * conf.eigenVectors().row(v->x).transpose() * (conf.eigenVectors().row(v->x) * acc.matrixU());
				acc.increase_logdet(std::log(std::fabs(1.0+s*v->sigma)));
				nv++;
				//if (std::fabs(acc.logdet())>15.0) {
				if (nv>nt) {
					acc.decomposeU();
					acc.assertLogDet();
					nv = 0;
				}
			}
			evolve(acc, conf, t0-t, dtau);
			acc.matrixU().applyOnTheLeft(R_inverse);
			acc.decomposeU();
			acc_up.assertLogDet();
			//std::cerr << (-(b-t)*eigenvalues.array()).exp().transpose() << std::endl << std::endl;
			//std::cerr << G << std::endl << std::endl;
		}

		void collect_alt (const V3Configuration &conf, size_t index) {
			//double beta = conf.inverseTemperature();
			//double mu = conf.chemicalPotential();
			double t0 = conf.inverseTemperature()/conf.sliceNumber()*index;
			if (R.rows()!=R.cols() || R.rows()!=int(conf.volume())) {
				prepare_random_matrix(conf);
			}
			accumulate(acc_up, conf, t0, +1.0);
			//accumulate(acc_dn, conf, t0, -1.0);
			//debug << (acc_up.SVD().S.array()*acc_dn.SVD().S.array().reverse()).transpose() << '\n';
			//try {
				acc_up.assertLogDet();
			//} catch (Accumulator::AssertionFailed ass) {
			//}
			//acc_dn.assertLogDet();
			svd_up = acc_up.SVD();
			svd_dn = acc_up.SVD();
		}

		void dumb_test (const V3Configuration &conf, size_t index = 0) {
			Eigen::MatrixXd A, B;
			size_t n = conf.sliceNumber();
			A = B = Eigen::MatrixXd::Identity(conf.volume(), conf.volume());
			for (size_t t=0;t<n;t++) {
				A.applyOnTheLeft(conf.slice_up((t+index)%n));
			}
			//auto first = conf.vertices().begin();
			//auto last = conf.vertices().end();
			const double t0 = conf.inverseTemperature()/n*index;
			double t = t0;
			for (auto v=conf.vertices().lower_bound(Vertex(t0, 0, 0));v!=conf.vertices().end();v++) {
				B.array().colwise() *= (-(v->tau-t)*conf.eigenValues().array()).exp();
				t = v->tau;
				B += v->sigma * conf.eigenVectors().row(v->x).transpose() * (conf.eigenVectors().row(v->x) * B);
			}
			B.array().colwise() *= (-(conf.inverseTemperature()-t)*conf.eigenValues().array()).exp();
			t = 0.0;
			for (auto v=conf.vertices().begin();v!=conf.vertices().lower_bound(Vertex(t0, 0, 0));v++) {
				B.array().colwise() *= (-(v->tau-t)*conf.eigenValues().array()).exp();
				t = v->tau;
				B += v->sigma * conf.eigenVectors().row(v->x).transpose() * (conf.eigenVectors().row(v->x) * B);
			}
			B.array().colwise() *= (-(t0-t)*conf.eigenValues().array()).exp();
			debug << "dumb=" << (A-B).norm() << '\n';
			//debug << A << '\n';
			//debug << B << '\n';
		}

		std::pair<double, double> probability_alt (const V3Configuration &conf) {
			double beta = conf.inverseTemperature();
			//double mu = conf.chemicalPotential();
			//collect_alt(conf, 0);
			A_up = svd_up;
			A_dn = svd_dn;
			A_up.add_identity(exp(beta*conf.mu_up()));
			A_dn.add_identity(exp(beta*conf.mu_dn()));
			std::pair<double, double> ret;
			ret.first = A_up.S.array().log().sum() + A_dn.S.array().log().sum();
			ret.second = (A_up.U*A_up.Vt*A_dn.U*A_dn.Vt).determinant()>0.0?1.0:-1.0;
			if ((A_up.U*A_up.Vt).determinant()<0.0 || (A_dn.U*A_dn.Vt).determinant()<0.0) {
				//debug << (A_up.U*A_up.Vt).determinant() << (A_dn.U*A_dn.Vt).determinant() << ret.second;
				//throw -1;
			}
			return ret;
		}

		const Eigen::MatrixXd& updateMatrixUp () const { return update_matrix_up; }
		const Eigen::MatrixXd& updateMatrixDn () const { return update_matrix_dn; }

		Eigen::MatrixXd greenFunctionUp () const { return G_up.matrix(); }
		Eigen::MatrixXd greenFunctionDn () const { return G_dn.matrix(); }
		Eigen::MatrixXd greenFunctionDn_flipped (const V3Configuration &conf) {
			double beta = conf.inverseTemperature();
			G_dn = svd_dn;
			G_dn.add_identity(exp(beta*conf.mu_dn()));
			G_dn.invertInPlace();
			G_dn.U.applyOnTheLeft(R);
			G_dn.Vt.applyOnTheRight(R_inverse);
			return G_dn.matrix();
		}

		Eigen::MatrixXd propagatorUp () const { return svd_up.matrix(); }
		Eigen::MatrixXd propagatorDn () const { return svd_dn.matrix(); }

		void single_vertex_test (const V3Configuration &conf) {
			Accumulator acc;
			acc.start(R);
			size_t N = 50;
			double dtau = conf.inverseTemperature()/N;
			for (size_t i=0;i<N;i++) {
				acc.matrixU().array().colwise() *= (-dtau*conf.eigenValues().array()).exp();
				acc.decomposeU();
			}
			//Vertex v = generate();
			//size_t x = v.x;
			//double sigma = v.sigma;
			//acc.matrixU().array().colwise() *= (-dtau*conf.eigenValues().array()).exp();
			//acc.matrixU() += sigma * conf.eigenVectors().row(x).transpose() * (conf.eigenVectors().row(x) * acc.matrixU());
			//acc.matrixU().array().colwise() *= (-dtau*conf.eigenValues().array()).exp();
			//acc.matrixU().applyOnTheLeft(R);
			//acc.decomposeU();
			//debug << acc.logdet() - std::log(std::fabs(1.0+sigma));
		}
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

	double U, K;
	double A, B;
	double dtau;
	std::pair<double, double> p;
	std::pair<double, double> update_p;

	Eigen::MatrixXd U_up, U_dn;
	Eigen::MatrixXd V_up, V_dn;
	Eigen::VectorXd u_up, u_dn;
	Eigen::VectorXd v_up, v_dn;

	std::ofstream dump;

	size_t updates;
	size_t slice;

	std::vector<Vertex> last_add;
	std::vector<Vertex> last_del;
	public:
	void setK (double k) { K = k; prepare_AB(); }
	void setU (double u) { U = u; prepare_AB(); }
	void prepare_AB () { A = 1.0*U/2.0/K; B = sqrt(U/K+A*A); debug << (A+B) << (A-B) << (1+A+B)*(1+A-B); }
	void setSeed (unsigned int s) { generator.seed(s); }

	void randomize (V3Configuration &conf, size_t N = 0) {
		for (size_t i=0;i<N;i++) conf.addVertex(generate());
	}

	void setup (V3Configuration &conf, V3Probability &prob) {
		size_t n = conf.sliceNumber();
		dtau = conf.inverseTemperature()/n;
		setBeta(conf.inverseTemperature()/n);
		setSliceNumber(n);
		setVolume(conf.volume());
		prepare(conf, prob, 0);
		prepare_alt(conf, prob, 0);
		p = prob.probability_alt(conf);
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
	}

	void set_dump (const char *fn) {
		dump.open(fn);
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

	V3Updater () : random(0.0, 1.0), trialDistribution(1.0), U(0.0), K(0.0), A(0.0), B(0.0) {
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
		coin_flip = std::bernoulli_distribution(0.5);
		randomPosition = std::uniform_int_distribution<size_t>(0, 0);
		randomTime = std::uniform_real_distribution<double>(0.0, 1.0);
	}

	Vertex generate () {
		return Vertex(randomTime(generator) + slice*dtau, randomPosition(generator), coin_flip(generator)?(A+B):(A-B));
	}

	void reprepare (V3Configuration &conf, V3Probability &prob) {
		p.first += update_p.first;
		p.second *= update_p.second;
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
		prepare_alt(conf, prob, conf.sliceNumber() * random(generator));
	}

	void prepare (V3Configuration &conf, V3Probability &prob) {
		prepare_alt(conf, prob, conf.sliceNumber() * random(generator));
	}

	void prepare (V3Configuration &conf, V3Probability &prob, size_t index) {
		prob.collectSlices(conf, index);
		prob.makeGreenFunction(conf);
		prob.prepareUpdateMatrices(conf, index);
		slice = index;
	}

	void prepare_alt (V3Configuration &conf, V3Probability &prob, size_t index) {
		//Eigen::MatrixXd J;
		//prepare(conf, prob, index);
		//J = prob.updateMatrixUp();
		prob.collect_alt(conf, index);
		prob.makeGreenFunction_alt(conf);
		prob.prepareUpdateMatrices(conf, index);
		slice = index;
		//if ((J-prob.updateMatrixUp()).norm()>1.0e-6) {
			//debug << J << '\n';
			//debug << prob.updateMatrixUp() << '\n';
			//debug << J.determinant() << prob.propagatorUp().determinant();
			//prob.dumb_test(conf, index);
			//debug << slice;
			//for (auto v : conf.vertices()) {
				//std::cerr << v << ' ';
			//}
			//std::cerr << std::endl;
			//throw;
		//}
	}

	bool tryInsert (V3Configuration &conf, V3Probability &prob) {
		if (int(updates)>=V_dn.cols()) {
			flush_updates(conf, prob);
		}
		Vertex v = generate();
		conf.computeUpdateVectors(u_up, v_up, v, +1.0);
		conf.computeUpdateVectors(u_dn, v_dn, v, +1.0);
		//debug << "ins";
		//conf.computeReversedVector(u_up, v, +1.0);
		//conf.computeReversedVector(u_dn, v, +1.0);
		U_up.col(updates) = u_up;
		U_dn.col(updates) = u_dn;
		V_up.col(updates) = v_up;
		V_dn.col(updates) = v_dn;
		double d1, d2;
		d1 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_up.leftCols(updates+1).transpose() * prob.updateMatrixUp() * U_up.leftCols(updates+1)).determinant();
		d2 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_dn.leftCols(updates+1).transpose() * prob.updateMatrixDn() * U_dn.leftCols(updates+1)).determinant();
		double new_p = std::log(std::fabs(d1)) + std::log(std::fabs(d2));
		double new_s = d1*d2<0.0?-1.0:1.0;

		bool ret = -trialDistribution(generator)<new_p-update_p.first+log(conf.inverseTemperature()/conf.sliceNumber())-log(conf.sliceSize(slice)+1)+std::log(K*conf.volume());
		//std::cerr << new_p-update_p.first+log(conf.inverseTemperature())-log(conf.sliceSize(slice)+1)+log(K) << endl;
		if (ret) {
			if (dump.is_open()) last_add.push_back(v);
			conf.addVertex(v);
			update_p = std::pair<double, double>(new_p, new_s);
			updates++;
			//flush_updates(conf, prob);
		} else {
		}
		return ret;
	}

	void reset_updates (V3Configuration &conf, V3Probability &prob) {
		prepare_alt(conf, prob, slice);
		p = prob.probability_alt(conf);
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
	}

	void flush_updates (V3Configuration &conf, V3Probability &prob) {
		//debug << "flushing";
		//debug << p.first << p.second;
		//debug << update_p.first << update_p.second;
		if (updates==0) return;
		double old_p = p.first+update_p.first;
		double old_s = p.second*update_p.second;
		prepare_alt(conf, prob, slice);
		p = prob.probability_alt(conf);
		if (fabs((old_p-p.first)/old_p)>1e-5 || old_s!=p.second) {
			debug << p.first << p.second;
			debug << old_p << old_s;
			debug << (conf.inverseTemperature()/conf.sliceNumber()) << (conf.inverseTemperature()/conf.sliceNumber()*slice) << updates;
			//debug << conf.sliceSize(slice) << last_add.size() << last_del.size();
			if (fabs(old_p-p.first)>1) {
				if (dump.is_open()) {
					conf.show_verts(dump);
					for (auto v : last_add) debug << '+' << v;
					for (auto v : last_del) debug << '-' << v;
					throw;
				}
			}
			std::cerr << std::endl;
			//throw;
		}
		updates = 0;
		update_p = std::pair<double, double>(0.0, 1.0);
		if (dump.is_open()) {
			last_add.clear();
			last_del.clear();
		}
	}

	//bool debug () const { return dump.is_open(); }

	double sign () const { return p.second*update_p.second; }

	bool tryRemove (V3Configuration &conf, V3Probability &prob) {
		if (int(updates)>=V_dn.cols()) {
			flush_updates(conf, prob);
		}
		if (conf.sliceSize(slice)==0) return false;
		size_t vert_index = random(generator)*conf.sliceSize(slice);
		Vertex v = conf.pickVertex(slice, vert_index);
		conf.computeUpdateVectors(u_up, v_up, v, +1.0);
		conf.computeUpdateVectors(u_dn, v_dn, v, +1.0);
		//debug << "rem";
		//conf.computeReversedVector(u_up, v, +1.0);
		//conf.computeReversedVector(u_dn, v, +1.0);
		U_up.col(updates) = -u_up;
		U_dn.col(updates) = -u_dn;
		V_up.col(updates) = v_up;
		V_dn.col(updates) = v_dn;
		double d1, d2;
		d1 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_up.leftCols(updates+1).transpose() * prob.updateMatrixUp() * U_up.leftCols(updates+1)).determinant();
		d2 = (Eigen::MatrixXd::Identity(updates+1, updates+1) + V_dn.leftCols(updates+1).transpose() * prob.updateMatrixDn() * U_dn.leftCols(updates+1)).determinant();
		double new_p = std::log(std::fabs(d1)) + std::log(std::fabs(d2));
		double new_s = d1*d2<0.0?-1.0:1.0;

		bool ret = -trialDistribution(generator)<new_p-update_p.first-log(conf.inverseTemperature()/conf.sliceNumber())+log(conf.sliceSize(slice))-std::log(K*conf.volume());
		//std::cerr << new_p-update_p.first-log(conf.inverseTemperature())+log(conf.verticesNumber()+1)-log(K) << endl;
		if (ret) {
			if (dump.is_open()) last_del.push_back(v);
			conf.removeVertex(slice, vert_index);
			update_p = std::pair<double, double>(new_p, new_s);
			updates++;
			//flush_updates(conf, prob);
		} else {
		}
		return ret;
	}

	bool tryStep (V3Configuration &conf, V3Probability &prob) {
		if (random(generator)<0.01) {
		}
		if (coin_flip(generator)) {
			//debug << "try insert";
			return tryInsert(conf, prob);
		} else {
			//debug << "try remove";
			return tryRemove(conf, prob);
		}
	}

	double sweep (V3Configuration &conf, V3Probability &prob) {
		double ret = 0.0;
		size_t N = conf.volume();
		for (size_t n=0;n<N;n++) {
			ret += tryStep(conf, prob)?1.0:0.0;
		}
		slice = conf.sliceNumber() * random(generator);
		//debug << "jumping to slice" << slice;
		reset_updates(conf, prob);
		return ret/N;
	}
};

class V3Measurements {
	private:
		std::vector<double> order_ts;
		std::vector<double> mag_ts;


		measurement<double> sign;
		measurement<double> order;
		measurement<double> density;
		measurement<double> magnetization;
		measurement<double> order_parameter;
		measurement<double> kinetic_energy;
		measurement<double> double_occupancy;
		measurement<double> chi_af;
		measurement<Eigen::ArrayXd> density_distribution_up;
		measurement<Eigen::ArrayXd> density_distribution_dn;

		Eigen::MatrixXd rho_up, rho_dn;
	public:
		void measure (V3Configuration &conf, V3Probability &prob, V3Updater &updater) {
			updater.flush_updates(conf, prob);
			size_t V = conf.volume();
			double beta = conf.inverseTemperature();
			//double mu = conf.chemicalPotential();
			double s = updater.sign();
			rho_up = prob.greenFunctionUp();
			rho_dn = Eigen::MatrixXd::Identity(V, V) - prob.greenFunctionDn(); //_flipped(conf);
			//Eigen::MatrixXd rho_alt = prob.greenFunctionDn_flipped(conf);
			//debug << "n_dn diff =" << rho_dn.diagonal().sum() - rho_alt.diagonal().sum();
			double K = (rho_up.diagonal() - rho_dn.diagonal()).transpose() * conf.eigenValues();
			double n_up = rho_up.diagonal().array().sum();
			double n_dn = rho_dn.diagonal().array().sum();
			//debug << "n" << n_up << n_dn;
			//std::cerr << rho_up.diagonal().transpose() << " -> " << n_up/conf.volume() << std::endl;
			//std::cerr << rho_dn.diagonal().transpose() << " -> " << n_dn/conf.volume() << std::endl;
			//K -= (n_up+n_dn) * mu;
			rho_up = conf.eigenVectors() * rho_up * conf.eigenVectors().transpose();
			rho_dn = conf.eigenVectors() * rho_dn * conf.eigenVectors().transpose();
			//std::cerr << rho_up.diagonal().transpose() << " -> " << n_up << std::endl;
			//std::cerr << rho_dn.diagonal().transpose() << " -> " << n_dn << std::endl;
			double op = (rho_up.diagonal().array()-rho_dn.diagonal().array()).square().sum();
			double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
			// add to measurements
			sign.add(s);
			order.add(conf.verticesNumber());
			density.add(s*(n_up+n_dn)/conf.volume());
			magnetization.add(s*(n_up-n_dn)/conf.volume());
			order_parameter.add(op);
			kinetic_energy.add(s*K/conf.volume());
			double_occupancy.add(s*n2/conf.volume());
			density_distribution_up.add(s*rho_up.diagonal());
			density_distribution_dn.add(s*rho_dn.diagonal());
			double af = 0.0;
			for (int i=0;i<rho_up.diagonal().size();i++) {
				af += (rho_up.diagonal().array()-rho_dn.diagonal().array())[i]*(i%2?1:-1);
			}
			af /= conf.volume();
			chi_af.add(s*beta*af*af);
			//std::ofstream out("dens.dat");
			//for (int x=0;x<4;x++) for (int y=0;y<4;y++) {
				//out << x << ' ' << y << ' ' << rho_up(4*x+y, 4*x+y) << ' ' << rho_dn(4*x+y, 4*x+y) << '\n';
			//}
			//out.close();
		}

		void measure_ts (V3Configuration &conf, V3Probability &prob, V3Updater &updater) {
			order_ts.reserve(200000);
			mag_ts.reserve(200000);
			rho_up = prob.greenFunctionUp();
			rho_dn = Eigen::MatrixXd::Identity(conf.volume(), conf.volume()) - prob.greenFunctionDn(); //_flipped(conf);
			double n_up = rho_up.diagonal().array().sum();
			double n_dn = rho_dn.diagonal().array().sum();
			order_ts.push_back(conf.verticesNumber());
			mag_ts.push_back((n_up-n_dn)/conf.volume());
		}

		void write_ts (std::string fn) {
			std::ofstream f(fn);
			for (size_t i=0;i<order_ts.size();i++) {
				f << i << ' ' << order_ts[i] << ' ' << mag_ts[i] << std::endl;
			}
			f.close();
		}

		size_t samples () const { return sign.samples(); }

		template <typename T>
		void report (T &out) const {
			out << "sign = " << sign.mean() << " +- " << sign.error() << " tau=" << sign.time() << ",\n";
			out << "order = " << order.mean() << " +- " << order.error() << " tau=" << order.time() << ",\n";
			out << "density = " << density.mean() << " +- " << density.error() << " tau=" << density.time() << ",\n";
			out << "magnetization = " << magnetization.mean() << " +- " << magnetization.error() << " tau=" << magnetization.time() << ",\n";
			out << "order_parameter = " << order_parameter.mean() << " +- " << order_parameter.error() << " tau=" << order_parameter.time() << ",\n";
			out << "kinetic_energy = " << kinetic_energy.mean() << " +- " << kinetic_energy.error() << " tau=" << kinetic_energy.time() << ",\n";
			out << "double_occupancy = " << double_occupancy.mean() << " +- " << double_occupancy.error() << " tau=" << double_occupancy.time() << ",\n";
			out << "chi_af = " << chi_af.mean() << " +- " << chi_af.error() << " tau=" << chi_af.time() << ",\n";
		}
};

typedef std::chrono::duration<double> seconds_type;

int main (int argc, char **argv) {
	steady_clock::time_point t0 = steady_clock::now();
	signal(10, my_signal_handler);
	signal(12, my_signal_handler);
	signal(14, my_signal_handler);
	unsigned int seed;
	std::ifstream seedfile("/dev/urandom");
	debug << "reading random seed";
	seedfile.read(reinterpret_cast<char*>(&seed), sizeof(unsigned int));
	debug << "read random seed";
	seedfile.close();

	double beta = 5.0, mu = 0.5, U = 4.0, K = 5.0;
	string outfile = "data.out";

	V3Configuration configuration;
	V3Probability prob;
	V3Updater updater;

	if (argc<6) {
		std::cerr << argv[0] << " $beta $mu $U $K $outfile" << std::endl;
		return -1;
	}

	if (argc>6) {
		updater.set_dump(argv[6]);
	}

	beta = atof(argv[1]);
	mu = atof(argv[2]);
	U = atof(argv[3]);
	K = atof(argv[4]);
	outfile = argv[5];

	std::string::size_type placeholder = outfile.find("%J");
	if (placeholder!=std::string::npos) {
		std::string jid = getenv("SLURM_PROCID");
		outfile.replace(placeholder, 2, jid);
		debug << "replaced placeholder" << outfile;
	}

	// t.U, t.B, t.mu = -t.U, 2.0*t.mu-t.U, 0.5*(t.B-t.U)
	configuration.setBeta(std::min(20.0, beta));
	configuration.setMu(-0.5*U);
	configuration.setB(2.0*mu-U);

	debug << configuration.mu_up() << configuration.mu_dn();

	updater.setU(U);
	updater.setK(K);
	updater.setSeed(seed);

	SquareLattice lattice;
	lattice.setSize(4, 4, 1);
	lattice.compute();
	configuration.setEigenvectors(lattice.eigenvectors());
	configuration.setEigenvalues(lattice.eigenvalues());
	configuration.make_slices(4.0*beta);

	//debug << lattice.eigenvectors() << '\n';
	//debug << lattice.eigenvectors().rowwise().reverse().colwise().reverse() << '\n';

	V3Measurements measurements;

	const int thermalization = 000000;
	const int sweeps = 1000000;

	t0 = steady_clock::now();
	updater.setup(configuration, prob);
	for (int n=0;n<thermalization+sweeps;n++) {
		double a = updater.sweep(configuration, prob);
		if (n>=thermalization) measurements.measure(configuration, prob, updater);
		//measurements.measure_ts(configuration, prob, updater);
		if (signalled==10) {
			signalled = 0;
			cerr << "SIGNAL 1" << endl;
			cerr << "beta =" << configuration.inverseTemperature() << ' ' << n << " sweeps, " << configuration.verticesNumber() << " vertices" << endl;
			cerr << "acceptance: " << a << endl;
			if (n>=thermalization) measurements.report(std::cerr);
			cerr << endl;
		}
		if (signalled==12) {
			V3Slice slice;
			vector<V3Slice> slices;
			int nslices = beta;
			double dtau = 1.0/nslices;
			double ld1 = 0.0, ld2 = 0.0;
			for (int i=0;i<nslices;i++) {
				slice.setup(dtau, lattice.volume(), lattice.eigenvectors(), lattice.eigenvalues());
				for (auto v : configuration.vertices()) {
					if (i*dtau<=v.tau && v.tau<(i+1)*dtau) {
						slice.insert(v);
						ld1 += std::log(1.0+v.sigma);
					}
				}
				slices.push_back(slice);
				slice.clear();
			}
			SVDHelper svd;
			svd.setIdentity(lattice.volume());
			for (int i=0;i<nslices;i++) {
				svd.U.applyOnTheLeft(slices[i].matrix());
				svd.absorbU();
			}
			std::cerr << svd.S.array().log().sum() << ' ' << ld1 << ' ' << ld2 << endl;
			for (int j=0;j<6;j++) for (int i=0;i<nslices;i++) {
				svd.U.applyOnTheLeft(slices[i].matrix());
				svd.absorbU();
				svd.Vt.applyOnTheRight(slices[i].inverse());
				svd.absorbVt();
				std::cerr << "shift " << i << ' ' << (svd.S.array().log().sum()-ld1)/ld1 << ' ' << ld2 << endl;
			}
			//std::cerr << slice.matrix()*slice.inverse() << endl << endl;
			signalled = 0;
			cerr << "SIGNAL 2" << endl;
			cerr << "beta =" << configuration.inverseTemperature() << ' ' << n << " sweeps, " << configuration.verticesNumber() << " vertices" << endl;
			cerr << "acceptance: " << a << endl;
			if (n>=thermalization) measurements.report(std::cerr);
			cerr << endl;
			//configuration.printout("debug.state");
		}
		if (signalled==14) {
			break;
		}
	}
	for (size_t k=0;k<configuration.sliceNumber();k++) {
		configuration.recheck_slice(k);
	}
	for (size_t k=0;k<configuration.sliceNumber();k++) {
		debug << configuration.sliceSize(k);
	}

	std::ofstream out(outfile);

	out << "beta = " << beta << ",\n";
	out << "mu = " << mu << ",\n";
	out << "U = " << U << ",\n";
	out << "K = " << K << ",\n";
	out << "seed = " << seed << ",\n";
	out << "total time = " << duration_cast<seconds_type>(steady_clock::now()-t0).count() << ",\n";
	out << "samples = " << measurements.samples() << ",\n";

	measurements.report(out);

	return 0;
}

