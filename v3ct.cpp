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

#include "helpers.hpp"
#include "measurements.hpp"
#include "weighted_measurements.hpp"
#include "logger.hpp"
#include "svd.hpp"

extern "C" {
#include <fftw3.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <set>

#include <csignal>

//#define fftw_execute (void)

using SVDMatrix = SVDHelper;

struct Vertex {
	double tau;
	size_t x;
	double sigma;
	class Compare {
		public:
			constexpr bool operator() (const Vertex& a, const Vertex& b) {
				return (a.tau<b.tau) || (a.tau==b.tau && a.x<b.x);
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

	std::vector<Matrix_d> slices_up;
	std::vector<Matrix_d> slices_dn;

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

	void addVertex (const Vertex& v) { verts.insert(v); }

	void make_slice (Matrix_d &G, double a, double b, double s) {
		auto first = verts.lower_bound(Vertex(a, 0, 0));
		auto last = verts.lower_bound(Vertex(b, 0, 0));
		double t = a;
		Matrix_d cache;
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

	void make_slices (size_t n) {
		slices_up.resize(n);
		slices_dn.resize(n);
		std::fill(slices_up.begin(), slices_up.end(), Matrix_d::Identity(V, V));
		std::fill(slices_dn.begin(), slices_dn.end(), Matrix_d::Identity(V, V));
		for (size_t i=0;i<n;i++) {
			make_slice(slices_up[i], beta/n*i, beta/n*(i+1), +1.0);
			make_slice(slices_dn[i], beta/n*i, beta/n*(i+1), -1.0);
		}
	}

	std::pair<double, double> probability_from_scratch (size_t n) {
		make_slices(n);
		SVDMatrix svd_up, svd_dn;
		svd_up.setIdentity(V);
		svd_dn.setIdentity(V);
		for (size_t t=0;t<n;t++) {
			svd_up.U.applyOnTheLeft(slices_up[t]);
			svd_up.absorbU();
			svd_dn.U.applyOnTheLeft(slices_dn[t]);
			svd_dn.absorbU();
		}
		svd_up.add_identity(exp(beta*mu));
		svd_dn.add_identity(exp(beta*mu));
		std::pair<double, double> ret;
		ret.first = svd_up.S.array().log().sum() + svd_dn.S.array().log().sum();
		ret.second = (svd_up.U*svd_up.Vt*svd_dn.U*svd_dn.Vt).determinant()>0.0?1.0:-1.0;
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

	SquareLattice (): Lx(2), Ly(2), Lz(1), V(4), tx(1.0), ty(1.0), tz(1.0), computed(false) {}
};

int main (int argc, char **argv) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	if (luaL_dofile(L, argv[1])) {
		std::cerr << "Error loading configuration file \"" << argv[1] << "\":" << std::endl;
		std::cerr << '\t' << lua_tostring(L, -1) << std::endl;
		return -1;
	}

	fftw_init_threads();
	fftw_plan_with_nthreads(1);

	int nthreads = 1;
	Logger log(cout);
	log << "using" << nthreads << "threads";

	double beta = 5.0, mu = 2.0;
	V3Configuration configuration;

	configuration.setBeta(beta);
	configuration.setMu(mu);

	SquareLattice lattice;
	lattice.setSize(4, 4, 1);
	lattice.compute();
	configuration.setEigenvectors(lattice.eigenvectors());
	configuration.setEigenvalues(lattice.eigenvalues());

	std::mt19937_64 generator;
	VertexFactory factory(generator);
	factory.setVolume(configuration.volume());
	factory.setBeta(beta);

	cerr << lattice.eigenvalues().transpose() << endl << endl << lattice.eigenvectors() << endl << endl;

	cerr << "base probability " << ((-beta*lattice.eigenvalues().array()+beta*mu).exp()+1.0).log().sum()*2.0 << endl;
	cerr << "computed probability " << configuration.probability_from_scratch(10).first << endl;

	for (int n=0;n<beta*configuration.volume()*5;n++) {
		configuration.addVertex(factory.generate(0.5));
		cerr << (n+1) << " vertices" << endl;
		for (int i=0;i<30;i+=5)
			cerr << (i+1) << " svds probability " << configuration.probability_from_scratch(i+1).first << endl;
		cerr << endl;
	}

	lua_close(L);
	fftw_cleanup_threads();
	return 0;
}



