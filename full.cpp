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

extern "C" {
#include <fftw3.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

//#define fftw_execute (void)

static const double pi = 3.141592653589793238462643383279502884197;

class Simulation {
	private:
	int Lx, Ly, Lz; // size of the system
	int V; // volume of the system
	int N; // number of time-steps
	double beta; // inverse temperature
	double dt; // time step 
	double g; // interaction strength
	double mu; // chemical potential
	double A; // sqrt(exp(g*dt)-1)
	double B; // magnetic field
	double tx, ty, tz; // nearest neighbour hopping
	double Vx, Vy, Vz; // trap strength

	double staggered_field;

	std::vector<Vector_d> diagonals;

	std::mt19937_64 generator;
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::exponential_distribution<double> trialDistribution;

	int steps;

	Vector_d energies;
	Vector_d freePropagator;
	Vector_d freePropagator_b;
	Vector_d potential;
	Vector_d freePropagator_x;
	Vector_d freePropagator_x_b;
	Array_d staggering;

	Matrix_d positionSpace; // current matrix in position space
	Matrix_cd positionSpace_c; // current matrix in position space
	Matrix_cd momentumSpace;

	Vector_cd v_x;
	Vector_cd v_p;

	fftw_plan x2p_vec;
	fftw_plan p2x_vec;

	fftw_plan x2p_col;
	fftw_plan p2x_col;

	fftw_plan x2p_row;
	fftw_plan p2x_row;

	double plog;

	int thermalization_sweeps;
	int total_sweeps;
	bool reset;
	//int reweight;
	int decompositions;
	std::string outfn;
	//std::ofstream logfile;

	Matrix_d U_s;
	Matrix_d U_s_inv;

	struct {
		Vector_d u;
		Vector_d v;
		Vector_d u_inv;
		Vector_d v_inv;
		Vector_cd ev;
		struct {
			Vector_d S;
			Matrix_d U;
			Matrix_d V;
		} svd;
	} cache;

	std::vector<Matrix_d> short_list;
	struct {
		Vector_d S;
		Matrix_d U;
		Matrix_d V;
	} svd_up;
	struct {
		Vector_d S;
		Matrix_d U;
		Matrix_d V;
	} svd_dn;


	mymeasurement<double> acceptance;
	mymeasurement<double> density;
	mymeasurement<double> magnetization;
	mymeasurement<double> kinetic;
	mymeasurement<double> interaction;
	std::vector<mymeasurement<double>> d_up;
	std::vector<mymeasurement<double>> d_dn;
	std::vector<mymeasurement<double>> spincorrelation;
	mymeasurement<double> sws;
	mymeasurement<double> staggered_magnetization;

	int shift_x (int x, int k) {
		int a = (x/Ly/Lz)%Lx;
		int b = x%(Ly*Lz);
		return ((a+k)%Lx)*Ly*Lz + b;
	}

	int shift_y (int y, int k) {
		int a = (y/Lz)%Ly;
		int b = y-a*Lz;
		return ((a+k)%Ly)*Lz + b;
	}

	public:

	void init () {
		if (Lx<2) { Lx = 1; tx = 0.0; }
		if (Ly<2) { Ly = 1; ty = 0.0; }
		if (Lz<2) { Lz = 1; tz = 0.0; }
		V = Lx * Ly * Lz;
		randomPosition = std::uniform_int_distribution<int>(0, V-1);
		randomTime = std::uniform_int_distribution<int>(0, N-1);
		dt = beta/N;
		A = sqrt(exp(g*dt)-1.0);
		diagonals.insert(diagonals.begin(), N, Vector_d::Zero(V));
		for (size_t i=0;i<diagonals.size();i++) {
			for (int j=0;j<V;j++) {
				diagonals[i][j] = distribution(generator)?A:-A;
				//diagonals[i][j] = A;
			}
		}
		v_x = Vector_cd::Zero(V);
		v_p = Vector_cd::Zero(V);
		positionSpace.setIdentity(V, V);
		positionSpace_c.setIdentity(V, V);
		momentumSpace.setIdentity(V, V);

		cache.svd.S = Vector_d::Zero(V);
		cache.svd.U = Matrix_d::Zero(V, V);
		cache.svd.V = Matrix_d::Zero(V, V);

		svd_up.S = Vector_d::Zero(V);
		svd_up.U = Matrix_d::Zero(V, V);
		svd_up.V = Matrix_d::Zero(V, V);

		svd_dn.S = Vector_d::Zero(V);
		svd_dn.U = Matrix_d::Zero(V, V);
		svd_dn.V = Matrix_d::Zero(V, V);

		const int size[] = { Lx, Ly, Lz, };
		x2p_vec = fftw_plan_dft(3, size, reinterpret_cast<fftw_complex*>(v_x.data()), reinterpret_cast<fftw_complex*>(v_p.data()), FFTW_FORWARD, FFTW_PATIENT);
		p2x_vec = fftw_plan_dft(3, size, reinterpret_cast<fftw_complex*>(v_p.data()), reinterpret_cast<fftw_complex*>(v_x.data()), FFTW_BACKWARD, FFTW_PATIENT);
		x2p_col = fftw_plan_many_dft(3, size, V, reinterpret_cast<fftw_complex*>(positionSpace_c.data()),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_FORWARD, FFTW_PATIENT);
		p2x_col = fftw_plan_many_dft(3, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(positionSpace_c.data()), NULL, 1, V, FFTW_BACKWARD, FFTW_PATIENT);
		x2p_row = fftw_plan_many_dft_r2c(3, size, V, positionSpace.data(),
				NULL, V, 1, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, V, 1, FFTW_PATIENT);
		p2x_row = fftw_plan_many_dft_c2r(3, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, V, 1, positionSpace.data(), NULL, V, 1, FFTW_PATIENT);

		for (int l=0;l<0;l++) {
			Matrix_cd R = Matrix_cd::Random(V, V);
			positionSpace_c = R;
			fftw_execute(x2p_col);
			fftw_execute(p2x_col);
			positionSpace_c /= V;
			std::cerr << l << ' ' << (positionSpace_c-R).norm() << std::endl;
		}

		positionSpace.setIdentity(V, V);
		momentumSpace.setIdentity(V, V);

		U_s = Matrix_d::Identity(V, V);
		U_s_inv = Matrix_d::Identity(V, V);

		energies = Vector_d::Zero(V);
		freePropagator = Vector_d::Zero(V);
		freePropagator_b = Vector_d::Zero(V);
		potential = Vector_d::Zero(V);
		freePropagator_x = Vector_d::Zero(V);
		freePropagator_x_b = Vector_d::Zero(V);
		staggering = Array_d::Zero(V);
		for (int i=0;i<V;i++) {
			int x = (i/Lz/Ly)%Lx;
			int y = (i/Lz)%Ly;
			int z = i%Lz;
			int Kx = Lx, Ky = Ly, Kz = Lz;
			int kx = (i/Kz/Ky)%Kx;
			int ky = (i/Kz)%Ky;
			int kz = i%Kz;
			energies[i] += -2.0 * ( tx * cos(2.0*kx*pi/Kx) + ty * cos(2.0*ky*pi/Ky) + tz * cos(2.0*kz*pi/Kz) );
			freePropagator[i] = exp(-dt*energies[i]);
			freePropagator_b[i] = exp(dt*energies[i]);
			potential[i] = (x+y+z)%2?-staggered_field:staggered_field;
			freePropagator_x[i] = exp(-dt*potential[i]);
			freePropagator_x_b[i] = exp(dt*potential[i]);
			staggering[i] = (x+y+z)%2?-1.0:1.0;
		}

		//std::cerr << "en_sum " << energies.array().sum() << std::endl;

		//accumulate_forward();
		//U_s = positionSpace;
		//accumulate_backward();
		//U_s_inv = positionSpace;

		fill_short_list();
		svd_from_short_list();
		plog = svd_probability();

		for (int i=0;i<V;i++) {
			d_up.push_back(mymeasurement<double>());
			d_dn.push_back(mymeasurement<double>());
		}
		for (int i=0;i<=Lx/2;i++) {
			spincorrelation.push_back(mymeasurement<double>());
		}

		if (decompositions<1) decompositions = N; 
	}

	Simulation (lua_State *L, int index) : distribution(0.5), trialDistribution(1.0), steps(0) {
		lua_getfield(L, index, "THERMALIZATION"); thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
		lua_getfield(L, index, "SWEEPS"); total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
		lua_getfield(L, index, "SEED"); generator.seed(lua_tointeger(L, -1)); lua_pop(L, 1);
		lua_getfield(L, index, "Lx");   this->Lx = lua_tointeger(L, -1);           lua_pop(L, 1);
		lua_getfield(L, index, "Ly");   this->Ly = lua_tointeger(L, -1);           lua_pop(L, 1);
		lua_getfield(L, index, "Lz");   this->Lz = lua_tointeger(L, -1);           lua_pop(L, 1);
		lua_getfield(L, index, "N");    N = lua_tointeger(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "T");    beta = 1.0/lua_tonumber(L, -1);            lua_pop(L, 1);
		lua_getfield(L, index, "tx");   tx = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "ty");   ty = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "tz");   tz = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "Vx");   Vx = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "Vy");   Vy = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "Vz");   Vz = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "U");    g = -lua_tonumber(L, -1);                  lua_pop(L, 1); // FIXME: check this // should be right as seen in A above
		lua_getfield(L, index, "mu");   mu = lua_tonumber(L, -1);                  lua_pop(L, 1);
		lua_getfield(L, index, "B");    B = lua_tonumber(L, -1);                   lua_pop(L, 1);
		lua_getfield(L, index, "h");    staggered_field = lua_tonumber(L, -1);     lua_pop(L, 1);
		lua_getfield(L, index, "RESET");  reset = lua_toboolean(L, -1);            lua_pop(L, 1);
		//lua_getfield(L, index, "REWEIGHT");  reweight = lua_tointeger(L, -1);      lua_pop(L, 1);
		lua_getfield(L, index, "OUTPUT");  outfn = lua_tostring(L, -1);            lua_pop(L, 1);
		//lua_getfield(L, index, "LOGFILE");  logfile.open(lua_tostring(L, -1));     lua_pop(L, 1);
		lua_getfield(L, index, "SVDPERIOD");  decompositions = lua_tointeger(L, -1);     lua_pop(L, 1);
		init();
	}

	double logDetU_s (int x = -1, int t = -1) {
		int nspinup = 0;
		for (int i=0;i<N;i++) {
			for (int j=0;j<V;j++) {
				if (diagonals[i][j]>0.0) nspinup++;
			}
		}
		if (x>=0 && t>=0) {
			nspinup += diagonals[t][x]>0.0?-1:+1;
		}
		return nspinup*std::log(1.0+A) + (N*V-nspinup)*std::log(1.0-A);
	}

	void accumulate_forward (int start = 0, int end = -1) {
		positionSpace_c.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			//std::cerr << "accumulate_f. " << i << " determinant = " << positionSpace_c.determinant() << std::endl;
			positionSpace_c.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonals[i]).array()*freePropagator_x.array()).matrix().asDiagonal());
			fftw_execute(x2p_col);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x_col);
			positionSpace_c /= V;
		}
		positionSpace = positionSpace_c.real();
	}

	void accumulate_backward (int start = 0, int end = -1) {
		Real X = 1.0 - A*A;
		positionSpace.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			positionSpace.applyOnTheRight(((Vector_d::Constant(V, 1.0)-diagonals[i]).array()*freePropagator_x_b.array()).matrix().asDiagonal());
			fftw_execute(x2p_row);
			momentumSpace.applyOnTheRight(freePropagator_b.asDiagonal());
			fftw_execute(p2x_row);
			positionSpace /= V*X;
		}
	}

	void fill_short_list () {
		const int M = decompositions==0?N:decompositions;
		short_list.clear();
		for (int i=0;i<N;i+=M) {
			accumulate_forward(i, i+M);
			short_list.push_back(positionSpace);
		}
	}

	void update_short_list (int x, int t) {
		const int M = decompositions==0?N:decompositions;
		const int s = t/M;
		diagonals[t][x] = -diagonals[t][x];
		//std::cerr << "update: " << t << std::endl;
		accumulate_forward(s*M, s*M+M);
		short_list[s] = positionSpace;
	}

	void svd_from_short_list () {
		collapseSVD(short_list, cache.svd.S, cache.svd.U, cache.svd.V);
		U_s = cache.svd.U * cache.svd.S.asDiagonal() * cache.svd.V.transpose();
		U_s_inv = cache.svd.V * cache.svd.S.array().inverse().matrix().asDiagonal() * cache.svd.U.transpose();
		//std::cerr << cache.svd.S.array().log().sum() << " <> " << logDetU_s() << " // " << (cache.svd.S.array().log().sum()-logDetU_s())/beta/V*Lz << std::endl;
		Complex ret = 0.0;
		Matrix_d M;

		M = cache.svd.U.transpose()*cache.svd.V;
		M.diagonal() += std::exp(+beta*B*0.5+beta*mu)*cache.svd.S;
		dgesvd(M, svd_up.S, svd_up.U, svd_up.V); // 1+U*S*V^t -> (V + U*S) V^t -> U (U^t*V + S) V^t -> U U' S' V'^t V^t -> (UU') S' (VV')^t
		svd_up.U.applyOnTheLeft(cache.svd.U);
		svd_up.V.applyOnTheLeft(cache.svd.V);

		M = cache.svd.U.transpose()*cache.svd.V;
		M.diagonal() += std::exp(-beta*B*0.5+beta*mu)*cache.svd.S;
		dgesvd(M, svd_dn.S, svd_dn.U, svd_dn.V); // 1+U*S*V^t -> (V + U*S) V^t -> U (U^t*V + S) V^t
		svd_dn.U.applyOnTheLeft(cache.svd.U);
		svd_dn.V.applyOnTheLeft(cache.svd.V);

		ret += svd_up.S.array().log().sum();
		ret += svd_dn.S.array().log().sum();

		double sign = (svd_up.U*svd_up.V*svd_dn.U*svd_dn.V).determinant();
		if ( (sign)<1e-5 || std::isnan(ret.real()) || std::isnan(ret.imag())) {
			//std::cerr << "prob_complex = " << ret << " det=" << cache.svd.S.array().log().sum() << " " << sign << std::endl;
			//std::cerr << cache.svd.U.determinant() << std::endl;
			//std::cerr << cache.svd.V.determinant() << std::endl;
			//std::cerr << svd_up.U.determinant() << std::endl;
			//std::cerr << svd_up.V.determinant() << std::endl;
			//std::cerr << svd_dn.U.determinant() << std::endl;
			//std::cerr << svd_dn.V.determinant() << std::endl;
			//throw "";
		}
	}

	double rank1_probability (int x, int t) {
		compute_uv_f(x, t);
		double a = (cache.v.transpose()*svd_up.V).eval()*svd_up.S.array().inverse().matrix().asDiagonal()*(svd_up.U.transpose()*cache.u).eval();
		double b = (cache.v.transpose()*svd_dn.V).eval()*svd_dn.S.array().inverse().matrix().asDiagonal()*(svd_dn.U.transpose()*cache.u).eval();
		if (false) {
			compute_uv_f(0, 0);
			std::cerr << cache.u.transpose() << std::endl;
			std::cerr << cache.v.transpose() << std::endl;
			accumulate_forward();
			std::cerr << std::endl << positionSpace << std::endl;
			fill_short_list();
			svd_from_short_list();
			//std::cerr << std::endl << cache.svd.U*cache.svd.S.asDiagonal()*cache.svd.V.transpose() << std::endl;
			//diagonals[0][0] *= -1;
			//accumulate_forward();
			//std::cerr << std::endl << positionSpace << std::endl;
			accumulate_backward();
			std::cerr << cache.u.transpose()*positionSpace << std::endl;
			std::cerr << (positionSpace*cache.u).transpose() << std::endl;
			std::cerr << cache.v.transpose() << std::endl;
			std::cerr << (cache.svd.V*cache.svd.S.array().inverse().matrix().asDiagonal()*cache.svd.U.transpose()*cache.u).transpose() << std::endl;
			std::cerr << "dot prod = " << (cache.svd.S.array().inverse().matrix().asDiagonal()*cache.svd.U.transpose()*cache.u).transpose()*(cache.svd.V.transpose()*cache.v) << std::endl;
		}
		//std::cerr << a << ' ' << b << ' ' << cache.v.transpose()*cache.svd.V*cache.svd.S.array().inverse().matrix().asDiagonal()*cache.svd.U.transpose()*cache.u << std::endl;
		return std::log((1+std::exp(+beta*B*0.5+beta*mu)*a)*(1+std::exp(-beta*B*0.5+beta*mu)*b));
	}

	double svd_probability () {
		return svd_up.S.array().log().sum() + svd_dn.S.array().log().sum();
	}

	double logProbability_complex () {
		const int M = decompositions==0?N:decompositions;
		std::vector<Matrix_d> fvec;
		std::vector<Matrix_d> bvec;
		for (int i=0;i<N;i+=M) {
			accumulate_forward(i, i+M);
			fvec.push_back(positionSpace);
		}
		for (int i=0;i<N;i+=M) {
			accumulate_backward(i, i+M);
			bvec.push_back(positionSpace);
		}
		//test_sequences(fvec, bvec);
		collapseSVD(fvec, cache.svd.S, cache.svd.U, cache.svd.V);
		Complex ret = 0.0;
		double sign = 1.0;
		Vector_d S;
		Matrix_d U;
		Matrix_d V;
		U = cache.svd.U.transpose()*cache.svd.V;
		U.diagonal() += std::exp(+beta*B*0.5+beta*mu)*cache.svd.S;
		dgesvd(U, S, U, V); // 1+U*S*V^t -> (V + U*S) V^t -> U (U^t*V + S) V^t
		sign *= (U*V).determinant();
		U = cache.svd.U.transpose()*cache.svd.V;
		ret += S.array().log().sum();
		U.diagonal() += std::exp(-beta*B*0.5+beta*mu)*cache.svd.S;
		dgesvd(U, S, U, V); // 1+U*S*V^t -> (V + U*S) V^t -> U (U^t*V + S) V^t
		sign *= (U*V).determinant();
		//collapseSVD(bvec, cache.svd.S, cache.svd.U, cache.svd.V);
		ret += S.array().log().sum();

		if ( (sign)<1e-5 || std::isnan(ret.real()) || std::isnan(ret.imag())) {
			std::cerr << "prob_complex = " << ret << " det=" << cache.svd.S.array().log().sum() << " " << sign << std::endl;
			throw "";
		}
		return ret.real();
	}

	void compute_uv_f (int x, int t) {
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<N;i++) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonals[i]).array() * freePropagator_x.array();
			v_x /= V;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.u = (-2*diagonals[t][x]*v_x*freePropagator_x[x]).real();
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=0;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonals[i]).array() * freePropagator_x.array();
			v_x /= V;
		}
		cache.v = v_x.real();
	}

	void compute_uv_b (int x, int t) {
		Real X = 1-A*A;
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<N;i++) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array() * freePropagator_x_b.array();
			v_x /= V*X;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator_b.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.v_inv = (v_x * freePropagator_x_b[x]).real();
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=0;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array() * freePropagator_x_b.array();
			v_x /= V*X;
		}
		cache.u_inv = (+2*diagonals[t][x]/X*v_x).real();
	}

	Vector_cd rank1EV_f (int x, int t, const Matrix_d &M) {
		compute_uv_f(x, t);
		return (M+cache.u*cache.v.transpose()).eigenvalues();
	}

	Vector_cd rank1EV_b (int x, int t, const Matrix_d &M) {
		compute_uv_b(x, t);
		return (M+cache.u_inv*cache.v_inv.transpose()).eigenvalues();
	}

	double rank1prob (int x, int t) {
		Vector_cd eva = Vector_cd::Ones(V);
		Vector_cd evc = rank1EV_b(x, t, U_s_inv);
		Vector_cd evb = rank1EV_f(x, t, U_s);
		sort_vector(evb);
		sort_vector(evc);
		reverse_vector(evc);
		for (int i=0;i<V;i++) {
			if (std::norm(evb[i]/evb[0])<std::norm(evc[i]/evc[V-1])) {
				eva[i] = ((Real)1.0)/evc[i];
			} else {
				eva[i] = evb[i];
			}
		}
		cache.ev = eva;
		std::complex<double> ret = 0.0;
		ret += (Vector_cd::Ones(V) + std::exp(+beta*B*0.5+beta*mu)*eva).array().log().sum();
		ret += (Vector_cd::Ones(V) + std::exp(-beta*B*0.5+beta*mu)*eva).array().log().sum();
		return ret.real();
	}

	bool metropolis (int M = 0) {
		steps++;
		bool ret = false;
		//bool svd = false;
		int x = randomPosition(generator);
		int t = randomTime(generator);
		double trial;
		//double exact = logDetU_s(x, t);
		//double trial = rank1prob(x, t);
		//Complex c =  cache.ev.array().log().sum();

		//if ( std::cos(c.imag())<0.99 || std::abs(1.0-c.real()/exact)>1.0e-5 ) {
			//std::cerr << " recomputing exact = " << exact << " trial=" << c;
			//accumulate_forward();
			//U_s = positionSpace;
			//accumulate_backward();
			//U_s_inv = positionSpace;
			//trial = rank1prob(x, t);
			//c = cache.ev.array().log().sum();
			//std::cerr << " new =" << c << " CN = " << cache.ev[0]/cache.ev[V-1] << std::endl;
		//}

		if (true) {
			//std::cerr << exact << std::endl;
			//diagonals[t][x] = -diagonals[t][x];
			//c = logProbability_complex();
			//diagonals[t][x] = -diagonals[t][x];
			//fill_short_list();
			//svd_from_short_list();
			//double old = svd_probability();
			//diagonals[t][x] = -diagonals[t][x];
			//fill_short_list();
			//update_short_list(x, t);
			//svd_from_short_list();
			//trial = svd_probability();
			//std::cerr << trial-plog << " r1: " << r << std::endl;
			//svd = true;
			//trial = plog + r;
		}
		double r = rank1_probability(x, t);
		ret = -trialDistribution(generator)<r;
		//std::cerr << ret << " r = " << r << std::endl;

		if (ret) {
			plog += r;
			update_short_list(x, t);
			svd_from_short_list();
		} else {
		}
		return ret;
	}

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		for (int i=0;i<10;i++) acceptance.add(metropolis()?1.0:0.0);
		//if (steps%1000==0) {
			//accumulate_forward();
			//U_s = positionSpace;
			//accumulate_backward();
			//U_s_inv = positionSpace;
		//}
	}

	double extract_data (const Matrix_d &M) {
		positionSpace_c = M.cast<Complex>();
		fftw_execute(x2p_col);
		momentumSpace.applyOnTheLeft(energies.asDiagonal());
		fftw_execute(p2x_col);
		return positionSpace_c.real().trace() / V;
	}

	void measure () {
		//svd_from_short_list();
		Matrix_d rho_up = (Matrix_d::Identity(V, V) + exp(-beta*B*0.5-beta*mu)*U_s_inv).inverse();
		Matrix_d rho_dn = (Matrix_d::Identity(V, V) + exp(-beta*B*0.5+beta*mu)*U_s).inverse();
		double K_up = extract_data(rho_up);
		double K_dn = extract_data(rho_dn);
		double n_up = rho_up.diagonal().array().sum();
		double n_dn = rho_dn.diagonal().array().sum();
		double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
		density.add((n_up+n_dn)/V);
		magnetization.add((n_up-n_dn)/2.0/V);
		kinetic.add(K_up-K_dn);
		interaction.add(g*n2);
		//- (d1_up*d2_up).sum() - (d1_dn*d2_dn).sum();
		for (int i=0;i<V;i++) {
			d_up[i].add(rho_up(i, i));
			d_dn[i].add(rho_dn(i, i));
		}
		for (int k=1;k<=Lx/2;k++) {
			double ssz = 0.0;
			for (int j=0;j<V;j++) {
				int x = j;
				int y = shift_x(j, k);
				ssz += rho_up(x, x)*rho_up(y, y) + rho_dn(x, x)*rho_dn(y, y);
				ssz -= rho_up(x, x)*rho_dn(y, y) + rho_dn(x, x)*rho_up(y, y);
				ssz -= rho_up(x, y)*rho_up(y, x) + rho_dn(x, y)*rho_dn(y, x);
			}
			spincorrelation[k].add(0.25*ssz);
			staggered_magnetization.add((rho_up.diagonal().array()*staggering - rho_dn.diagonal().array()*staggering).sum()/V);
			if (isnan(ssz)) {
				//std::cerr << "explain:" << std::endl;
				//std::cerr << "k=" << k << " ssz=" << ssz << std::endl;
				//for (int j=0;j<V;j++) {
					//int x = j;
					//int y = shift_x(j, k);
					//std::cerr << " j=" << j
						//<< " a_j=" << (rho_up(x, x)*rho_up(y, y) + rho_dn(x, x)*rho_dn(y, y))
						//<< " b_j=" << (rho_up(x, x)*rho_dn(y, y) + rho_dn(x, x)*rho_up(y, y))
						//<< " c_j=" << (rho_up(x, y)*rho_up(y, x) + rho_dn(x, y)*rho_dn(y, x)) << std::endl;
				//}
				//throw "";
			}
		}
	}

	int volume () { return V; }
	int timeSlices () { return N; }

	void output_results () {
		std::ostringstream buf;
		buf << outfn << "U" << (g/tx) << "_T" << 1.0/(beta*tx) << '_' << Lx << 'x' << Ly << 'x' << Lz << ".dat";
		outfn = buf.str();
		std::ofstream out(outfn, reset?std::ios::trunc:std::ios::app);
		out << 1.0/(beta*tx) << ' ' << 0.5*(B+g)/tx
			<< ' ' << density.mean() << ' ' << density.variance()
			<< ' ' << magnetization.mean() << ' ' << magnetization.variance()
			//<< ' ' << acceptance.mean() << ' ' << acceptance.variance()
			<< ' ' << kinetic.mean()/tx/V << ' ' << kinetic.variance()/tx/tx/V/V
			<< ' ' << interaction.mean()/tx/V << ' ' << interaction.variance()/tx/tx/V/V
			<< ' ' << -staggered_magnetization.mean()/staggered_field << ' ' << staggered_magnetization.variance();
		for (int i=0;i<V;i++) {
			//out << ' ' << d_up[i].mean();
		}
		for (int i=0;i<V;i++) {
			//out << ' ' << d_dn[i].mean();
		}
		for (int i=1;i<=Lx/2;i++) {
			//out << ' ' << spincorrelation[i].mean()/V << ' ' << spincorrelation[i].variance();
		}
		out << std::endl;
	}

	std::string params () {
		std::ostringstream buf;
		buf << "T=" << 1.0/(beta*tx) << "";
		return buf.str();
	}

	~Simulation () {
		fftw_destroy_plan(x2p_vec);
		fftw_destroy_plan(p2x_vec);
		fftw_destroy_plan(x2p_col);
		fftw_destroy_plan(p2x_col);
		fftw_destroy_plan(x2p_row);
		fftw_destroy_plan(p2x_row);
	}
	protected:
};

using namespace std;
using namespace std::chrono;

typedef std::chrono::duration<double> seconds_type;

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
	char *e = getenv("LSB_HOSTS");
	while (e!=NULL && *e!='\0') if (*(e++)==' ') nthreads++;

	lua_getfield(L, -1, "THREADS");
	if (lua_tointeger(L, -1)) {
		nthreads = lua_tointeger(L, -1);
	}
	lua_pop(L, 1);
	Logger log;
	//log.setVerbosity(5);
	log << "using" << nthreads << "threads";

	std::vector<std::thread> threads(nthreads);
	std::mutex lock;
	std::atomic<int> failed;
	failed = 0;
	std::atomic<int> current;
	current = 1;
	for (int j=0;j<nthreads;j++) {
		threads[j] = std::thread( [=, &log, &lock, &current, &failed] () {
				steady_clock::time_point t0 = steady_clock::now();
				steady_clock::time_point t1 = steady_clock::now();
				log << "thread" << j << "starting";
				while (true) {
					int job = current.fetch_add(1);
					lock.lock();
					lua_rawgeti(L, -1, job);
					if (lua_isnil(L, -1)) {
						log << "thread" << j << "terminating";
						lua_pop(L, 1);
						lock.unlock();
						break;
					}
					log << "thread" << j << "running simulation" << job;
					lua_getfield(L, -1, "THERMALIZATION"); int thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
					lua_getfield(L, -1, "SWEEPS"); int total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
					Simulation simulation(L, -1);
					lua_pop(L, 1);
					lock.unlock();
					try {
						t0 = steady_clock::now();
						for (int i=0;i<thermalization_sweeps;i++) {
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>2) {
								t1 = steady_clock::now();
								log << "thread" << j << "thermalizing: " << i << '/' << thermalization_sweeps << '.' << (double(i)/duration_cast<seconds_type>(t1-t0).count()) << "updates per second";
							}
							simulation.update();
						}
						log << "thread" << j << "thermalized";
						t0 = steady_clock::now();
						for (int i=0;i<total_sweeps;i++) {
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>2) {
								t1 = steady_clock::now();
								log << "thread" << j << "running: " << i << '/' << total_sweeps << '.' << (double(i)/duration_cast<seconds_type>(t1-t0).count()) << "updates per second";
							}
							simulation.update();
							simulation.measure();
						}
						log << "thread" << j << "finished simulation" << job;
						lock.lock();
						simulation.output_results();
						lock.unlock();
					} catch (...) {
						failed++;
						log << "thread" << j << "caught exception in simulation" << job << " with params " << simulation.params();
					}
				}
		});
	}
	for (std::thread& t : threads) t.join();

	std::cout << failed << " tasks failed" << std::endl;

	lua_close(L);
	fftw_cleanup_threads();
	return 0;
}

