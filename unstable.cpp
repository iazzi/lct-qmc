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

template <typename T> using mymeasurement = measurement<T, false>;

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

	int mslices;
	std::vector<Matrix_d> slices;

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
	std::string outfn;
	//std::ofstream logfile;

	Matrix_d U_s;
	Matrix_d U_s_inv;

	Matrix_d rho_up;
	Matrix_d rho_dn;

	struct {
		double a;
		double b;
		//double c;
		Vector_d u;
		Vector_d v;
		Vector_d u_smart;
		Vector_d v_smart;
		Matrix_d A;
		Matrix_d B;
		//Matrix_d C;
	} cache;

	public:

	mymeasurement<double> acceptance;
	mymeasurement<double> density;
	mymeasurement<double> magnetization;
	mymeasurement<double> kinetic;
	mymeasurement<double> interaction;
	std::vector<mymeasurement<double>> d_up;
	std::vector<mymeasurement<double>> d_dn;
	std::vector<mymeasurement<double>> spincorrelation;
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

		const int size[] = { Lx, Ly, Lz, };
		x2p_vec = fftw_plan_dft(3, size, reinterpret_cast<fftw_complex*>(v_x.data()), reinterpret_cast<fftw_complex*>(v_p.data()), FFTW_FORWARD, FFTW_PATIENT);
		p2x_vec = fftw_plan_dft(3, size, reinterpret_cast<fftw_complex*>(v_p.data()), reinterpret_cast<fftw_complex*>(v_x.data()), FFTW_BACKWARD, FFTW_PATIENT);
		x2p_col = fftw_plan_many_dft(3, size, V, reinterpret_cast<fftw_complex*>(positionSpace_c.data()),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_FORWARD, FFTW_PATIENT);
		p2x_col = fftw_plan_many_dft(3, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(positionSpace_c.data()), NULL, 1, V, FFTW_BACKWARD, FFTW_PATIENT);
		x2p_row = fftw_plan_many_dft(3, size, V, reinterpret_cast<fftw_complex*>(positionSpace_c.data()),
				NULL, V, 1, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, V, 1, FFTW_FORWARD, FFTW_PATIENT);
		p2x_row = fftw_plan_many_dft(3, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, V, 1, reinterpret_cast<fftw_complex*>(positionSpace_c.data()), NULL, V, 1, FFTW_BACKWARD, FFTW_PATIENT);

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

		accumulate_forward();
		U_s = positionSpace;
		accumulate_backward();
		U_s_inv = positionSpace;

		compute_U_s();

		for (int i=0;i<V;i++) {
			d_up.push_back(mymeasurement<double>());
			d_dn.push_back(mymeasurement<double>());
		}
		for (int i=0;i<=Lx/2;i++) {
			spincorrelation.push_back(mymeasurement<double>());
		}

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
		lua_getfield(L, index, "SLICES");  mslices = lua_tointeger(L, -1);         lua_pop(L, 1);
		//lua_getfield(L, index, "LOGFILE");  logfile.open(lua_tostring(L, -1));     lua_pop(L, 1);
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
		positionSpace_c.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			positionSpace_c.applyOnTheRight(((Vector_d::Constant(V, 1.0)-diagonals[i]).array()*freePropagator_x_b.array()).matrix().asDiagonal());
			fftw_execute(x2p_row);
			momentumSpace.applyOnTheRight(freePropagator_b.asDiagonal());
			fftw_execute(p2x_row);
			positionSpace_c /= V*X;
		}
		positionSpace = positionSpace_c.real();
	}

	Vector_d compute_weird_vec (int x, int t) {
		Real X = 1-A*A;
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t;i>0;i--) {
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array() * freePropagator_x_b.array();
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x /= V*X;
		}
		v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[0]).array() * freePropagator_x_b.array();
		v_x /= X;
		return (-2*diagonals[t][x]*v_x*freePropagator_x[x]).real();
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

	void compute_uv_f_smart (int x, int t) {
		int start = mslices*(t/mslices);
		int end = mslices*(1+t/mslices);
		if (end>N) end = N;
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<end;i++) {
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
		cache.u_smart = cache.u = (-2*diagonals[t][x]*v_x*freePropagator_x[x]).real();
		for (size_t i=t/mslices+1;i<slices.size();i++) {
			//std::cerr << i << ' ' << t/mslices << ' ' << slices.size() << std::endl;
			cache.u.applyOnTheLeft(slices[i]);
		}
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=start;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonals[i]).array() * freePropagator_x.array();
			v_x /= V;
		}
		cache.v_smart = cache.v = v_x.real();
		for (int i=t/mslices-1;i>=0;i--) {
			cache.v.applyOnTheLeft(slices[i].transpose());
		}
	}

	void compute_uv_b (int x, int t) {
		Real X = 1-A*A;
		v_x.setZero(V);
		v_x[x] = (2.0*diagonals[t][x])/(1.0-diagonals[t][x]);
		for (int i=t;i<N;i++) {
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array() * freePropagator_x_b.array();
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x /= V*X;
		}
		cache.v = v_x.real();
		v_x.setZero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=0;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array() * freePropagator_x_b.array();
			v_x /= V*X;
		}
		cache.u = v_x.real();
	}

	void compute_U_s () {
		accumulate_forward();
		U_s = positionSpace;
		cache.A = (Matrix_d::Identity(V, V) + std::exp(+beta*B*0.5+beta*mu)*U_s).inverse();
		cache.B = (Matrix_d::Identity(V, V) + std::exp(-beta*B*0.5+beta*mu)*U_s).inverse();
		mslices = mslices>0?mslices:N;
		slices.clear();
		for (int i=0;i<N;i+=mslices) {
			accumulate_forward(i, i+mslices);
			slices.push_back(positionSpace);
		}
		//cache.C = (U_s.inverse()+std::exp(-beta*B*0.5+beta*mu)*Matrix_d::Identity(V, V)).inverse();
	}

	void test_U_s (bool update) {
		//accumulate_backward();
		//Matrix_d newC = (positionSpace+std::exp(-beta*B*0.5+beta*mu)*Matrix_d::Identity(V, V)).inverse();
		accumulate_forward();
		Matrix_d newA = (Matrix_d::Identity(V, V) + std::exp(+beta*B*0.5+beta*mu)*positionSpace).inverse();
		Matrix_d newB = (Matrix_d::Identity(V, V) + std::exp(-beta*B*0.5+beta*mu)*positionSpace).inverse();
		//accumulate_backward();
		//Matrix_d newB = Matrix_d::Identity(V, V) - (Matrix_d::Identity(V, V) + std::exp(+beta*B*0.5-beta*mu)*positionSpace).inverse();
		if ((cache.A-newA).norm()>1e-7*newA.norm() || (cache.B-newB).norm()>1e-7*newB.norm()) {
			std::cerr << log((cache.A-newA).norm())-log(newA.norm())
				<< ' ' << log((cache.B-newB).norm())-log(newB.norm())
				//<< ' ' << log((cache.C-newC).norm())-log(newC.norm())
				<< std::endl;
			//std::cerr << ((cache.A*(Matrix_d::Identity(V, V) + std::exp(+beta*B*0.5+beta*mu)*positionSpace) - Matrix_d::Identity(V, V))).norm()
				//<< ' ' << ((newA*(Matrix_d::Identity(V, V) + std::exp(+beta*B*0.5+beta*mu)*positionSpace) - Matrix_d::Identity(V, V))).norm() << std::endl;
			//std::cerr << ((cache.B*(Matrix_d::Identity(V, V) + std::exp(-beta*B*0.5+beta*mu)*positionSpace) - Matrix_d::Identity(V, V))).norm()
				//<< ' ' << ((newB*(Matrix_d::Identity(V, V) + std::exp(-beta*B*0.5+beta*mu)*positionSpace) - Matrix_d::Identity(V, V))).norm() << std::endl;
		}
		if (update) {
			U_s = positionSpace;
			cache.A = newA;
			cache.B = newB;
			//cache.C = newC;
			for (int i=0;i<N;i+=mslices) {
				accumulate_forward(i, i+mslices);
				slices[i/mslices] = positionSpace;
			}
		}
	}

	void update_U_s (int x, int t) { // FIXME submatrix updates
		//compute_uv_f(x, t);
		U_s += cache.u*cache.v.transpose();
		cache.A -= (cache.A*cache.u)*std::exp(+beta*B*0.5+beta*mu)*(cache.v.transpose()*cache.A)/(1.0+std::exp(+beta*B*0.5+beta*mu)*cache.a);
		cache.B -= (cache.B*cache.u)*std::exp(-beta*B*0.5+beta*mu)*(cache.v.transpose()*cache.B)/(1.0+std::exp(-beta*B*0.5+beta*mu)*cache.b);
		//compute_uv_b(x, t);
		slices[t/mslices] += cache.u_smart*cache.v_smart.transpose();
		diagonals[t][x] = -diagonals[t][x];
		//Matrix_d newC = (positionSpace+std::exp(-beta*B*0.5+beta*mu)*Matrix_d::Identity(V, V)).inverse();
		//cache.C -= (cache.C*cache.u)*(cache.v.transpose()*cache.C)/(1.0+cache.v.transpose()*cache.C*cache.u);
	}

	double rank1_probability (int x, int t) { // FIXME: use SVD / higher beta
		compute_uv_f_smart(x, t);
		//Vector_d w = compute_weird_vec(x, t);
		double a = cache.v.transpose()*cache.A*cache.u;
		double b = cache.v.transpose()*cache.B*cache.u;
		//double c = cache.v.transpose()*cache.C*w;
		cache.a = a;
		cache.b = b;
		//cache.c = c;
		//std::cerr << b << ' ' << std::exp(+beta*B*0.5-beta*mu)*cache.v.transpose()*(std::exp(+beta*B*0.5-beta*mu)*U_s.inverse()+Matrix_d::Identity(V, V)).inverse()*w << std::endl;
		//std::cerr << (U_s.inverse()*cache.u).transpose() << std::endl;
		//std::cerr << w.transpose() << std::endl;
		//std::cerr << "b, c = " << b << ", " << c << std::endl;
		double ret = std::log((1+std::exp(+beta*B*0.5+beta*mu)*a)*(1+std::exp(-beta*B*0.5+beta*mu)*b));
		if (std::isnan(ret) || isinf(ret)){
			accumulate_backward();
			Matrix_d M1 = (Matrix_d::Identity(V, V) + std::exp(-beta*B*0.5-beta*mu)*positionSpace).inverse();
			Matrix_d M2 = (Matrix_d::Identity(V, V) + std::exp(+beta*B*0.5-beta*mu)*positionSpace).inverse();
			//double c = cache.v.transpose()*cache.u - cache.v.transpose()*M*cache.u;
			double c = double(cache.v.transpose()*cache.u) - double(cache.v.transpose()*M1*cache.u);
			double d = double(cache.v.transpose()*cache.u) - double(cache.v.transpose()*M2*cache.u);
			std::cerr << "update probs " << std::exp(+beta*B*0.5+beta*mu)*a << " " << std::exp(-beta*B*0.5+beta*mu)*b << ' ' << std::exp(+beta*B*0.5+beta*mu)*c << ' ' << std::exp(-beta*B*0.5+beta*mu)*d << std::endl;
			std::cerr << c << ' ' << double(cache.v.transpose()*M1*cache.u) << ' ' << double(cache.v.transpose()*cache.u) << std::endl;
			std::cerr << d << ' ' << double(cache.v.transpose()*M2*cache.u) << ' ' << double(cache.v.transpose()*cache.u) << std::endl;
			std::cerr << std::exp(+beta*B*0.5+beta*mu) << " " << std::exp(-beta*B*0.5+beta*mu) << std::endl;
			//throw "";
		}
		return ret;
	}

	void make_tests () {
		for (int i=0;i<1;i++) {
			int x = randomPosition(generator);
			int t = randomTime(generator);
			accumulate_backward();
			U_s_inv = positionSpace;
			compute_uv_b(x, t);
			diagonals[t][x] = -diagonals[t][x];
			accumulate_backward();
			std::cerr << "U_s^-1 + uv^t - U'_s^-1 = " << (U_s_inv+cache.u*cache.v.transpose()-positionSpace).norm() << std::endl << std::endl;
			std::cerr << "U_s^-1 - U'_s^-1 = " << std::endl << ((U_s_inv-positionSpace).array()/(cache.u*cache.v.transpose()).array()) << std::endl << std::endl;
			std::cerr << "uv^t = " << std::endl << (cache.u*cache.v.transpose()) << std::endl << std::endl;
		}
	}

	bool metropolis (int M = 0) {
		steps++;
		bool ret = false;
		//bool svd = false;
		int x = randomPosition(generator);
		int t = randomTime(generator);
		double r = rank1_probability(x, t);
		ret = -trialDistribution(generator)<r;
		//std::cerr << ret << " r = " << r << std::endl;
		if (ret) {
			plog += r;
			update_U_s(x, t);
		} else {
		}
		return ret;
	}

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		for (int i=0;i<100;i++) {
			acceptance.add(metropolis()?1.0:0.0);
			//test_U_s(false);
			//make_tests();
			if (steps%1000==0) test_U_s(true);
		}
	}

	double get_kinetic_energy (const Matrix_d &M) {
		positionSpace_c = M.cast<Complex>();
		fftw_execute(x2p_col);
		momentumSpace.applyOnTheLeft(energies.asDiagonal());
		fftw_execute(p2x_col);
		return positionSpace_c.real().trace() / V;
	}

	void measure () {
		//accumulate_forward();
		//U_s = positionSpace;
		//accumulate_backward();
		U_s_inv = U_s.inverse();
		rho_up = (Matrix_d::Identity(V, V) + exp(-beta*B*0.5-beta*mu)*U_s_inv).inverse();
		rho_dn = (Matrix_d::Identity(V, V) + exp(-beta*B*0.5+beta*mu)*U_s).inverse();
		double K_up = get_kinetic_energy(rho_up);
		double K_dn = get_kinetic_energy(rho_dn);
		double n_up = rho_up.diagonal().array().sum();
		double n_dn = rho_dn.diagonal().array().sum();
		double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
		density.add(n_up/V);
		magnetization.add(n_dn/V);
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
			if (staggered_field!=0.0) staggered_magnetization.add((rho_up.diagonal().array()*staggering - rho_dn.diagonal().array()*staggering).sum()/V);
			if (std::isnan(ssz)) {
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
		buf << outfn << "unstable_U" << (g/tx) << "_T" << 1.0/(beta*tx) << '_' << Lx << 'x' << Ly << 'x' << Lz << ".dat";
		outfn = buf.str();
		std::ofstream out(outfn, reset?std::ios::trunc:std::ios::app);
		out << 1.0/(beta*tx) << ' ' << 0.5*(B+g)/tx
			<< ' ' << density.mean() << ' ' << density.variance()
			<< ' ' << magnetization.mean() << ' ' << magnetization.variance()
			//<< ' ' << acceptance.mean() << ' ' << acceptance.variance()
			<< ' ' << kinetic.mean()/tx/V << ' ' << kinetic.variance()/tx/tx/V/V
			<< ' ' << interaction.mean()/tx/V << ' ' << interaction.variance()/tx/tx/V/V;
		if (staggered_field!=0.0) out << ' ' << -staggered_magnetization.mean()/staggered_field << ' ' << staggered_magnetization.variance();
		for (int i=0;i<V;i++) {
			//out << ' ' << d_up[i].mean();
		}
		for (int i=0;i<V;i++) {
			//out << ' ' << d_dn[i].mean();
		}
		for (int i=1;i<=Lx/2;i++) {
			out << ' ' << spincorrelation[i].mean()/V << ' ' << spincorrelation[i].variance();
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
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
								t1 = steady_clock::now();
								log << "thread" << j << "thermalizing: " << i << '/' << thermalization_sweeps << '.' << (double(i)/duration_cast<seconds_type>(t1-t0).count()) << "updates per second";
							}
							simulation.update();
						}
						log << "thread" << j << "thermalized";
						t0 = steady_clock::now();
						for (int i=0;i<total_sweeps;i++) {
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
								t1 = steady_clock::now();
								log << "thread" << j << "running: " << i << '/' << total_sweeps << '.' << (double(i)/duration_cast<seconds_type>(t1-t0).count()) << "updates per second";
								log << simulation.density;
								log << simulation.magnetization;
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

