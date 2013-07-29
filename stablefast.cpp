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

	//int update_start;
	//int update_end;

	std::mt19937_64 generator;
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::uniform_int_distribution<int> randomStep;
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

	double update_prob;
	double update_sign;
	int update_size;
	int max_update_size;
	Matrix_d update_U;
	Matrix_d update_Vt;

	int msvd;
	SVDHelper svd;
	SVDHelper svdA;
	SVDHelper svdB;
	SVDHelper svd_inverse;
	SVDHelper svd_inverse_up;
	SVDHelper svd_inverse_dn;
	Matrix_d first_slice_inverse;

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
	mymeasurement<double> sign;
	std::vector<mymeasurement<double>> d_up;
	std::vector<mymeasurement<double>> d_dn;
	std::vector<mymeasurement<double>> spincorrelation;
	std::vector<mymeasurement<double>> error;
	mymeasurement<double> staggered_magnetization;

	int last_t;
	int time_shift;

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

	Vector_d& diagonal (int t) {
		return diagonals[(t+time_shift)%N];
	}

	public:

	void init_measurements () {
		sign.set_name("Sign");
		acceptance.set_name("Acceptance");
		density.set_name("Density");
		magnetization.set_name("Magnetization");
		for (int i=0;i<V;i++) {
			d_up.push_back(mymeasurement<double>());
			d_dn.push_back(mymeasurement<double>());
		}
		for (int i=0;i<=Lx/2;i++) {
			spincorrelation.push_back(mymeasurement<double>());
		}
		for (int i=0;i<N;i++) {
			error.push_back(mymeasurement<double>());
		}
	}

	void reset_updates () {
		update_prob = 0.0;
		update_sign = 1.0;
		update_size = 0.0;
		update_U.setZero(V, max_update_size);
		update_Vt.setZero(max_update_size, V);
	}

	void init () {
		if (Lx<2) { Lx = 1; tx = 0.0; }
		if (Ly<2) { Ly = 1; ty = 0.0; }
		if (Lz<2) { Lz = 1; tz = 0.0; }
		V = Lx * Ly * Lz;
		mslices = mslices>0?mslices:N;
		mslices = mslices<N?mslices:N;
		time_shift = 0;
		last_t = 0;
		if (max_update_size<1 || max_update_size>V) max_update_size = 1;
		randomPosition = std::uniform_int_distribution<int>(0, V-1);
		randomTime = std::uniform_int_distribution<int>(0, N-1);
		randomStep = std::uniform_int_distribution<int>(0, mslices-1);
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

		//accumulate_forward();
		//U_s = positionSpace;
		//accumulate_backward();
		//U_s_inv = positionSpace;

		compute_U_s();

		init_measurements();
		reset_updates();
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
		lua_getfield(L, index, "SVD");     msvd = lua_tointeger(L, -1);            lua_pop(L, 1);
		lua_getfield(L, index, "max_update_size");     max_update_size = lua_tointeger(L, -1);            lua_pop(L, 1);
		//lua_getfield(L, index, "update_start");     update_start = lua_tointeger(L, -1);         lua_pop(L, 1);
		//lua_getfield(L, index, "update_end");       update_end = lua_tointeger(L, -1);           lua_pop(L, 1);
		//lua_getfield(L, index, "LOGFILE");  logfile.open(lua_tostring(L, -1));     lua_pop(L, 1);
		init();
	}

	double logDetU_s (int x = -1, int t = -1) {
		int nspinup = 0;
		for (int i=0;i<N;i++) {
			for (int j=0;j<V;j++) {
				if (diagonal(i)[j]>0.0) nspinup++;
			}
		}
		if (x>=0 && t>=0) {
			nspinup += diagonal(t)[x]>0.0?-1:+1;
		}
		return nspinup*std::log(1.0+A) + (N*V-nspinup)*std::log(1.0-A);
	}

	void make_slices () {
		slices.clear();
		for (int i=0;i<N;i+=mslices) {
			accumulate_forward(i, i+mslices);
			slices.push_back(positionSpace);
		}
	}

	void make_svd () {
		svd.setIdentity(V);
		for (size_t i=0;i<slices.size();i++) {
			svd.U.applyOnTheLeft(slices[i]);
			if (i%msvd==0 || i==slices.size()-1) svd.absorbU();
		}
	}

	void make_density_matrices () {
		svdA = svd;
		svdA.add_identity(std::exp(+beta*B*0.5+beta*mu));
		svdB = svd;
		svdB.add_identity(std::exp(-beta*B*0.5+beta*mu));
	}

	double svd_probability () {
		double ret = svdA.S.array().log().sum() + svdB.S.array().log().sum();
		//std::cerr << svd.S.transpose() << std::endl;
		return ret; // * (svdA.U*svdA.Vt*svdB.U*svdB.Vt).determinant();
	}

	double svd_sign () {
		return (svdA.U*svdA.Vt*svdB.U*svdB.Vt).determinant()>0.0?1.0:-1.0;
	}

	void accumulate_forward (int start = 0, int end = -1) {
		positionSpace_c.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			//std::cerr << "accumulate_f. " << i << " determinant = " << positionSpace_c.determinant() << std::endl;
			positionSpace_c.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()*freePropagator_x.array()).matrix().asDiagonal());
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
			positionSpace_c.applyOnTheRight(((Vector_d::Constant(V, 1.0)-diagonal(i)).array()*freePropagator_x_b.array()).matrix().asDiagonal());
			fftw_execute(x2p_row);
			momentumSpace.applyOnTheRight(freePropagator_b.asDiagonal());
			fftw_execute(p2x_row);
			positionSpace_c /= V*X;
		}
		positionSpace = positionSpace_c.real();
	}

	void compute_uv_f (int x, int t) {
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<N;i++) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
			v_x /= V;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.u = (-2*diagonal(t)[x]*v_x*freePropagator_x[x]).real();
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=0;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
			v_x /= V;
		}
		cache.v = v_x.real();
	}

	void compute_uv_f_short (int x, int t) {
		int start = mslices*(t/mslices);
		int end = mslices*(1+t/mslices);
		if (end>N) end = N;
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<end;i++) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
			v_x /= V;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.u_smart = (-2*diagonal(t)[x]*v_x*freePropagator_x[x]).real();
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=start;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
			v_x /= V;
		}
		cache.v_smart = v_x.real();
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
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
			v_x /= V;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.u_smart = cache.u = (-2*diagonal(t)[x]*v_x*freePropagator_x[x]).real();
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
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
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
		v_x[x] = (2.0*diagonal(t)[x])/(1.0-diagonal(t)[x]);
		for (int i=t;i<N;i++) {
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonal(i)).array() * freePropagator_x_b.array();
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
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonal(i)).array() * freePropagator_x_b.array();
			v_x /= V*X;
		}
		cache.u = v_x.real();
	}

	void compute_U_s () {
		make_slices();
		make_svd();
		make_density_matrices();
		plog = svd_probability();
	}

	void test_U_s (bool update) {
		//accumulate_backward();
		//Matrix_d newC = (positionSpace+std::exp(-beta*B*0.5+beta*mu)*Matrix_d::Identity(V, V)).inverse();
		cache.A = svdA.inverse();
		cache.B = svdB.inverse();
		make_svd();
		svdA.setIdentity(V);
		svdA.U = svd.U.transpose() * svd.Vt.transpose();
		svdA.U.diagonal() += std::exp(+beta*B*0.5+beta*mu)*svd.S;
		svdA.absorbU();
		svdA.U.applyOnTheLeft(svd.U);
		svdA.Vt.applyOnTheRight(svd.Vt);
		svdB.setIdentity(V);
		svdB.U = svd.U.transpose() * svd.Vt.transpose();
		svdB.U.diagonal() += std::exp(-beta*B*0.5+beta*mu)*svd.S;
		svdB.absorbU();
		svdB.U.applyOnTheLeft(svd.U);
		svdB.Vt.applyOnTheRight(svd.Vt);
		accumulate_forward();
		Matrix_d newA = svdA.inverse();
		Matrix_d newB = svdB.inverse();
		if ((cache.A-newA).norm()>1e-7*newA.norm() || (cache.B-newB).norm()>1e-7*newB.norm() || (svd.matrix()-positionSpace).norm()>1e-7*positionSpace.norm()) {
			std::cerr << log((cache.A-newA).norm())-log(newA.norm())
				<< ' ' << log((cache.B-newB).norm())-log(newB.norm())
				<< ' ' << log((svd.matrix()-positionSpace).norm())-log(positionSpace.norm())
				<< ' ' << svd.S.array().log().sum() << " " << logDetU_s()
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

	SVDHelper make_rankN_update (int t, const std::vector<int> &vec) {
		const int L = vec.size();
		SVDHelper sm;
		sm.U.setZero(V, L);
		sm.S.setOnes(L);
		sm.Vt.setZero(L, V);
		for (size_t i=0;i<vec.size();i++) {
			if (t>=0 && t<N) {
				compute_uv_f_smart(vec[i], t);
				sm.U.col(i) = cache.u_smart;
				sm.Vt.row(i) = cache.v_smart.transpose();
			} else {
				sm.U.col(i)[vec[i]] = 1.0;
				sm.Vt.row(i)[vec[i]] = 1.0;
			}
		}
		sm.absorbU();
		sm.absorbVt();
		for (size_t i=t/mslices+1;i<slices.size();i++) {
			sm.U.applyOnTheLeft(slices[i]);
			sm.absorbU();
		}

		//std::cerr << cache.u.transpose() << std::endl;
		//std::cerr << sm.U.transpose() << std::endl;
		for (int i=t/mslices-1;i>=0;i--) {
			sm.Vt.applyOnTheRight(slices[i]);
			sm.absorbVt();
		}
		return sm;
	}

	void flip (int t, int x) {
		diagonal(t)[x] = -diagonal(t)[x];
	}

	void flip (int t, const std::vector<int> &vec) {
		for (int x : vec) {
			diagonal(t)[x] = -diagonal(t)[x];
		}
	}

	void update_U_s (int x, int t) { // FIXME submatrix updates
		//Matrix_d M = svdA.matrix() + std::exp(+beta*B*0.5+beta*mu)*cache.u*cache.v.transpose();
		//SVDHelper svdC = svdA;
		//SVDHelper svdD = svdB;
		//SVDHelper svdE = svd;
		//svdC.setIdentity(V);
		//svdC.U = std::exp(+beta*B*0.5+beta*mu)*(svdA.U.transpose()*cache.u) * (cache.v.transpose()*svdA.Vt.transpose());
		//svdC.U.diagonal() += svdA.S;
		//svdC.absorbU();
		//svdC.U.applyOnTheLeft(svdA.U);
		//svdC.Vt.applyOnTheRight(svdA.Vt);
		//svdC.rank1_update(cache.u, cache.v, std::exp(+beta*B*0.5+beta*mu));
		//svdD.rank1_update(cache.u, cache.v, std::exp(-beta*B*0.5+beta*mu));
		//svdE.rank1_update(cache.u, cache.v);
		
		//compute_uv_f(x, t);
		//U_s += cache.u*cache.v.transpose();
		//cache.A -= (cache.A*cache.u)*std::exp(+beta*B*0.5+beta*mu)*(cache.v.transpose()*cache.A)/(1.0+std::exp(+beta*B*0.5+beta*mu)*cache.a);
		//cache.B -= (cache.B*cache.u)*std::exp(-beta*B*0.5+beta*mu)*(cache.v.transpose()*cache.B)/(1.0+std::exp(-beta*B*0.5+beta*mu)*cache.b);
		//compute_uv_b(x, t);
		slices[t/mslices] += cache.u_smart*cache.v_smart.transpose();
		diagonal(t)[x] = -diagonal(t)[x];
		//Matrix_d newC = (positionSpace+std::exp(-beta*B*0.5+beta*mu)*Matrix_d::Identity(V, V)).inverse();
		//cache.C -= (cache.C*cache.u)*(cache.v.transpose()*cache.C)/(1.0+cache.v.transpose()*cache.C*cache.u);
		
		//test_U_s(true);
		make_svd();
		make_density_matrices();

		//std::cerr << "svd diffs "
			//<< (svdA.U-svdC.U).norm() << ' '
			//<< (svdA.S-svdC.S).norm() << ' '
			//<< (svdA.Vt-svdC.Vt).norm() << ' '
			//<< (svdA.matrix()-svdC.matrix()).norm()
			//<< std::endl;

		//std::cerr << svdE.S.transpose() << std::endl;
		//std::cerr << svd.S.transpose() << std::endl;
		//std::cerr << (svdE.S-svd.S).transpose() << std::endl;
		//svdA = svdC;
	}

	void redo_all () {
		make_slices();
		make_svd();
		svd_inverse = svd;
		svd_inverse.invertInPlace();
		svd_inverse_up = svd_inverse;
		svd_inverse_up.add_identity(std::exp(-beta*B*0.5-beta*mu));
		svd_inverse_up.invertInPlace();
		svd_inverse_dn = svd_inverse;
		svd_inverse_dn.add_identity(std::exp(+beta*B*0.5-beta*mu));
		svd_inverse_dn.invertInPlace();
		first_slice_inverse = slices[0].inverse();
		make_density_matrices();
		double np = svd_probability();
		std::cerr << plog+update_prob << " <> " << np << " ~~ " << np-plog-update_prob << std::endl;
		error[last_t].add(np-plog-update_prob);
		plog = np;
		reset_updates();
	}

	std::pair<double, double> rank1_probability (int x, int t) { // FIXME: use SVD / higher beta
		compute_uv_f_smart(x, t);
		const int L = update_size;
		update_U.col(L) = first_slice_inverse * cache.u_smart;
		update_Vt.row(L) = cache.v_smart.transpose();
		double d1 = ((update_Vt*svd_inverse_up.U) * svd_inverse_up.S.asDiagonal() * (svd_inverse_up.Vt*update_U) + Matrix_d::Identity(L+1, L+1)).determinant();
		double d2 = ((update_Vt*svd_inverse_dn.U) * svd_inverse_dn.S.asDiagonal() * (svd_inverse_dn.Vt*update_U) + Matrix_d::Identity(L+1, L+1)).determinant();
		//double d1 = ( (update_Vt.topRows(L+1)*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*update_U.leftCols(L+1))*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L+1, L+1) ).determinant();
		//double d2 = ( (update_Vt.topRows(L+1)*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*update_U.leftCols(L+1))*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L+1, L+1) ).determinant();
		std::cerr << L <<  " (" << x << ", " << t << ')' << std::endl;
		return std::pair<double, double>(std::log(d1)+std::log(d2), 1.0);
	}

	void make_tests () {
		for (int i=0;i<0;i++) {
			int x = randomPosition(generator);
			int t = randomTime(generator);
			accumulate_backward();
			U_s_inv = positionSpace;
			compute_uv_b(x, t);
			diagonal(t)[x] = -diagonal(t)[x];
			accumulate_backward();
			std::cerr << "U_s^-1 + uv^t - U'_s^-1 = " << (U_s_inv+cache.u*cache.v.transpose()-positionSpace).norm() << std::endl << std::endl;
			std::cerr << "U_s^-1 - U'_s^-1 = " << std::endl << ((U_s_inv-positionSpace).array()/(cache.u*cache.v.transpose()).array()) << std::endl << std::endl;
			std::cerr << "uv^t = " << std::endl << (cache.u*cache.v.transpose()) << std::endl << std::endl;
		}
		if (false) {
			int t = randomTime(generator);
			const int n = 2;
			std::vector<int> vec;
			for (int i=0;i<n;i++) {
				vec.push_back(randomPosition(generator));
			}
			SVDHelper s = make_rankN_update(t, vec);
			double d1 = ( (s.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(n, n) ).determinant();
			double d2 = ( (s.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(n, n) ).determinant();
			flip(t, vec);
			make_slices();
			make_svd();
			make_density_matrices();
			double np = svd_probability();
			std::cerr << plog +std::log(d1) +std::log(d2) << ' ' << np << std::endl;
			for (int x : vec) std::cerr << x << ' ';
			std::cerr << std::endl;
			flip(t, vec);
			make_slices();
			make_svd();
			make_density_matrices();
		}
		if (false) {
			const int L = 1;
			std::vector<int> vt;
			std::vector<int> vx;
			SVDHelper s;
			s.U.setZero(V, L);
			s.S.setOnes(L);
			s.Vt.setZero(L, V);
			double r1 = 0.0;
			for (int i=0;i<L;i++) {
				int t = 0;
				int x = i;
				//r1 += rank1_probability(x, t);
				vt.push_back(t);
				vx.push_back(x);
				compute_uv_f_smart(x, t);
				s.U.col(i) = cache.u;
				s.Vt.row(i) = cache.v.transpose();
			}
			SVDHelper h = make_rankN_update(0, vx);
			for (int i=0;i<L;i++) {
				flip(vt[i], vx[i]);
				//slices[vt[i]/mslices] += cache.u_smart*cache.v_smart.transpose();
			}
			double p1 = svdA.S.array().log().sum();
			double p2 = svdB.S.array().log().sum();
			double d1 = ( (s.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			double d2 = ( (s.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			double e1 = ( (h.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*h.U)*h.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			double e2 = ( (h.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*h.U)*h.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			h.Vt.applyOnTheRight(svdB.Vt.transpose());
			h.Vt.applyOnTheRight(svdB.S.array().inverse().matrix().asDiagonal());
			h.absorbVt();
			h.Vt.applyOnTheRight(svdB.U.transpose());
			h.Vt.applyOnTheRight(h.U);
			make_slices();
			make_svd();
			make_density_matrices();
			double np = svd_probability();
			std::cerr << p1 + std::log(d1) << ' ' << p1 + std::log(e1) << ' ' << svdA.S.array().log().sum() << std::endl;
			std::cerr << p2 + std::log(d2) << ' ' << p2 + std::log(e2) << ' ' << svdB.S.array().log().sum() << std::endl;
			std::cerr << "breakout: " << d1 << ' ' << std::log(d1) << ' ' << d2 << ' ' << std::log(d2) << std::endl;
			for (int t : vt) std::cerr << t << '\t';
			std::cerr << std::endl;
			for (int x : vx) std::cerr << x << '\t';
			std::cerr << std::endl;
			for (int i=0;i<L;i++) {
				flip(vt[i], vx[i]);
			}
			make_slices();
			make_svd();
			make_density_matrices();
		}
		if (false) {
			const int L = 1;
			std::vector<int> vt;
			std::vector<int> vx;
			SVDHelper s, sb;
			SVDHelper svdC;
			svdC.S = svd.S.array().inverse().matrix();
			svdC.U = svd.Vt.transpose();
			svdC.Vt = svd.U.transpose();
			svdC.add_identity(std::exp(+beta*B*0.5-beta*mu));
			s.U.setZero(V, L);
			s.S.setOnes(L);
			s.Vt.setZero(L, V);
			sb.U.setZero(V, L);
			sb.S.setOnes(L);
			sb.Vt.setZero(L, V);
			double r1 = 0.0;
			for (int i=0;i<L;i++) {
				int t = 0;
				int x = i;
				//r1 += rank1_probability(x, t);
				vt.push_back(t);
				vx.push_back(x);
				compute_uv_f_smart(x, t);
				s.U.col(i) = cache.u;
				s.Vt.row(i) = cache.v.transpose();
				compute_uv_b(x, t);
				sb.U.col(i) = cache.u;
				sb.Vt.row(i) = cache.v.transpose();
			}
			SVDHelper h = make_rankN_update(0, vx);
			for (int i=0;i<L;i++) {
				flip(vt[i], vx[i]);
				//slices[vt[i]/mslices] += cache.u_smart*cache.v_smart.transpose();
			}
			double e3;
			double X, Y;
			double p3 = svdC.S.array().log().sum();
			double p4 = svd.S.array().log().sum();
			{
				X = ((sb.Vt*svd.U)*svd.S.asDiagonal()*(svd.Vt*sb.U) + Matrix_d::Identity(L, L)).determinant();
				Y = ((sb.Vt*svdC.Vt.transpose())*svdC.S.array().inverse().matrix().asDiagonal()*(svdC.U.transpose()*sb.U)*std::exp(+beta*B*0.5-beta*mu) + Matrix_d::Identity(L, L)).determinant();
				std::cerr << "e3 " << X << ' ' << Y << std::endl;
				e3 = std::log(Y) - std::log(X);
			}
			double p1 = svdA.S.array().log().sum();
			double p2 = svdB.S.array().log().sum();
			double d1 = ( (s.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Matrix_d::Identity(L, L) ).determinant();
			double d2 = ( (s.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Matrix_d::Identity(L, L) ).determinant();
			double e1 = ( (h.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*h.U)*h.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Matrix_d::Identity(L, L) ).determinant();
			double e2 = ( (h.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*h.U)*h.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Matrix_d::Identity(L, L) ).determinant();
			h.Vt.applyOnTheRight(svdB.Vt.transpose());
			h.Vt.applyOnTheRight(svdB.S.array().inverse().matrix().asDiagonal());
			h.absorbVt();
			h.Vt.applyOnTheRight(svdB.U.transpose());
			h.Vt.applyOnTheRight(h.U);
			make_slices();
			make_svd();
			make_density_matrices();
			svdC.S = svd.S.array().inverse().matrix();
			svdC.U = svd.Vt.transpose();
			svdC.Vt = svd.U.transpose();
			svdC.add_identity(std::exp(+beta*B*0.5-beta*mu));
			double np = svd_probability();
			std::cerr << p1 + std::log(d1) << ' ' << p1 + std::log(e1) << ' ' << svdA.S.array().log().sum() << std::endl;
			std::cerr << p2 + std::log(d2) << ' ' << p2 + std::log(e2) << ' ' << svdB.S.array().log().sum() << std::endl;
			std::cerr << std::log(X) << ' ' << X << ' ' << svd.S.array().log().sum()-p4 << std::endl;
			std::cerr << std::log(Y) << ' ' << Y << ' ' << svdC.S.array().log().sum()-p3 << std::endl;
			std::cerr << "breakout: " << d1 << ' ' << std::log(d1) << ' ' << d2 << ' ' << std::log(d2) << std::endl;
			for (int t : vt) std::cerr << t << '\t';
			std::cerr << std::endl;
			for (int x : vx) std::cerr << x << '\t';
			std::cerr << std::endl;
			for (int i=0;i<L;i++) {
				flip(vt[i], vx[i]);
			}
			make_slices();
			make_svd();
			make_density_matrices();
		}
		if (false) {
			// TODO use circular multiplication of slices
			int t = randomTime(generator);
			int x = randomPosition(generator);
			int slice = t/mslices;
			SVDHelper svdC;
			svdC.setIdentity(V);
			for (size_t i=0;i<slices.size();i++) {
				svdC.U.applyOnTheLeft(slices[(slice+i)%slices.size()]);
				svdC.absorbU();
			}
			svdC.invertInPlace();
			std::cerr << "det check " << logDetU_s() << ' ' << svdC.S.array().log().sum() << std::endl;
			SVDHelper svdD = svdC, svdE = svdC;
			std::cerr << svdC.S.transpose() << std::endl << svdD.S.transpose() << std::endl;
			svdD.add_identity(std::exp(-beta*B*0.5-beta*mu));
			svdD.invertInPlace();
			std::cerr << svdD.S.transpose() << std::endl;
			svdE.add_identity(std::exp(+beta*B*0.5-beta*mu));
			svdE.invertInPlace();
			compute_uv_f_smart(x, t);
			cache.u_smart.applyOnTheLeft(slices[slice].inverse());
			double p1 = svdA.S.array().log().sum();
			double p2 = svdB.S.array().log().sum();
			double d1 = (cache.v_smart.transpose()*svdD.U) * svdD.S.asDiagonal() * (svdD.Vt*cache.u_smart) + 1.0;
			double d2 = (cache.v_smart.transpose()*svdE.U) * svdE.S.asDiagonal() * (svdE.Vt*cache.u_smart) + 1.0;
			flip(t, x);
			make_slices();
			make_svd();
			make_density_matrices();
			double np = svd_probability();
			double np1 = svdA.S.array().log().sum();
			double np2 = svdB.S.array().log().sum();
			std::cerr << "update -> " << d1 << ' ' << std::log(d1) << ' ' << np1-p1 << std::endl;
			std::cerr << "update -> " << d2 << ' ' << std::log(d2) << ' ' << np2-p2 << std::endl;
			make_slices();
			make_svd();
			make_density_matrices();
		}
		if (false) {
			const int L = V;
			std::vector<int> vt;
			std::vector<int> vx;
			SVDHelper s;
			SVDHelper simple;
			simple.U.setZero(V, L);
			simple.S.setOnes(L);
			simple.Vt.setZero(L, V);
			int t = 0;
			for (int i=0;i<L;i++) {
				int x = randomPosition(generator);
				//r1 += rank1_probability(x, t);
				vt.push_back(t);
				vx.push_back(i);
				compute_uv_f_smart(x, t);
				simple.U.col(i) = cache.u;
				simple.Vt.row(i) = cache.v.transpose();
				//flip(t, x);
				//slices[t/mslices] += cache.u_smart*cache.v_smart.transpose();
			}
			s = make_rankN_update(t, vx);
			for (int i=0;i<L;i++) {
				flip(vt[i], vx[i]);
			}
			double p1 = svdA.S.array().log().sum();
			double p2 = svdB.S.array().log().sum();
			SVDHelper S1 = s;
			S1.Vt.applyOnTheRight(svdA.Vt.transpose());
			S1.Vt.applyOnTheRight(svdA.S.array().inverse().matrix().asDiagonal());
			S1.absorbVt();
			S1.U.applyOnTheRight(svdA.U.transpose());
			S1.absorbU();
			S1.add_identity(std::exp(+beta*B*0.5+beta*mu));
			SVDHelper S2 = s;
			S2.Vt.applyOnTheRight(svdB.Vt.transpose());
			S2.Vt.applyOnTheRight(svdB.S.array().inverse().matrix().asDiagonal());
			S2.absorbVt();
			S2.U.applyOnTheRight(svdB.U.transpose());
			S2.absorbU();
			S2.add_identity(std::exp(+beta*B*0.5-beta*mu));
			std::cerr << "svd r1 update: " << S1.S.array().log().sum() << ' ' << svdA.S.array().log().sum() << ' ' << s.S.array().log().sum() << std::endl;
			std::cerr << " " << svd.S.transpose() << std::endl;
			std::cerr << " " << svdA.S.transpose() << std::endl;
			std::cerr << " " << svdB.S.transpose() << std::endl;
			std::cerr << " " << S1.S.transpose() << std::endl;
			std::cerr << " " << s.S.transpose() << std::endl;
			//double d1 = ( (s.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			//double d2 = ( (s.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*s.U)*s.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			double d1 = S1.S.array().log().sum();
			double d2 = S2.S.array().log().sum();
			double e1 = ( (simple.Vt*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*simple.U)*simple.S.asDiagonal()*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			double e2 = ( (simple.Vt*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*simple.U)*simple.S.asDiagonal()*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L, L) ).determinant();
			make_slices();
			make_svd();
			make_density_matrices();
			double np = svd_probability();
			std::cerr << "probs: " << p1  << ' ' << d1 << ' ' << std::log(e1) << ' ' << svdA.S.array().log().sum() << std::endl;
			std::cerr << "z = " << std::exp(+beta*B*0.5+beta*mu) << std::endl;
			for (int t : vt) std::cerr << t << '\t';
			std::cerr << std::endl;
			for (int x : vx) std::cerr << x << '\t';
			std::cerr << std::endl;
			for (int i=0;i<L;i++) {
				flip(vt[i], vx[i]);
			}
			make_slices();
			make_svd();
			make_density_matrices();
			throw "";
		}
	}

	bool metropolis (int M = 0) {
		steps++;
		bool ret = false;
		int x = randomPosition(generator);
		int t = randomStep(generator);
		std::pair<double, double> r1 = rank1_probability(x, t);
		ret = -trialDistribution(generator)<r1.first-update_prob;
		if (ret) {
			//std::cerr << "accepted" << std::endl;
			diagonal(t)[x] = -diagonal(t)[x];
			slices[t/mslices] += cache.u_smart*cache.v_smart.transpose();
			update_size++;
			update_prob = r1.first;
			update_sign = r1.second;
			//last_t = t;
		} else {
		}
		return ret;
	}

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		time_shift = randomTime(generator);
		make_slices();
		make_svd();
		svd_inverse = svd;
		svd_inverse.invertInPlace();
		svd_inverse_up = svd_inverse;
		svd_inverse_up.add_identity(std::exp(-beta*B*0.5-beta*mu));
		svd_inverse_up.invertInPlace();
		svd_inverse_dn = svd_inverse;
		svd_inverse_dn.add_identity(std::exp(+beta*B*0.5-beta*mu));
		svd_inverse_dn.invertInPlace();
		first_slice_inverse = slices[0].inverse();
		for (int i=0;i<100;i++) {
			acceptance.add(metropolis()?1.0:0.0);
			sign.add(svd_sign());
			if (update_size>=max_update_size) redo_all();
			make_tests();
		}
		compute_U_s();
		reset_updates();
		std::cerr << "update finished" << std::endl;
		std::ofstream out("error.dat");
		for (int i=0;i<N;i++) {
			if (error[i].samples()>0) out << i << ' ' << error[i].mean() << std::endl;
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
		double s = svd_sign();
		rho_up = Matrix_d::Identity(V, V) - svdA.inverse();
		rho_dn = svdB.inverse();
		double K_up = get_kinetic_energy(rho_up);
		double K_dn = get_kinetic_energy(rho_dn);
		double n_up = rho_up.diagonal().array().sum();
		double n_dn = rho_dn.diagonal().array().sum();
		double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
		density.add(s*n_up/V);
		magnetization.add(s*n_dn/V);
		kinetic.add(s*K_up-s*K_dn);
		interaction.add(s*g*n2);
		//sign.add(svd_sign());
		//- (d1_up*d2_up).sum() - (d1_dn*d2_dn).sum();
		for (int i=0;i<V;i++) {
			d_up[i].add(s*rho_up(i, i));
			d_dn[i].add(s*rho_dn(i, i));
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
			spincorrelation[k].add(s*0.25*ssz);
			if (staggered_field!=0.0) staggered_magnetization.add(s*(rho_up.diagonal().array()*staggering - rho_dn.diagonal().array()*staggering).sum()/V);
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
								log << simulation.sign;
								log << simulation.acceptance;
							}
							simulation.update();
						}
						log << "thread" << j << "thermalized";
						t0 = steady_clock::now();
						for (int i=0;i<total_sweeps;i++) {
							if (duration_cast<seconds_type>(steady_clock::now()-t1).count()>5) {
								t1 = steady_clock::now();
								log << "thread" << j << "running: " << i << '/' << total_sweeps << '.' << (double(i)/duration_cast<seconds_type>(t1-t0).count()) << "updates per second";
								log << simulation.sign;
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


