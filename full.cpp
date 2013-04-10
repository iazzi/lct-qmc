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

#ifdef USE_LONG_DOUBLE
# define fftw_plan fftwl_plan
# define fftw_plan_dft_r2c fftwl_plan_dft_r2c
# define fftw_plan_dft_c2r fftwl_plan_dft_c2r
# define fftw_complex fftwl_complex
# define fftw_plan_many_dft_r2c fftwl_plan_many_dft_r2c
# define fftw_plan_many_dft_c2r fftwl_plan_many_dft_c2r
# define fftw_execute fftwl_execute
# define fftw_destroy_plan fftwl_destroy_plan
#elif USE_FLOAT
# define fftw_plan fftwf_plan
# define fftw_plan_dft_r2c fftwf_plan_dft_r2c
# define fftw_plan_dft_c2r fftwf_plan_dft_c2r
# define fftw_complex fftwf_complex
# define fftw_plan_many_dft_r2c fftwf_plan_many_dft_r2c
# define fftw_plan_many_dft_c2r fftwf_plan_many_dft_c2r
# define fftw_execute fftwf_execute
# define fftw_destroy_plan fftwf_destroy_plan
#endif

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
	Vector_d freePropagator_x;
	Vector_d freePropagator_x_b;

	Matrix_d positionSpace; // current matrix in position space
	Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> momentumSpace;

	Vector_d v_x;
	Vector_cd v_p;

	fftw_plan x2p_vec;
	fftw_plan p2x_vec;

	fftw_plan x2p_col;
	fftw_plan p2x_col;

	fftw_plan x2p_row;
	fftw_plan p2x_row;

	double plog;

	mymeasurement<double> m_dens;
	mymeasurement<double> m_magn;

	int thermalization_sweeps;
	int total_sweeps;
	bool reset;
	int reweight;
	int decompositions;
	std::string outfn;
	//std::ofstream logfile;

	Matrix_d U_s;
	Matrix_d U_s_inv;
	Vector_cd ev_s;

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

	mymeasurement<double> density;
	mymeasurement<double> magnetization;
	mymeasurement<double> kinetic;
	mymeasurement<double> interaction;
	std::vector<mymeasurement<double>> spincorrelation;

	int shift_x (int x, int k) {
		int a = (x/Ly/Lz)%Lx;
		int b = x%(Ly*Lz);
		return ((a+k)%Lx)*Ly*Lz + b;
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
		v_x = Vector_d::Zero(V);
		v_p = Vector_cd::Zero(V);
		positionSpace = Matrix_d::Identity(V, V);
		momentumSpace.setIdentity(V, V);

		const int size[] = { Lx, Ly, Lz, };
		x2p_vec = fftw_plan_dft_r2c(3, size, v_x.data(), reinterpret_cast<fftw_complex*>(v_p.data()), FFTW_PATIENT);
		p2x_vec = fftw_plan_dft_c2r(3, size, reinterpret_cast<fftw_complex*>(v_p.data()), v_x.data(), FFTW_PATIENT);
		x2p_col = fftw_plan_many_dft_r2c(3, size, V, positionSpace.data(),
				NULL, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, 1, V, FFTW_PATIENT);
		p2x_col = fftw_plan_many_dft_c2r(3, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, 1, V, positionSpace.data(), NULL, 1, V, FFTW_PATIENT);
		x2p_row = fftw_plan_many_dft_r2c(3, size, V, positionSpace.data(),
				NULL, V, 1, reinterpret_cast<fftw_complex*>(momentumSpace.data()), NULL, V, 1, FFTW_PATIENT);
		p2x_row = fftw_plan_many_dft_c2r(3, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
				NULL, V, 1, positionSpace.data(), NULL, V, 1, FFTW_PATIENT);

		positionSpace = Matrix_d::Identity(V, V);
		momentumSpace.setIdentity(V, V);

		U_s = Matrix_d::Identity(V, V);
		U_s_inv = Matrix_d::Identity(V, V);

		energies = Vector_d::Zero(V);
		freePropagator = Vector_d::Zero(V);
		freePropagator_b = Vector_d::Zero(V);
		for (int i=0;i<V;i++) {
			energies[i] += -2.0 * ( tx * cos(2.0*((i/Ly/Lz)%Lx)*pi/Lx) + ty * cos(2.0*((i/Lz)%Ly)*pi/Ly) + tz * cos(2.0*(i%Lz)*pi/Lz) );
			freePropagator[i] = exp(-dt*energies[i]);
			freePropagator_b[i] = exp(dt*energies[i]);
		}

		accumulate_forward();
		U_s = positionSpace;
		accumulate_backward();
		U_s_inv = positionSpace;
		plog = -1.0e-10;

		for (int i=0;i<=Lx/2;i++) {
			spincorrelation.push_back(mymeasurement<double>());
		}
	}

	Simulation (lua_State *L, int index, int seed = 42) : distribution(0.5), trialDistribution(1.0), steps(0) {
		lua_getfield(L, index, "THERMALIZATION"); thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
		lua_getfield(L, index, "SWEEPS"); total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
		lua_getfield(L, index, "SEED"); generator.seed(lua_tointeger(L, -1)+seed); lua_pop(L, 1);
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
		lua_getfield(L, index, "RESET");  reset = lua_toboolean(L, -1);            lua_pop(L, 1);
		lua_getfield(L, index, "REWEIGHT");  reweight = lua_tointeger(L, -1);      lua_pop(L, 1);
		lua_getfield(L, index, "OUTPUT");  outfn = lua_tostring(L, -1);            lua_pop(L, 1);
		//lua_getfield(L, index, "LOGFILE");  logfile.open(lua_tostring(L, -1));     lua_pop(L, 1);
		lua_getfield(L, index, "DECOMPOSITIONS");  decompositions = lua_tointeger(L, -1);     lua_pop(L, 1);
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
		positionSpace.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			positionSpace.applyOnTheLeft((Vector_d::Constant(V, 1.0)+diagonals[i]).asDiagonal());
			fftw_execute(x2p_col);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x_col);
			positionSpace /= V;
		}
	}

	void accumulate_backward (int start = 0, int end = -1) {
		Real X = 1.0 - A*A;
		positionSpace.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			positionSpace.applyOnTheRight((Vector_d::Constant(V, 1.0)-diagonals[i]).asDiagonal());
			fftw_execute(x2p_row);
			momentumSpace.applyOnTheRight(freePropagator_b.asDiagonal());
			fftw_execute(p2x_row);
			positionSpace /= V*X;
		}
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
		Vector_d S;
		Matrix_d U;
		Matrix_d V;
		U = cache.svd.U.transpose()*cache.svd.V;
		U.diagonal() += std::exp(+beta*B*0.5+beta*mu)*cache.svd.S;
		dgesvd(U, S, U, V); // 1+U*S*V^t -> (V + U*S) V^t -> U (U^t*V + S) V^t
		U = cache.svd.U.transpose()*cache.svd.V;
		ret += S.array().log().sum();
		U.diagonal() += std::exp(-beta*B*0.5+beta*mu)*cache.svd.S;
		dgesvd(U, S, U, V); // 1+U*S*V^t -> (V + U*S) V^t -> U (U^t*V + S) V^t
		//collapseSVD(bvec, cache.svd.S, cache.svd.U, cache.svd.V);
		ret += S.array().log().sum();

		if (std::cos(ret.imag())<0.99 || std::isnan(ret.real()) || std::isnan(ret.imag())) {
			std::cerr << "prob_complex = " << ret << " det=" << cache.svd.S.array().log().sum() << std::endl;
			throw "";
		}
		return ret.real();
	}

	void compute_uv_f (int x, int t) {
		v_x = Vector_d::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<N;i++) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonals[i]).array();
			v_x /= V;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.u = v_x;
		v_x = Vector_d::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=0;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonals[i]).array();
			v_x /= V;
		}
		cache.v = -2*diagonals[t][x]*v_x;
	}

	void compute_uv_b (int x, int t) {
		Real X = 1-A*A;
		v_x = Vector_d::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<N;i++) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array();
			v_x /= V*X;
		}
		fftw_execute(x2p_vec);
		v_p = v_p.array() * freePropagator_b.array();
		fftw_execute(p2x_vec);
		v_x /= V;
		cache.v_inv = v_x;
		v_x = Vector_d::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=0;i--) {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator_b.array();
			fftw_execute(p2x_vec);
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)-diagonals[i]).array();
			v_x /= V*X;
		}
		cache.u_inv = +2*diagonals[t][x]/X*v_x;
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
		int x = randomPosition(generator);
		int t = randomTime(generator);
		double exact = logDetU_s(x, t);
		double trial = rank1prob(x, t);
		Complex c =  cache.ev.array().log().sum();

		Vector_cd ev1, ev2, ev3;

		//ev1 = cache.ev;
		//logfile << exact << ' ' << cache.ev.array().log().sum().real() << ' ' << std::norm(cache.ev[0]/cache.ev[V-1]);

		if ( std::cos(c.imag())<0.99 || std::abs(1.0-c.real()/exact)>1.0e-5 ) {
			std::cerr << " recomputing exact = " << exact << " trial=" << c;
			accumulate_forward();
			U_s = positionSpace;
			accumulate_backward();
			U_s_inv = positionSpace;
			trial = rank1prob(x, t);
			c = cache.ev.array().log().sum();
			std::cerr << " new =" << c << std::endl;
			std::cerr << "CN = " << cache.ev[0]/cache.ev[V-1] << std::endl;
		}
		//ev2 = cache.ev;
		//logfile << ' ' << cache.ev.array().log().sum().real() << ' ' << std::norm(cache.ev[0]/cache.ev[V-1]);

		if ( std::cos(c.imag())<0.99 || std::abs(1.0-c.real()/exact)>1.0e-4 ) {
		}
		//ev3 = cache.ev;
		//logfile << ' ' << cache.ev.array().log().sum().real() << ' ' << std::norm(cache.ev[0]/cache.ev[V-1]);
		//logfile << std::endl;

		if (std::cos(c.imag())<0.99) {
			std::cerr << exact << ' ' << ev1.array().log().sum().real() << ' ' << ev2.array().log().sum().real() << ' ' << ev3.array().log().sum().real() << std::endl;
			diagonals[t][x] = -diagonals[t][x];
			c = logProbability_complex();
			diagonals[t][x] = -diagonals[t][x];
		}
		//std::cerr << "exact " << exact << "\nrank1 " << c << std::endl;
		//diagonals[t][x] = -diagonals[t][x];
		//double other = logProbability_complex();
		//diagonals[t][x] = -diagonals[t][x];
		//std::cerr << "probabilities: " << trial << " <-> " << other << std::endl;
		if (-trialDistribution(generator)<trial-plog) {
			plog = trial;
			diagonals[t][x] = -diagonals[t][x];
			U_s = U_s+cache.u*cache.v.transpose();
			U_s_inv = U_s_inv+cache.u_inv*cache.v_inv.transpose();
			ev_s = cache.ev;
			ret = true;
		} else {
			ret = false;
		}
		return ret;
	}

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		for (int i=0;i<10;i++) metropolis();
		if (steps%1000==0) {
			accumulate_forward();
			U_s = positionSpace;
			accumulate_backward();
			U_s_inv = positionSpace;
		}
	}

	void extract_data (const Matrix_d &M, double &K) {
		positionSpace = M;
		//d = positionSpace.diagonal();
		//d1.resize(positionSpace.rows());
		//d2.resize(positionSpace.rows());
		// get super- and sub- diagonal
		for (int i=0;i<V;i++) {
			//d1[i] = positionSpace(i, (i+1)%V);
			//d2[i] = positionSpace((i+1)%V, i);
		}
		fftw_execute(x2p_col);
		momentumSpace.applyOnTheLeft(energies.asDiagonal());
		fftw_execute(p2x_col);
		K = positionSpace.trace() / V;
	}

	void measure () {
		double K_up, K_dn;
		double n_up, n_dn, n2;
		{
			//extract_data(Matrix_d::Identity(V, V) - (Matrix_d::Identity(V, V) + exp(+beta*B*0.5+beta*mu)*U_s).inverse(), d_up, d1_up, d2_up, K_up);
			Matrix_d rho_up = (Matrix_d::Identity(V, V) + exp(-beta*B*0.5-beta*mu)*U_s_inv).inverse();
			extract_data(rho_up, K_up);
			//extract_data(Matrix_d::Identity(V, V) - (Matrix_d::Identity(V, V) + exp(-beta*B*0.5+beta*mu)*U_s).inverse(), d_dn, d1_dn, d2_dn, K_dn);
			Matrix_d rho_dn = (Matrix_d::Identity(V, V) + exp(-beta*B*0.5+beta*mu)*U_s).inverse();
			extract_data(rho_dn, K_dn);
			n_up = rho_up.diagonal().array().sum();
			n_dn = rho_dn.diagonal().array().sum();
			n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
			density.add((n_up + n_dn) / V);
			magnetization.add((n_up - n_dn) / 2.0 / V);
			kinetic.add(K_up-K_dn);
			interaction.add(g*n2);
			//- (d1_up*d2_up).sum() - (d1_dn*d2_dn).sum();
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
				if (isnan(ssz)) {
					std::cerr << "explain:" << std::endl;
					std::cerr << "k=" << k << " ssz=" << ssz << std::endl;
					for (int j=0;j<V;j++) {
						int x = j;
						int y = shift_x(j, k);
						std::cerr << " j=" << j
							<< " a_j=" << (rho_up(x, x)*rho_up(y, y) + rho_dn(x, x)*rho_dn(y, y))
							<< " b_j=" << (rho_up(x, x)*rho_dn(y, y) + rho_dn(x, x)*rho_up(y, y))
							<< " c_j=" << (rho_up(x, y)*rho_up(y, x) + rho_dn(x, y)*rho_dn(y, x)) << std::endl;
					}
					throw "";
				}
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
			<< ' ' << kinetic.mean()/tx/V << ' ' << kinetic.variance()
			<< ' ' << interaction.mean()/tx/V << ' ' << interaction.variance();
		for (int i=1;i<=Lx/2;i++)
			out << ' ' << spincorrelation[i].mean()/V << ' ' << spincorrelation[i].variance();
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

int main (int argc, char **argv) {
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	if (luaL_dofile(L, argv[1])) {
		std::cerr << "Error loading configuration file \"" << argv[1] << "\":" << std::endl;
		std::cerr << '\t' << lua_tostring(L, -1) << std::endl;
		return -1;
	}

	int nthreads = 1;
	lua_getfield(L, -1, "THREADS");
	if (lua_tointeger(L, -1)) {
		nthreads = lua_tointeger(L, -1);
	}
	lua_pop(L, 1);
	std::vector<std::thread> threads(nthreads);
	Logger log;
	std::mutex lock;
	std::atomic<int> failed;
	failed = 0;
	for (int j=0;j<nthreads;j++) {
		int i = 1;
		threads[j] = std::thread( [=, &log, &lock, &i, &failed] () {
				log << "thread" << j << "starting";
				while (true) {
					lock.lock();
					lua_rawgeti(L, -1, i);
					if (lua_isnil(L, -1)) {
						log << "thread" << j << "terminating";
						lua_pop(L, 1);
						lock.unlock();
						break;
					}
					log << "thread" << j << "running simulation" << i;
					lua_getfield(L, -1, "THERMALIZATION"); int thermalization_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
					lua_getfield(L, -1, "SWEEPS"); int total_sweeps = lua_tointeger(L, -1); lua_pop(L, 1);
					Simulation simulation(L, -1, i);
					lua_pop(L, 1);
					i++;
					lock.unlock();
					try {
						for (int i=0;i<thermalization_sweeps;i++) {
							if (i%1000==0) { log << "thread" << j << "thermalizing: " << i << '/' << thermalization_sweeps; }
							simulation.update();
						}
						log << "thread" << j << "thermalized";
						for (int i=0;i<total_sweeps;i++) {
							if (i%1000==0) { log << "thread" << j << "running: " << i << '/' << total_sweeps; }
							simulation.update();
							simulation.measure();
						}
						log << "thread" << j << "finished simulation" << i; // i has changed!
						lock.lock();
						simulation.output_results();
						lock.unlock();
					} catch (...) {
						failed++;
						log << "thread" << j << "caught exception in simulation" << i;
						std::cerr << " with params " << simulation.params() << std::endl;
					}
				}
		});
	}
	for (std::thread& t : threads) t.join();

	std::cout << failed << " tasks failed" << std::endl;

	lua_close(L);
	return 0;
}

