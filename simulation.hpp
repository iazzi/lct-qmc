#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "svd.hpp"
#include "types.hpp"
#include "measurements.hpp"

#include <fstream>
#include <random>
#include <iostream>

extern "C" {
#include <fftw3.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

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
	bool open_boundary;
	std::vector<Vector_d> diagonals;

	//int update_start;
	//int update_end;

	std::mt19937_64 generator;
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::uniform_int_distribution<int> randomStep;
	std::exponential_distribution<double> trialDistribution;

	Vector_d energies;
	Vector_d freePropagator;
	Vector_d freePropagator_b;
	Matrix_d freePropagator_open;
	Vector_d potential;
	Vector_d freePropagator_x;
	Vector_d freePropagator_x_b;
	Array_d staggering;

	Matrix_d positionSpace; // current matrix in position space
	Matrix_cd positionSpace_c; // current matrix in position space
	Matrix_cd momentumSpace;

	int mslices;
	std::vector<Matrix_d> slices;
	int flips_per_update;

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
	double psign;

	bool reset;
	//int reweight;
	std::string outfn;
	//std::ofstream logfile;

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

	int steps;

	mymeasurement<double> acceptance;
	mymeasurement<double> density;
	mymeasurement<double> magnetization;
	mymeasurement<double> order_parameter;
	mymeasurement<double> chi_d;
	mymeasurement<double> chi_af;
	//measurement<double, false> magnetization_slow;
	mymeasurement<double> kinetic;
	mymeasurement<double> interaction;
	mymeasurement<double> sign;
	mymeasurement<double> measured_sign;
	std::vector<mymeasurement<double>> d_up;
	std::vector<mymeasurement<double>> d_dn;
	std::vector<mymeasurement<double>> spincorrelation;
	std::vector<mymeasurement<double>> error;
	mymeasurement<double> staggered_magnetization;

	int time_shift;

	int shift_x (int x, int k) {
		int a = (x/Ly/Lz)%Lx;
		int b = x%(Ly*Lz);
		return ((a+k+Lx)%Lx)*Ly*Lz + b;
	}

	int shift_y (int y, int k) {
		int a = (y/Lz)%Ly;
		int b = y-a*Lz;
		return ((a+k+Ly)%Ly)*Lz + b;
	}

	Vector_d& diagonal (int t) {
		return diagonals[(t+time_shift)%N];
	}

	const Vector_d& diagonal (int t) const {
		return diagonals[(t+time_shift)%N];
	}

	public:

	void prepare_propagators ();
	void prepare_open_boundaries ();

	void init_measurements () {
		sign.set_name("Sign");
		acceptance.set_name("Acceptance");
		density.set_name("Density");
		magnetization.set_name("Magnetization");
		order_parameter.set_name("Order Parameter");
		chi_d.set_name("Chi (D-wave)");
		chi_af.set_name("Chi (AF)");
		measured_sign.set_name("Sign (Measurements)");
		//magnetization_slow.set_name("Magnetization (slow)");
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
		if (max_update_size<1) max_update_size = 1;
		if (flips_per_update<1) flips_per_update = max_update_size;
		randomPosition = std::uniform_int_distribution<int>(0, V-1);
		randomTime = std::uniform_int_distribution<int>(0, N-1);
		randomStep = std::uniform_int_distribution<int>(0, mslices-1);
		dt = beta/N;
		A = sqrt(exp(g*dt)-1.0);
		diagonals.insert(diagonals.begin(), N, Vector_d::Zero(V));
		for (size_t i=0;i<diagonals.size();i++) {
			for (int j=0;j<V;j++) {
				diagonals[i][j] = distribution(generator)?A:-A;
				//diagonals[i][j] = i<N/4.9?-A:A;
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

		positionSpace.setIdentity(V, V);
		momentumSpace.setIdentity(V, V);

		prepare_propagators();
		prepare_open_boundaries();

		make_slices();
		make_svd();
		make_svd_inverse();
		make_density_matrices();
		plog = svd_probability();
		psign = svd_sign();

		init_measurements();
		reset_updates();
	}

	void load (lua_State *L, int index);

	void save (lua_State *L, int index);

	Simulation (lua_State *L, int index) : distribution(0.8), trialDistribution(1.0), steps(0) {
		load(L, index);
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
		//std::cerr << svd.S.array().log().sum() << ' ' << logDetU_s() << std::endl;
		//std::cerr << svd.S.transpose() << std::endl;
		//std::cerr << svd.U << std::endl << std::endl;
		//std::cerr << svd.Vt << std::endl << std::endl;
	}

	void make_density_matrices () {
		svdA = svd;
		svdA.add_identity(std::exp(+beta*B*0.5+beta*mu));
		svdB = svd;
		svdB.add_identity(std::exp(-beta*B*0.5+beta*mu));
	}

	void make_svd_inverse () {
		svd_inverse = svd;
		svd_inverse.invertInPlace();
		svd_inverse_up = svd_inverse;
		svd_inverse_up.add_identity(std::exp(-beta*B*0.5-beta*mu));
		svd_inverse_up.invertInPlace();
		svd_inverse_dn = svd_inverse;
		svd_inverse_dn.add_identity(std::exp(+beta*B*0.5-beta*mu));
		svd_inverse_dn.invertInPlace();
		first_slice_inverse = slices[0].inverse();
	}

	double svd_probability () {
		double ret = svdA.S.array().log().sum() + svdB.S.array().log().sum();
		//std::cerr << svd.S.transpose() << std::endl;
		return ret; // * (svdA.U*svdA.Vt*svdB.U*svdB.Vt).determinant();
	}

	double svd_sign () {
		return (svdA.U*svdA.Vt*svdB.U*svdB.Vt).determinant()>0.0?1.0:-1.0;
	}

	void apply_propagator_matrix () {
		if (open_boundary) {
			positionSpace_c.applyOnTheLeft(freePropagator_open);
		} else {
			fftw_execute(x2p_col);
			momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
			fftw_execute(p2x_col);
			positionSpace_c /= V;
		}
	}

	void apply_propagator_vector () {
		if (open_boundary) {
			v_x = freePropagator_open * v_x;
		} else {
			fftw_execute(x2p_vec);
			v_p = v_p.array() * freePropagator.array();
			fftw_execute(p2x_vec);
			v_x /= V;
		}
	}

	void accumulate_forward (int start = 0, int end = -1) {
		positionSpace_c.setIdentity(V, V);
		end = end<0?N:end;
		end = end>N?N:end;
		for (int i=start;i<end;i++) {
			//std::cerr << "accumulate_f. " << i << " determinant = " << positionSpace_c.determinant() << std::endl;
			positionSpace_c.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()*freePropagator_x.array()).matrix().asDiagonal());
			apply_propagator_matrix();
		}
		positionSpace = positionSpace_c.real();
	}

	//void accumulate_backward (int start = 0, int end = -1) {
		//Real X = 1.0 - A*A;
		//positionSpace_c.setIdentity(V, V);
		//end = end<0?N:end;
		//end = end>N?N:end;
		//for (int i=start;i<end;i++) {
			//positionSpace_c.applyOnTheRight(((Vector_d::Constant(V, 1.0)-diagonal(i)).array()*freePropagator_x_b.array()).matrix().asDiagonal());
			//fftw_execute(x2p_row);
			//momentumSpace.applyOnTheRight(freePropagator_b.asDiagonal());
			//fftw_execute(p2x_row);
			//positionSpace_c /= V*X;
		//}
		//positionSpace = positionSpace_c.real();
	//}

	//void compute_uv_f (int x, int t) {
		//v_x = Vector_cd::Zero(V);
		//v_x[x] = 1.0;
		//for (int i=t+1;i<N;i++) {
			//apply_propagator_vector();
			//v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
		//}
		//apply_propagator_vector();
		//cache.u = (-2*diagonal(t)[x]*v_x*freePropagator_x[x]).real();
		//v_x = Vector_cd::Zero(V);
		//v_x[x] = 1.0;
		//for (int i=t-1;i>=0;i--) {
			//apply_propagator_vector();
			//v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
		//}
		//cache.v = v_x.real();
	//}

	void compute_uv_f_short (int x, int t) {
		int start = mslices*(t/mslices);
		int end = mslices*(1+t/mslices);
		if (end>N) end = N;
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<end;i++) {
			apply_propagator_vector();
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
		}
		apply_propagator_vector();
		cache.u_smart = (-2*diagonal(t)[x]*v_x*freePropagator_x[x]).real();
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=start;i--) {
			apply_propagator_vector();
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
		}
		cache.v_smart = v_x.real();
	}

	//void compute_uv_f_smart (int x, int t) {
		//int start = mslices*(t/mslices);
		//int end = mslices*(1+t/mslices);
		//if (end>N) end = N;
		//v_x = Vector_cd::Zero(V);
		//v_x[x] = 1.0;
		//for (int i=t+1;i<end;i++) {
			//apply_propagator_vector();
			//v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
		//}
		//apply_propagator_vector();
		//cache.u_smart = cache.u = (-2*diagonal(t)[x]*v_x*freePropagator_x[x]).real();
		//for (size_t i=t/mslices+1;i<slices.size();i++) {
			////std::cerr << i << ' ' << t/mslices << ' ' << slices.size() << std::endl;
			//cache.u.applyOnTheLeft(slices[i]);
		//}
		//v_x = Vector_cd::Zero(V);
		//v_x[x] = 1.0;
		//for (int i=t-1;i>=start;i--) {
			//apply_propagator_vector();
			//v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array() * freePropagator_x.array();
		//}
		//cache.v_smart = cache.v = v_x.real();
		//for (int i=t/mslices-1;i>=0;i--) {
			//cache.v.applyOnTheLeft(slices[i].transpose());
		//}
	//}

	void flip (int t, int x) {
		diagonal(t)[x] = -diagonal(t)[x];
	}

	void flip (int t, const std::vector<int> &vec) {
		for (int x : vec) {
			diagonal(t)[x] = -diagonal(t)[x];
		}
	}

	void redo_all () {
		//std::cerr << "redoing " << svd.S.array().log().sum() << std::endl;
		//make_slices();
		//int old_msvd = msvd;
		//msvd = 1;
		make_svd();
		//msvd = old_msvd;
		make_svd_inverse();
		make_density_matrices();
		double np = svd_probability();
		if (fabs(np-plog-update_prob)>1.0e-8) {
			std::cerr << plog+update_prob << " <> " << np << " ~~ " << np-plog-update_prob << std::endl;
			std::cerr << "    " << np-plog << " ~~ " << update_prob << std::endl;
		}
		plog = np;
		psign = svd_sign();
		reset_updates();
	}

	std::pair<double, double> rank1_probability (int x, int t);

	void make_tests () {
	}

	bool metropolis ();

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		for (int i=0;i<flips_per_update;i++) {
			acceptance.add(metropolis()?1.0:0.0);
			sign.add(psign*update_sign);
			measured_sign.add(psign*update_sign);
			if (update_size>=max_update_size) {
				plog += update_prob;
				psign *= update_sign;
				make_svd();
				make_svd_inverse();
				reset_updates();
			}
			//make_tests();
		}
		time_shift = randomTime(generator);
		make_slices();
		redo_all();
		//time_shift = randomTime(generator);
		//make_slices();
		//redo_all();
		//std::cerr << "update finished" << std::endl;
		//std::ofstream out("error.dat");
		//for (int i=0;i<N;i++) {
			//if (error[i].samples()>0) out << i << ' ' << error[i].mean() << std::endl;
		//}
	}

	double get_kinetic_energy (const Matrix_d &M) {
		positionSpace_c = M.cast<Complex>();
		//fftw_execute(x2p_col);
		//momentumSpace.applyOnTheLeft(energies.asDiagonal());
		//fftw_execute(p2x_col);
		return positionSpace_c.real().trace() / V;
	}

	double pair_correlation (const Matrix_d& rho_up, const Matrix_d& rho_dn) {
		return 0.0;
		double ret = 0.0;
		for (int x=0;x<V;x++) {
			for (int y=0;y<V;y++) {
				double u = rho_up(x, y);
				double d = 0.0;
				d += rho_dn(shift_x(x, +1), shift_x(y, +1));
				d += rho_dn(shift_x(x, -1), shift_x(y, +1));
				d -= rho_dn(shift_y(x, +1), shift_x(y, +1));
				d -= rho_dn(shift_y(x, -1), shift_x(y, +1));
				d += rho_dn(shift_x(x, +1), shift_x(y, -1));
				d += rho_dn(shift_x(x, -1), shift_x(y, -1));
				d -= rho_dn(shift_y(x, +1), shift_x(y, -1));
				d -= rho_dn(shift_y(x, -1), shift_x(y, -1));
				d -= rho_dn(shift_x(x, +1), shift_y(y, +1));
				d -= rho_dn(shift_x(x, -1), shift_y(y, +1));
				d += rho_dn(shift_y(x, +1), shift_y(y, +1));
				d += rho_dn(shift_y(x, -1), shift_y(y, +1));
				d -= rho_dn(shift_x(x, +1), shift_y(y, -1));
				d -= rho_dn(shift_x(x, -1), shift_y(y, -1));
				d += rho_dn(shift_y(x, +1), shift_y(y, -1));
				d += rho_dn(shift_y(x, -1), shift_y(y, -1));
				ret += u*d;
			}
		}
		return ret / V / V;
	}



	void measure () {
		double s = svd_sign();
		rho_up = Matrix_d::Identity(V, V) - svdA.inverse();
		rho_dn = svdB.inverse();
		double K_up = get_kinetic_energy(rho_up);
		double K_dn = get_kinetic_energy(rho_dn);
		double n_up = rho_up.diagonal().array().sum();
		double n_dn = rho_dn.diagonal().array().sum();
		double op = (rho_up.diagonal().array()-rho_dn.diagonal().array()).square().sum();
		double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
		density.add(s*(n_up+n_dn)/V);
		magnetization.add(s*(n_up-n_dn)/2.0/V);
		//magnetization_slow.add(s*(n_up-n_dn)/2.0/V);
		order_parameter.add(op);
		kinetic.add(s*K_up-s*K_dn);
		interaction.add(s*g*n2/tx/V);
		//sign.add(svd_sign());
		//- (d1_up*d2_up).sum() - (d1_dn*d2_dn).sum();
		for (int i=0;i<V;i++) {
			d_up[i].add(s*rho_up(i, i));
			d_dn[i].add(s*rho_dn(i, i));
		}
		double d_wave_chi = 0.0;
		Matrix_d F_up = svdA.inverse();
		Matrix_d F_dn = Matrix_d::Identity(V, V) - svdB.inverse();
		const double dtau = beta/slices.size();
		for (const Matrix_d& U : slices) {
			//F_up.applyOnTheLeft(U*std::exp(+dtau*B*0.5+dtau*mu));
			//F_dn.applyOnTheLeft(U*std::exp(-dtau*B*0.5+dtau*mu));
			d_wave_chi += pair_correlation(F_up, F_dn);
		}
		chi_d.add(s*d_wave_chi*beta/slices.size());
		double af_ =((rho_up.diagonal().array()-rho_dn.diagonal().array())*staggering).sum()/double(V);
		chi_af.add(s*beta*af_*af_);
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
		if (staggered_field!=0.0) staggered_magnetization.add(s*(rho_up.diagonal().array()*staggering - rho_dn.diagonal().array()*staggering).sum()/V);
	}

	int volume () { return V; }
	int timeSlices () { return N; }

	void write_wavefunction (std::ostream &out);

	void output_results () {
		std::ostringstream buf;
		buf << outfn << "stablefast_U" << (g/tx) << "_T" << 1.0/(beta*tx) << '_' << Lx << 'x' << Ly << 'x' << Lz << ".dat";
		outfn = buf.str();
		std::ofstream out(outfn, reset?std::ios::trunc:std::ios::app);
		out << 1.0/(beta*tx) << ' ' << 0.5*(B+g)/tx
			<< ' ' << density.mean() << ' ' << density.variance()
			<< ' ' << magnetization.mean() << ' ' << magnetization.variance()
			//<< ' ' << acceptance.mean() << ' ' << acceptance.variance()
			<< ' ' << kinetic.mean()/tx/V << ' ' << kinetic.variance()/tx/tx/V/V
			<< ' ' << interaction.mean() << ' ' << interaction.variance();
		out << ' ' << order_parameter.mean() << ' ' << order_parameter.variance();
		out << ' ' << chi_af.mean() << ' ' << chi_af.variance();
		out << ' ' << chi_d.mean() << ' ' << chi_d.variance();
		out << ' ' << measured_sign.mean() << ' ' << measured_sign.variance();
		//if (staggered_field!=0.0) out << ' ' << -staggered_magnetization.mean()/staggered_field << ' ' << staggered_magnetization.variance();
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


#endif // SIMULATION_HPP

