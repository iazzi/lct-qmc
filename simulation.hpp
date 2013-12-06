#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "config.hpp"

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

#if 0
static auto measurements_proto = make_named_tuple(
		named_value2(mymeasurement<double>(), acceptance),
		named_value2(mymeasurement<double>(), density),
		named_value2(mymeasurement<double>(), magnetization),
		named_value2(mymeasurement<double>(), order_parameter),
		named_value2(mymeasurement<double>(), chi_d),
		named_value2(mymeasurement<double>(), chi_af),
		named_value2(mymeasurement<double>(), kinetic),
		named_value2(mymeasurement<double>(), interaction),
		named_value2(mymeasurement<double>(), sign),
		named_value2(mymeasurement<double>(), measured_sign),
		named_value2(mymeasurement<double>(), sign_correlation)
		);
};

typedef decltype(measurements_proto) measurements_type;
#endif

class Simulation {
	private:

	// Model parameters
	config::hubbard_config config;
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
	//double Vx, Vy, Vz; // trap strength
	//double staggered_field;


	//state
	std::vector<Vector_d> diagonals;
	std::vector<Vector_d> diagonals_saved;

	// Monte Carlo scheme settings
	std::mt19937_64 generator;
	bool reset;
	std::string outfn;
	int mslices;
	int msvd;
	int flips_per_update;
	int max_update_size;
	bool open_boundary;


	// RNG distributions
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::uniform_int_distribution<int> randomStep;
	std::exponential_distribution<double> trialDistribution;

	Vector_d energies;
	Vector_d freePropagator;
	Vector_d freePropagator_b;
	Matrix_d freePropagator_open;
	//Vector_d potential;
	//Vector_d freePropagator_x;
	//Vector_d freePropagator_x_b;
	Array_d staggering;

	Matrix_d positionSpace; // current matrix in position space
	Matrix_cd positionSpace_c; // current matrix in position space
	Matrix_cd momentumSpace;

	std::vector<Matrix_d> slices;

	double update_prob;
	double update_sign;
	int update_size;
	Matrix_d update_U;
	Matrix_d update_Vt;

	Matrix_d hamiltonian;

	public:

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
	//mymeasurement<double> measured_sign;
	//mymeasurement<double> sign_correlation;
	mymeasurement<double> exact_sign;
	std::vector<mymeasurement<double>> d_up;
	std::vector<mymeasurement<double>> d_dn;
	std::vector<mymeasurement<double>> spincorrelation;
	std::vector<mymeasurement<double>> error;
	// RNG distributions
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
		//measured_sign.set_name("Sign (Measurements)");
		//sign_correlation.set_name("Sign Correlation");
		exact_sign.set_name("Sign (Exact)");
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

	void init ();

	void load (lua_State *L, int index);
	void save (lua_State *L, int index);
	void load_checkpoint (lua_State *L);
	void save_checkpoint (lua_State *L);

	Simulation (lua_State *L, int index) : distribution(0.5), trialDistribution(1.0), steps(0) {
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
		slices.resize(N/mslices + ((N%mslices>0)?1:0));
		for (int i=0;i<N;i+=mslices) {
			accumulate_forward(i, i+mslices);
			slices[i/mslices] = positionSpace;
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
		first_slice_inverse = slices[0].inverse();
		if (true) {
			make_density_matrices();
			svd_inverse_up = svdA;
			svd_inverse_up.invertInPlace();
			svd_inverse_dn = svdB;
			svd_inverse_dn.invertInPlace();
		} else {
			svd_inverse = svd;
			svd_inverse.invertInPlace();
			svd_inverse_up = svd_inverse;
			svd_inverse_up.add_identity(std::exp(-beta*B*0.5-beta*mu));
			svd_inverse_up.invertInPlace();
			svd_inverse_dn = svd_inverse;
			svd_inverse_dn.add_identity(std::exp(+beta*B*0.5-beta*mu));
			svd_inverse_dn.invertInPlace();
		}
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
			positionSpace_c.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
			apply_propagator_matrix();
		}
		positionSpace = positionSpace_c.real();
	}

	void compute_uv_f_short (int x, int t) {
		int start = mslices*(t/mslices);
		int end = mslices*(1+t/mslices);
		if (end>N) end = N;
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t+1;i<end;i++) {
			apply_propagator_vector();
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array();
		}
		apply_propagator_vector();
		cache.u_smart = (-2*diagonal(t)[x]*v_x).real();
		v_x = Vector_cd::Zero(V);
		v_x[x] = 1.0;
		for (int i=t-1;i>=start;i--) {
			apply_propagator_vector();
			v_x = v_x.array() * (Vector_d::Constant(V, 1.0)+diagonal(i)).array();
		}
		cache.v_smart = v_x.real();
	}

	void flip (int t, int x) {
		diagonal(t)[x] = -diagonal(t)[x];
	}

	void flip (int t, const std::vector<int> &vec) {
		for (int x : vec) {
			diagonal(t)[x] = -diagonal(t)[x];
		}
	}

	void redo_all () {
		make_slices();
		make_svd();
		make_svd_inverse();
		make_density_matrices();
		double np = svd_probability();
		double ns = svd_sign();
		if (fabs(np-plog-update_prob)>1.0e-8 || psign*update_sign!=ns) {
		//if (psign*update_sign!=ns && false) {
			std::cerr << plog+update_prob << " <> " << np << " ~~ " << np-plog-update_prob << '\t' << (psign*update_sign*ns) << std::endl;
			plog = np;
			psign = ns;
			if (false && ns!=recheck().second && false) {
				int old_msvd = msvd;
				do {
					make_svd();
					make_svd_inverse();
					make_density_matrices();
					double np = svd_probability();
					double ns = svd_sign();
					std::cout << (ns>0.0?"+exp(":"-exp(") << np << ") ";
				} while (--msvd>20);
				std::cout << std::endl;
				msvd = old_msvd;
			}
			//std::cerr << "    " << np-plog << " ~~ " << update_prob << std::endl;
		}
		plog = np;
		psign = ns;
		if (isnan(plog)) {
			std::cerr << "NaN found: restoring" << std::endl;
			diagonals = diagonals_saved;
			make_slices();
			make_svd();
			make_svd_inverse();
			make_density_matrices();
			plog = svd_probability();
			psign = svd_sign();
		} else {
			diagonals_saved = diagonals;
		}
		//recheck();
		reset_updates();
	}

	std::pair<double, double> rank1_probability (int x, int t);

	double ising_energy (int x, int t);
	bool anneal_ising ();
	bool metropolis_ising ();
	bool metropolis ();

	void set_time_shift (int t) { time_shift = t%N; redo_all(); }
	bool shift_time () { 
		time_shift += 1*mslices;
		bool ret = time_shift>=N;
		if (ret) time_shift -= N;
		redo_all();
		return ret;
	}

	void test_wrap () {
		time_shift = 0;
		redo_all();
	}

	void load_sigma (lua_State *L, const char *fn);

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		for (int i=0;i<flips_per_update;i++) {
			collapse_updates();
			acceptance.add(metropolis()?1.0:0.0);
		}
		//time_shift = randomTime(generator);
		//redo_all();
		shift_time();
	}

	bool collapse_updates () {
		if (update_size>=max_update_size) {
			plog += update_prob;
			psign *= update_sign;
			make_svd();
			make_svd_inverse();
			reset_updates();
			return true;
		} else {
			return false;
		}
	}

	void update_ising () {
		for (int i=0;i<flips_per_update;i++) {
			collapse_updates();
			metropolis_ising();
		}
		time_shift = randomTime(generator);
		redo_all();
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



	void measure ();
	void measure_sign ();
	int volume () { return V; }
	int timeSlices () { return N; }

	void write_wavefunction (std::ostream &out);

	void output_sign () {
		std::ostringstream buf;
		buf << outfn << "_sign.dat";
		std::ofstream out(buf.str(), reset?std::ios::trunc:std::ios::app);
		out << "# " << params();
		out << 1.0/(beta*tx) << ' ' << 0.5*(B+g)/tx;
		//out << ' ' << measured_sign.mean() << ' ' << measured_sign.error();
		//out << ' ' << sign_correlation.mean() << ' ' << sign_correlation.error();
		//out << ' ' << exact_sign.mean() << ' ' << exact_sign.variance();
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
		out << std::endl << std::endl;
	}

	void output_results () {
		std::ostringstream buf;
		buf << outfn << "stablefast_U" << (g/tx) << "_T" << 1.0/(beta*tx) << '_' << Lx << 'x' << Ly << 'x' << Lz << ".dat";
		outfn = buf.str();
		std::ofstream out(buf.str(), reset?std::ios::trunc:std::ios::app);
		out << 1.0/(beta*tx) << ' ' << 0.5*(B+g)/tx
			<< ' ' << density.mean() << ' ' << density.error()
			<< ' ' << magnetization.mean() << ' ' << magnetization.error()
			//<< ' ' << acceptance.mean() << ' ' << acceptance.variance()
			<< ' ' << kinetic.mean() << ' ' << kinetic.error()
			<< ' ' << interaction.mean() << ' ' << interaction.error();
		out << ' ' << order_parameter.mean() << ' ' << order_parameter.error();
		out << ' ' << chi_af.mean() << ' ' << chi_af.error();
		//out << ' ' << chi_d.mean() << ' ' << chi_d.error();
		out << ' ' << exact_sign.mean() << ' ' << exact_sign.error();
		out << ' ' << sign.mean() << ' ' << sign.error();
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
		buf << config << std::endl;
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

	std::pair<double, double> recheck ();
	void straighten_slices ();

	protected:
};


#endif // SIMULATION_HPP

