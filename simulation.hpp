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
	std::string gf_name;
	int mslices;
	int msvd;
	int flips_per_update;
	bool open_boundary;


	// RNG distributions
	std::bernoulli_distribution distribution;
	std::uniform_int_distribution<int> randomPosition;
	std::uniform_int_distribution<int> randomTime;
	std::exponential_distribution<double> trialDistribution;

	Vector_d freePropagator;
	Matrix_d freePropagator_open;
	Matrix_d freePropagator_inverse;
	double w_x, w_y, w_z;
	Vector_d potential;
	Vector_d freePropagator_x;
	//Vector_d freePropagator_x_b;
	Array_d staggering;

	Matrix_d positionSpace; // current matrix in position space
	Matrix_cd momentumSpace; // current matrix in momentum space

	std::vector<Matrix_d> slices_up;
	std::vector<Matrix_d> slices_dn;
	std::vector<bool> valid_slices;

	double update_prob;
	double update_sign;
	int update_size;
	int new_update_size;
	//std::vector<bool> update_flips;
	std::vector<int> update_perm;
	Matrix_d update_matrix_up;
	Matrix_d update_matrix_dn;

	Matrix_d hamiltonian;
	Matrix_d eigenvectors;
	Array_d energies;

	public:

	Matrix_d plain;
	Matrix_d plainA;
	Matrix_d plainB;
	Eigen::ColPivHouseholderQR<Matrix_d> qr;

	SVDHelper svd;
	SVDHelper svdA;
	SVDHelper svdB;

	fftw_plan x2p_col;
	fftw_plan p2x_col;

	double plog;
	double psign;

	Matrix_d rho_up;
	Matrix_d rho_dn;

	public:

	int steps;

	mymeasurement<double> acceptance;
	mymeasurement<double> density;
	mymeasurement<double> magnetization;
	mymeasurement<double> singlet;
	mymeasurement<double> order_parameter;
	mymeasurement<double> chi_d;
	mymeasurement<double> chi_af;
	//measurement<double, false> magnetization_slow;
	mymeasurement<double> kinetic;
	mymeasurement<double> interaction;
	mymeasurement<double> sign;
	mymeasurement<double> measured_sign;
	//mymeasurement<double> sign_correlation;
	mymeasurement<double> exact_sign;
	std::vector<mymeasurement<double>> d_up;
	std::vector<mymeasurement<double>> d_dn;
	std::vector<mymeasurement<double>> spincorrelation;
	std::vector<mymeasurement<double>> error;
	// RNG distributions
	mymeasurement<double> staggered_magnetization;

	std::vector<mymeasurement<Eigen::ArrayXXd>> green_function_up;
	std::vector<mymeasurement<Eigen::ArrayXXd>> green_function_dn;

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

	const Vector_d& diagonal (int t) const { return diagonals[(t+time_shift)%N]; }
	Vector_d& diagonal (int t) { return diagonals[(t+time_shift)%N]; }

	public:

	void prepare_propagators ();
	void prepare_open_boundaries ();

	void init_measurements () {
		sign.set_name("Sign");
		acceptance.set_name("Acceptance");
		density.set_name("Density");
		magnetization.set_name("Magnetization");
		singlet.set_name("Singlet population");
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
		for (int i=0;i<V;i++) {
			spincorrelation.push_back(mymeasurement<double>());
		}
		for (int i=0;i<=N;i++) {
			error.push_back(mymeasurement<double>());
			green_function_up.push_back(mymeasurement<Eigen::ArrayXXd>());
			green_function_dn.push_back(mymeasurement<Eigen::ArrayXXd>());
		}
	}

	void reset_updates () {
		update_prob = 0.0;
		update_sign = 1.0;
		update_size = 0.0;
		update_perm.resize(V);
		for (int i=0;i<V;i++) update_perm[i] = i;
	}

	void init ();

	void load (lua_State *L, int index);
	void save (lua_State *L, int index);
	void load_checkpoint (lua_State *L);
	void save_checkpoint (lua_State *L);

	Simulation (lua_State *L, int index) : distribution(0.5), trialDistribution(1.0), steps(0) {
		load(L, index);
	}

	double logDetU_s (int x = -1, int t = -1) const {
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

	int nslices () const { return N/mslices + ((N%mslices>0)?1:0); }

	void make_slice (int i);
	void make_slices ();

	void make_svd () {
		svd.setIdentity(V);
		for (int i=0;i<N;) {
			svd.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
			if (false) {
				svd.U.applyOnTheLeft(freePropagator_open);
			} else {
				fftw_execute_dft_r2c(x2p_col, svd.U.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
				momentumSpace.applyOnTheLeft((freePropagator/double(V)).asDiagonal());
				fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), svd.U.data());
			}
			i++;
			if (i%msvd==0 || i==N) svd.absorbU();
		}
	}

	void make_plain () {
		plain.setIdentity(V, V);
		for (int i=0;i<N;) {
			plain.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
			if (false) {
				plain.applyOnTheLeft(freePropagator_open);
			} else {
				fftw_execute_dft_r2c(x2p_col, plain.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
				momentumSpace.applyOnTheLeft((freePropagator/double(V)).asDiagonal());
				fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), plain.data());
			}
			i++;
		}
	}

	void make_svd_double () {
		svdA.setIdentity(V);
		for (int i=0;i<N;) {
			svdA.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
			if (true) {
				svdA.U.applyOnTheLeft(freePropagator_open);
			} else {
				svdA.U.applyOnTheLeft(freePropagator_x.asDiagonal());
				fftw_execute_dft_r2c(x2p_col, svdA.U.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
				momentumSpace.applyOnTheLeft((freePropagator/double(V)).asDiagonal());
				fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), svdA.U.data());
			}
			i++;
			if (i%msvd==0 || i==N) svdA.absorbU();
		}
		svdB.setIdentity(V);
		for (int i=0;i<N;) {
			svdB.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
			if (true) {
				svdB.U.applyOnTheLeft(freePropagator_inverse);
			} else {
				svdB.U.applyOnTheLeft(freePropagator_x.array().inverse().matrix().asDiagonal());
				fftw_execute_dft_r2c(x2p_col, svdB.U.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
				momentumSpace.applyOnTheLeft((freePropagator.array().inverse().matrix()/double(V)).asDiagonal());
				fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), svdB.U.data());
			}
			i++;
			if (i%msvd==0 || i==N) svdB.absorbU();
		}
		svdA.add_identity(std::exp(+beta*B*0.5+beta*mu));
		svdB.add_identity(std::exp(-beta*B*0.5+beta*mu));
	}

	std::pair<double, double> make_density_matrices () {
		make_svd();
		svdA = svd;
		svdA.add_identity(std::exp(+beta*B*0.5+beta*mu));
		svdB = svd;
		svdB.add_identity(std::exp(-beta*B*0.5+beta*mu));
		return { svd_probability(), svd_sign() };
	}

	std::pair<double, double> make_svd_inverse () {
		std::pair<double, double> ret = make_density_matrices();
		rho_up = Matrix_d::Identity(V, V) - svdA.inverse();
		update_matrix_up = rho_up;
		update_matrix_up.applyOnTheLeft(-2.0*(diagonal(0).array().inverse()+1.0).inverse().matrix().asDiagonal());
		update_matrix_up.diagonal() += Vector_d::Ones(V);
		rho_dn = svdB.inverse();
		update_matrix_dn = Matrix_d::Identity(V, V) - rho_dn;
		update_matrix_dn.applyOnTheLeft(-2.0*(diagonal(0).array().inverse()+1.0).inverse().matrix().asDiagonal());
		update_matrix_dn.diagonal() += Vector_d::Ones(V);
		return ret;
	}

	std::pair<double, double> make_plain_inverse_second_step () {
		std::pair<double, double> ret = { 0.0, 0.0 };
		plainA = plain*std::exp(+beta*B*0.5+beta*mu) + Matrix_d::Identity(V, V);
		plainB = plain*std::exp(-beta*B*0.5+beta*mu) + Matrix_d::Identity(V, V);
		qr.compute(plainA);
		rho_up = Matrix_d::Identity(V, V) - qr.inverse();
		update_matrix_up = rho_up;
		update_matrix_up.applyOnTheLeft(-2.0*(diagonal(0).array().inverse()+1.0).inverse().matrix().asDiagonal());
		update_matrix_up.diagonal() += Vector_d::Ones(V);
		ret.first += qr.logAbsDeterminant();
		qr.compute(plainB);
		rho_dn = qr.inverse();
		update_matrix_dn = Matrix_d::Identity(V, V) - rho_dn;
		update_matrix_dn.applyOnTheLeft(-2.0*(diagonal(0).array().inverse()+1.0).inverse().matrix().asDiagonal());
		update_matrix_dn.diagonal() += Vector_d::Ones(V);
		ret.first += qr.logAbsDeterminant();
		ret.second = (plainA*plainB).determinant()<0.0?-1.0:1.0;
		return ret;
	}

	std::pair<double, double> make_plain_inverse () {
		make_plain();
		return make_plain_inverse_second_step();
	}

	double svd_probability () const {
		return svdA.S.array().log().sum() + svdB.S.array().log().sum();
	}

	double svd_sign () const {
		return (svdA.U*svdA.Vt*svdB.U*svdB.Vt).determinant()>0.0?1.0:-1.0;
	}

	void accumulate_forward (int start, int end, Matrix_d &G_up, Matrix_d &G_dn);

	void redo_all () {
		double np, ns;
		std::tie(np, ns) = make_svd_inverse();
		if (fabs(np-plog-update_prob)>1.0e-8 || psign*update_sign!=ns) {
			std::cerr << "redo " << plog+update_prob << " <> " << np << " ~~ " << np-plog-update_prob << '\t' << (psign*update_sign*ns) << std::endl;
			plog = np;
			psign = ns;
			//std::cerr << "    " << np-plog << " ~~ " << update_prob << std::endl;
		}
		plog = np;
		psign = ns;
		if (isnan(plog)) {
			std::cerr << "NaN found: restoring" << std::endl;
			//make_svd();
			std::tie(plog, psign) = make_svd_inverse();
			//make_density_matrices() // already called in make_svd_inverse;
		} else {
		}
		//recheck();
		reset_updates();
	}

	void apply_updates () {
		for (int i=0;i<update_size;i++) {
			int x = update_perm[i];
			diagonal(0)[x] = -diagonal(0)[x];
		}
	}

	std::pair<double, double> rank1_probability (int x);

	bool metropolis ();

	void remove_first_slice (Matrix_d &A) {
		A.applyOnTheRight((Vector_d::Constant(V, 1.0)+diagonal(0)).array().inverse().matrix().asDiagonal());
		A.transposeInPlace();
		fftw_execute_dft_r2c(x2p_col, A.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
		momentumSpace.applyOnTheLeft((freePropagator.array().inverse().matrix()/double(V)).asDiagonal());
		fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), A.data());
		A.transposeInPlace();
	}

	void queue_first_slice (Matrix_d &A) {
		A.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(0)).array()).matrix().asDiagonal());
		fftw_execute_dft_r2c(x2p_col, A.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
		momentumSpace.applyOnTheLeft((freePropagator.array().matrix()/double(V)).asDiagonal());
		fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), A.data());
	}

	void set_time_shift (int t) { time_shift = t%N; redo_all(); }
	bool shift_time () { 
		//std::cerr << plain.rows() << ' ' << plain.cols() << ' ' << (Vector_d::Constant(V, 1.0)+diagonal(0)).array().inverse().matrix().size() << std::endl;
		bool ret = time_shift==N-1;
		if (time_shift%5) {
			remove_first_slice(plain);
			apply_updates();
			queue_first_slice(plain);
			time_shift++;
			std::tie(plog, psign) = make_plain_inverse_second_step();
			reset_updates();
		} else {
			apply_updates();
			time_shift++;
			redo_all();
		}
		time_shift = time_shift%N;
		return ret;
	}

	bool shift_time_svd ();

	void test_wrap () {
		std::vector<Matrix_d> v(N);
		for (int i=0;i<N;i++) {
			v[i] = freePropagator_open;
			v[i].applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
			v[i] *= std::exp(-dt*(-mu-0.5*B));
		}
		SVDHelper help;
		Matrix_d prod;
	}

	void load_sigma (lua_State *L, const char *fn);

	double fraction_completed () const {
		return 1.0;
	}

	void update () {
		for (int i=0;i<flips_per_update;i++) {
			acceptance.add(metropolis()?1.0:0.0);
			measured_sign.add(psign*update_sign);
		}
		shift_time_svd();
	}

	void get_green_function (double s = 1.0, int t0 = 0);

	double get_kinetic_energy (const Matrix_d &M) {
		positionSpace = M;
		return positionSpace.trace() / V;
	}

	double pair_correlation (const Matrix_d& rho_up, const Matrix_d& rho_dn) {
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
	void measure_quick ();
	void measure_sign ();
	int volume () const { return V; }
	int timeSlices () const { return N; }

	void write_wavefunction (std::ostream &out);

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
		write_green_function();
	}

	void write_green_function ();

	std::string params () {
		std::ostringstream buf;
		buf << config << std::endl;
		return buf.str();
	}

	~Simulation () {
		fftw_destroy_plan(x2p_col);
		fftw_destroy_plan(p2x_col);
	}

	std::pair<double, double> recheck ();
	void straighten_slices ();

	void discard_measurements () {
		acceptance.clear();
		density.clear();
		magnetization.clear();
		singlet.clear();
		order_parameter.clear();
		chi_d.clear();
		chi_af.clear();
		kinetic.clear();
		interaction.clear();
		sign.clear();
		measured_sign.clear();
		exact_sign.clear();
		for (int i=0;i<V;i++) {
			d_up[i].clear();
			d_dn[i].clear();
			spincorrelation[i].clear();
		}
	}

	protected:
};


#endif // SIMULATION_HPP

