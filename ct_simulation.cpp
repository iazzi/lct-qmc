#include "simulation.hpp"
#include "mpfr.hpp"

#include "lua_tuple.hpp"

// FIXME only works in 2D
void Simulation::prepare_open_boundaries () {
	Matrix_d H = Matrix_d::Zero(V, V);
	for (int x=0;x<Lx;x++) {
		for (int y=0;y<Ly;y++) {
			for (int z=0;z<Lz;z++) {
				int a = x*Ly*Lz + y*Lz + z;
				int b = ((x+1)%Lx)*Ly*Lz + y*Lz + z;
				int c = x*Ly*Lz + ((y+1)%Ly)*Lz + z;
				int d = x*Ly*Lz + y*Lz + (z+1)%Lz;
				if (Lx>1 && x!=Lx-0) H(a, b) = H(b, a) = -tx;
				if (Ly>1 && y!=Ly-0) H(a, c) = H(c, a) = -ty;
				if (Lz>1 && z!=Lz-0) H(a, d) = H(d, a) = -tz;
				double pos_x = (x-Lx/2.0+0.5);
				double pos_y = (y-Ly/2.0+0.5);
				double pos_z = (z-Lz/2.0+0.5);
				H(a, a) = 0.5*(w_x*pos_x*pos_x + w_y*pos_y*pos_y + w_z*pos_z*pos_z);
			}
		}
	}

	Eigen::SelfAdjointEigenSolver<Matrix_d> solver;
	solver.compute(H);
	freePropagator_open = solver.eigenvectors() * (-dt*solver.eigenvalues().array()).exp().matrix().asDiagonal() * solver.eigenvectors().transpose();
	freePropagator_inverse = solver.eigenvectors() * (+dt*solver.eigenvalues().array()).exp().matrix().asDiagonal() * solver.eigenvectors().transpose();
	positionSpace.setIdentity(V, V);
	momentumSpace.setZero(V, V);
	fftw_execute(x2p_col);
	//std::cerr << "k-space\n" << momentumSpace << std::endl << std::endl;
	momentumSpace.applyOnTheLeft(freePropagator.asDiagonal());
	fftw_execute(p2x_col);
	std::cerr << "propagator difference = " << (freePropagator_open-positionSpace/V).norm() << std::endl;
	hamiltonian = H;
	eigenvectors = solver.eigenvectors();
	energies = solver.eigenvalues();
}


void Simulation::prepare_propagators () {
	Vector_d my_energies = Vector_d::Zero(V);
	freePropagator = Vector_d::Zero(V);
	freePropagator_b = Vector_d::Zero(V);
	potential = Vector_d::Zero(V);
	freePropagator_x = Vector_d::Zero(V);
	//freePropagator_x_b = Vector_d::Zero(V);
	staggering = Array_d::Zero(V);
	for (int i=0;i<V;i++) {
		int x = (i/Lz/Ly)%Lx;
		int y = (i/Lz)%Ly;
		int z = i%Lz;
		int Kx = Lx, Ky = Ly, Kz = Lz;
		int kx = (i/Kz/Ky)%Kx;
		int ky = (i/Kz)%Ky;
		int kz = i%Kz;
		if (Kx>1) my_energies[i] += (Kx>2?-2.0:-1.0) * tx * cos(2.0*kx*pi/Kx);
		if (Ky>1) my_energies[i] += (Ky>2?-2.0:-1.0) * ty * cos(2.0*ky*pi/Ky);
		if (Kz>1) my_energies[i] += (Kz>2?-2.0:-1.0) * tz * cos(2.0*kz*pi/Kz);
		freePropagator[i] = exp(-dt*my_energies[i]);
		freePropagator_b[i] = exp(dt*my_energies[i]);
		double pos_x = (x-Lx/2.0+0.5);
		potential[i] = 0.5*w_x*pos_x*pos_x;
		freePropagator_x[i] = exp(-dt*potential[i]);
		//freePropagator_x_b[i] = exp(dt*potential[i]);
		staggering[i] = (x+y+z)%2?-1.0:1.0;
	}

	int E = 3;
	if (Lz<2) E=2;
	if (Lz<2 && Ly<2) E=1;
	const int size[] = { Lx, Ly, Lz, };
	x2p_col = fftw_plan_many_dft_r2c(E, size, V, positionSpace.data(),
			size, 1, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()), size, 1, V, FFTW_PATIENT);
	p2x_col = fftw_plan_many_dft_c2r(E, size, V, reinterpret_cast<fftw_complex*>(momentumSpace.data()),
			size, 1, V, positionSpace.data(), size, 1, V, FFTW_PATIENT);
	positionSpace.setIdentity(V, V);
	momentumSpace.setZero(V, V);

}

void Simulation::init () {
	if (Lx<2) { Lx = 1; tx = 0.0; }
	if (Ly<2) { Ly = 1; ty = 0.0; }
	if (Lz<2) { Lz = 1; tz = 0.0; }
	V = Lx * Ly * Lz;
	time_shift = 0;
	if (flips_per_update<1) flips_per_update = V;
	randomPosition = std::uniform_int_distribution<int>(0, V-1);
	randomTime = std::uniform_real_distribution<double>(0, beta);
	dt = beta/N;
	A = sqrt(exp(g*dt)-1.0);
	coin_flip = std::bernoulli_distribution(0.5);
	v_x.setZero(V);
	v_p.setZero(V);

	positionSpace.setIdentity(V, V);
	momentumSpace.setIdentity(V, V);

	prepare_propagators();
	prepare_open_boundaries();

	make_svd_inverse(0.0);
	plog = svd_probability();
	psign = svd_sign();

	std::cerr << "testing make_svd_double\n";
	for (int l=0;l<10;l++) {
		double s = randomTime(generator);
		std::cerr << "inserting random slice at " << s << std::endl;
		diagonals[s] = Vector_d::Random(V);
	}
	make_svd_double(0.0);
	svdA.add_identity(1.0);
	svdB.add_identity(1.0);
	std::cerr << 0.0 << ' ' << svd_probability() << ' ' << svd_sign() << std::endl;
	make_svd_double(0.5);
	svdA.add_identity(1.0);
	svdB.add_identity(1.0);
	std::cerr << 0.5 << ' ' << svd_probability() << ' ' << svd_sign() << std::endl;
	for (int l=0;l<20;l++) {
		double s = randomTime(generator);
		make_svd_double(s);
		svdA.add_identity(1.0);
		svdB.add_identity(1.0);
		std::cerr << s << ' ' << svd_probability() << ' ' << svd_sign() << std::endl;
	}

	diagonals.clear();
	make_svd_double(0.0);
	svdA.add_identity(1.0);
	svdB.add_identity(1.0);


	init_measurements();
	reset_updates();
}

void Simulation::load (lua_State *L, int index) {
	lua_pushvalue(L, index);
	lua_get(L, config);
	lua_pop(L, 1);
	//std::cerr << config << std::endl;
	Lx = config.Lx;
	Ly = config.Ly;
	Lz = config.Lz;
	N = config.N;
	beta = config.beta;
	tx = config.tx;
	ty = config.ty;
	tz = config.tz;
	g = fabs(config.U);
	mu = config.mu;
	B = config.B;
	lua_getfield(L, index, "SEED");
	if (lua_isnumber(L, -1)) {
		generator.seed(lua_tointeger(L, -1));
	} else if (lua_isstring(L, -1)) {
		std::stringstream in(std::string(lua_tostring(L, -1)));
		in >> generator;
	}
	lua_pop(L, 1);
	lua_getfield(L, index, "w_x");     w_x = lua_tonumber(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "w_y");     w_y = lua_tonumber(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "w_z");     w_z = lua_tonumber(L, -1);            lua_pop(L, 1);
	//lua_getfield(L, index, "h");    staggered_field = lua_tonumber(L, -1);     lua_pop(L, 1);
	lua_getfield(L, index, "RESET");  reset = lua_toboolean(L, -1);            lua_pop(L, 1);
	//lua_getfield(L, index, "REWEIGHT");  reweight = lua_tointeger(L, -1);      lua_pop(L, 1);
	lua_getfield(L, index, "OUTPUT");  outfn = lua_tostring(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "gf_file"); gf_name = lua_tostring(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "SLICES");  mslices = lua_tointeger(L, -1);         lua_pop(L, 1);
	lua_getfield(L, index, "SVD");     msvd = lua_tointeger(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "flips_per_update");     flips_per_update = lua_tointeger(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "open_boundary");     open_boundary = lua_toboolean(L, -1);            lua_pop(L, 1);
	//lua_getfield(L, index, "LOGFILE");  logfile.open(lua_tostring(L, -1));     lua_pop(L, 1);
	init();
}

void Simulation::save (lua_State *L, int index) {
	if (index<1) index = lua_gettop(L)+index+1;
	std::stringstream out;
	out << generator;
	lua_pushstring(L, out.str().c_str());
	lua_setfield(L, index, "SEED");
	lua_pushinteger(L, this->Lx); lua_setfield(L, index, "Lx");
	lua_pushinteger(L, this->Ly); lua_setfield(L, index, "Ly");
	lua_pushinteger(L, this->Lz); lua_setfield(L, index, "Lz");
	lua_pushinteger(L, N); lua_setfield(L, index, "N");
	lua_pushnumber(L, 1.0/beta); lua_setfield(L, index, "T");
	lua_pushnumber(L, tx); lua_setfield(L, index, "tx");
	lua_pushnumber(L, ty); lua_setfield(L, index, "ty");
	lua_pushnumber(L, tz); lua_setfield(L, index, "tz");
	//lua_pushnumber(L, Vx); lua_setfield(L, index, "Vx");
	//lua_pushnumber(L, Vy); lua_setfield(L, index, "Vy");
	//lua_pushnumber(L, Vz); lua_setfield(L, index, "Vz");
	lua_pushnumber(L, -g); lua_setfield(L, index, "U");
	lua_pushnumber(L, mu); lua_setfield(L, index, "mu");
	lua_pushnumber(L, B); lua_setfield(L, index, "B");
	//lua_pushnumber(L, staggered_field); lua_setfield(L, index, "h");
	lua_pushinteger(L, mslices); lua_setfield(L, index, "SLICES");
	lua_pushinteger(L, msvd); lua_setfield(L, index, "SVD");
	lua_pushinteger(L, flips_per_update); lua_setfield(L, index, "flips_per_update");
	lua_pushboolean(L, open_boundary?1:0); lua_setfield(L, index, "open_boundary");
	lua_newtable(L);
	L << sign;
	lua_setfield(L, -2, "sign");
	L << acceptance;
	lua_setfield(L, -2, "acceptance");
	L << density;
	lua_setfield(L, -2, "density");
	L << magnetization;
	lua_setfield(L, -2, "magnetization");
	L << order_parameter;
	lua_setfield(L, -2, "order_parameter");
	L << chi_af;
	lua_setfield(L, -2, "chi_af");
	//L << measured_sign;
	//lua_setfield(L, -2, "measured_sign");
	L << chi_d;
	lua_setfield(L, -2, "chi_d");
	lua_setfield(L, index, "results");
}

void Simulation::load_checkpoint (lua_State *L) {
	lua_getfield(L, -1, "SEED");
	std::stringstream in;
	in.str(lua_tostring(L, -1));
	in >> generator;
	lua_pop(L, 1);
	lua_getfield(L, -1, "time_shift");
	time_shift = lua_tointeger(L, -1);
	lua_pop(L, 1);
	lua_getfield(L, -1, "results");
	lua_getfield(L, -1, "sign");
	lua_get(L, sign);
	lua_pop(L, 1);
	lua_getfield(L, -1, "acceptance");
	lua_get(L, acceptance);
	lua_pop(L, 1);
	lua_getfield(L, -1, "density");
	lua_get(L, density);
	lua_pop(L, 1);
	lua_getfield(L, -1, "magnetization");
	lua_get(L, magnetization);
	lua_pop(L, 1);
	lua_getfield(L, -1, "order_parameter");
	lua_get(L, order_parameter);
	lua_pop(L, 1);
	lua_getfield(L, -1, "chi_af");
	lua_get(L, chi_af);
	lua_pop(L, 1);
	lua_getfield(L, -1, "exact_sign");
	lua_get(L, exact_sign);
	lua_pop(L, 1);
	//lua_getfield(L, -1, "measured_sign");
	//lua_get(L, measured_sign);
	//lua_pop(L, 1);
	//lua_getfield(L, -1, "sign_correlation");
	//lua_get(L, sign_correlation);
	//lua_pop(L, 1);
	lua_getfield(L, -1, "chi_d");
	lua_get(L, chi_d);
	lua_pop(L, 1);
	lua_pop(L, 1);
	lua_getfield(L, -1, "N");
	int oldN = lua_tointeger(L, -1);
	lua_pop(L, 1);
	lua_getfield(L, -1, "V");
	int oldV = lua_tointeger(L, -1);
	lua_pop(L, 1);
	lua_getfield(L, -1, "sigma");
	//TODO no  way I'm going to make this work right now
	lua_pop(L, 1);
}

void Simulation::save_checkpoint (lua_State *L) {
	lua_newtable(L);
	std::stringstream out;
	out << generator;
	lua_pushstring(L, out.str().c_str());
	lua_setfield(L, -2, "SEED");
	lua_pushinteger(L, time_shift);
	lua_setfield(L, -2, "time_shift");
	lua_newtable(L);
	L << sign;
	lua_setfield(L, -2, "sign");
	L << acceptance;
	lua_setfield(L, -2, "acceptance");
	L << density;
	lua_setfield(L, -2, "density");
	L << magnetization;
	lua_setfield(L, -2, "magnetization");
	L << order_parameter;
	lua_setfield(L, -2, "order_parameter");
	L << chi_af;
	lua_setfield(L, -2, "chi_af");
	L << exact_sign;
	lua_setfield(L, -2, "exact_sign");
	//L << measured_sign;
	//lua_setfield(L, -2, "measured_sign");
	//L << sign_correlation;
	//lua_setfield(L, -2, "sign_correlation");
	L << chi_d;
	lua_setfield(L, -2, "chi_d");
	lua_setfield(L, -2, "results");
	lua_pushinteger(L, N);
	lua_setfield(L, -2, "N");
	lua_pushinteger(L, V);
	lua_setfield(L, -2, "V");
	lua_newtable(L);
	for (iter d=diagonals.begin();d!=diagonals.end();d++) {
		lua_pushnumber(L, d->first);
		lua_newtable(L);
		for (int j=0;j<V;j++) {
			lua_pushnumber(L, d->second[j]);
			lua_rawseti(L, -2, j+1);
			//std::cerr << (diagonals[i][j]<0.0?'-':'+') << ' ';
		}
		lua_settable(L, -3);
	}
	//std::cerr << std::endl;
	lua_setfield(L, -2, "sigma");
}

std::pair<double, double> Simulation::rank1_probability (int x) {
	int L = update_size;
	int j;
	double d1, d2;
	for (j=0;j<V;j++) {
		if (update_perm[j]==x) break;
	}
	if (j>=L) {
		std::swap(update_perm[j], update_perm[L]);
		new_update_size = update_size+1;
		update_matrix_up.col(j).swap(update_matrix_up.col(L));
		update_matrix_up.row(j).swap(update_matrix_up.row(L));
		d1 = update_matrix_up.topLeftCorner(L+1, L+1).determinant();
		update_matrix_dn.col(j).swap(update_matrix_dn.col(L));
		update_matrix_dn.row(j).swap(update_matrix_dn.row(L));
		d2 = update_matrix_dn.topLeftCorner(L+1, L+1).determinant();
	} else {
		std::swap(update_perm[j], update_perm[L-1]);
		new_update_size = update_size-1;
		update_matrix_up.col(j).swap(update_matrix_up.col(L-1));
		update_matrix_up.row(j).swap(update_matrix_up.row(L-1));
		d1 = update_matrix_up.topLeftCorner(L-1, L-1).determinant();
		update_matrix_dn.col(j).swap(update_matrix_dn.col(L-1));
		update_matrix_dn.row(j).swap(update_matrix_dn.row(L-1));
		d2 = update_matrix_dn.topLeftCorner(L-1, L-1).determinant();
	}
	double s = 1.0;
	if (d1 < 0) {
		s *= -1.0;
		d1 *= -1.0;
	}
	if (d2 < 0) {
		s *= -1.0;
		d2 *= -1.0;
	}
	return std::pair<double, double>(std::log(d1)+std::log(d2), s);
}

bool Simulation::metropolis_add () {
	double t = randomTime(generator);
	if (diagonals.find(t)!=diagonals.end()) return false;
	make_svd_double(t);
	Vector_d new_diag(V);
	for (int x=0;x<V;x++) new_diag[x] = coin_flip(generator)?A:-A;
	svdA.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+new_diag).asDiagonal());
	svdB.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+new_diag).asDiagonal());
	svdA.absorbU();
	svdB.absorbU();
	svdA.add_identity(1.0);
	svdB.add_identity(1.0);
	double np = svd_probability();
	std::cerr << "trying add: " << plog << " -> " << np << " = " << np-plog << std::endl;
	bool ret = -trialDistribution(generator)<np-plog+log(beta)-log(diagonals.size()+1);
	if (ret) {
		std::cerr << "increasing slices: " << diagonals.size() << " -> " << diagonals.size()+1 << std::endl;
		diagonals.insert(std::pair<double, Vector_d>(t, new_diag));
		plog = np;
		psign = svd_sign();
		make_svd_double(0.0);
		svdA.add_identity(1.0);
		svdB.add_identity(1.0);
		std::cerr << "increased slices: " << svd_probability() << std::endl;
		std::cerr << new_diag.transpose() << std::endl;
		std::cerr << diagonals[t].transpose() << std::endl;
	}
	return ret;
}

bool Simulation::metropolis_del () {
	if (diagonals.size()==0) return false;
	diagonal d = diagonals.begin();
	int n = std::uniform_int_distribution<int>(0, diagonals.size()-1)(generator);
	std::cerr << "deleting slice #" << n << std::endl;
	while (n-->0) d++;
	if (d==diagonals.end()) return false;
	std::cerr << "deleting slice @" << d->first << std::endl;
	make_svd_double(d->first);
	double t = d->first;
	Vector_d save = d->second;
	svdA.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+d->second).array().inverse().matrix().asDiagonal());
	svdB.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+d->second).array().inverse().matrix().asDiagonal());
	svdA.absorbU();
	svdB.absorbU();
	//std::cerr << svdA.matrix() << std::endl << std::endl;
	svdA.add_identity(1.0);
	svdB.add_identity(1.0);
	double np = svd_probability();
	bool ret = -trialDistribution(generator)<np-plog+log(diagonals.size())-log(beta);
	std::cerr << "trying del: " << plog << " -> " << np << " = " << np-plog << std::endl;
	if (ret) {
		std::cerr << "decreasing slices: " << diagonals.size() << " -> " << diagonals.size()-1 << std::endl;
		diagonals.erase(d->first);
		plog = np;
		psign = svd_sign();
		make_svd_double(0.0);
		svdA.add_identity(1.0);
		svdB.add_identity(1.0);
		//std::cerr << "decreased slices: " << svd_probability() << std::endl;
		//std::cerr << save.transpose() << std::endl;
		//make_svd_double(t);
		//svdA.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+save).matrix().asDiagonal());
		//svdB.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+save).matrix().asDiagonal());
		//svdA.absorbU();
		//svdB.absorbU();
		//svdA.add_identity(1.0);
		//svdB.add_identity(1.0);
		std::cerr << "decreased slices: " << svd_probability() << std::endl;
		//make_svd_double(t);
		//svdA.Vt.applyOnTheRight((Vector_d::Constant(V, 1.0)+save).matrix().asDiagonal());
		//svdB.Vt.applyOnTheRight((Vector_d::Constant(V, 1.0)+save).matrix().asDiagonal());
		//svdA.absorbVt();
		//svdB.absorbVt();
		//svdB.absorbU();
		//std::cerr << svdA.matrix() << std::endl << std::endl;
		//svdA.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+save).array().inverse().matrix().asDiagonal());
		//svdB.U.applyOnTheLeft((Vector_d::Constant(V, 1.0)+save).array().inverse().matrix().asDiagonal());
		//svdA.absorbU();
		//svdB.absorbU();
		//std::cerr << svdA.matrix() << std::endl << std::endl;
		//svdA.add_identity(1.0);
		//svdB.add_identity(1.0);
		//std::cerr << "decreased slices: " << svd_probability() << std::endl;
	}
	return ret;
}

bool Simulation::metropolis_sweep () {
	//std::cerr << "sweeping " << diagonals.size() << " slices" << std::endl;
	for (diagonal d = diagonals.begin();d!=diagonals.end();d++) {
		current = d;
		//make_svd_inverse(d->first);
		//for (int i=0;i<V;i++) metropolis_flip();
	}
	return true;
}

bool Simulation::metropolis_flip () {
	steps++;
	bool ret = false;
	int x = randomPosition(generator);
	std::pair<double, double> r1 = rank1_probability(x);
	ret = -trialDistribution(generator)<r1.first-update_prob;
	if (ret) {
		//std::cerr << "accepted " << x << ' ' << update_size << std::endl;
		current->second[x] = -current->second[x];
		update_size = new_update_size;
		update_prob = r1.first;
		update_sign = r1.second;
		//std::cerr << "accepted metropolis step" << std::endl;
	} else {
		//std::cerr << "rejected metropolis step" << std::endl;
	}
	return ret;
}

void Simulation::load_sigma (lua_State *L, const char *fn) {
	luaL_dofile(L, fn);
	lua_getfield(L, -1,  "sigma");
	for (int t=0;t<N;t++) {
		lua_rawgeti(L, -1, t+1);
		for (int x=0;x<V;x++) {
			lua_rawgeti(L, -1, x+1);
			diagonals[t][x] = lua_tonumber(L, -1)<0?-A:A;
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	lua_pop(L, 2);
}

void Simulation::write_wavefunction (std::ostream &out) {
	for (auto d : diagonals) {
		out << (d.second.array()>Array_d::Zero(V)).transpose() << std::endl;
	}
	out << std::endl;
}

std::pair<double, double> Simulation::recheck () {
	return std::pair<double, double>(1, 1);
}

void Simulation::straighten_slices () {
}

void Simulation::measure_sign () {
	exact_sign.add(psign*update_sign*recheck().second);
}

void Simulation::measure_quick () {
	double s = svd_sign();
	rho_up = Matrix_d::Identity(V, V) - svdA.inverse();
	rho_dn = svdB.inverse();
	double n_up = rho_up.diagonal().array().sum();
	double n_dn = rho_dn.diagonal().array().sum();
	sign.add(psign*update_sign);
	density.add(s*(n_up+n_dn)/V);
	magnetization.add(s*(n_up-n_dn)/2.0/V);
	for (int i=0;i<V;i++) {
		d_up[i].add(s*rho_up(i, i));
		d_dn[i].add(s*rho_dn(i, i));
	}
	for (int j=0;j<V;j++) {
		double ssz = 0.0;
		int x = j;
		int y = shift_x(j, 1);
		ssz += rho_up(x, x)*rho_up(y, y) + rho_dn(x, x)*rho_dn(y, y);
		ssz -= rho_up(x, x)*rho_dn(y, y) + rho_dn(x, x)*rho_up(y, y);
		ssz -= rho_up(x, y)*rho_up(y, x) + rho_dn(x, y)*rho_dn(y, x);
		spincorrelation[j].add(s*0.25*ssz);
	}
}

void Simulation::measure () {
	double s = svd_sign();
	rho_up = Matrix_d::Identity(V, V) - svdA.inverse();
	rho_dn = svdB.inverse();
	double K_up = get_kinetic_energy(rho_up);
	double K_dn = get_kinetic_energy(rho_dn);
	double n_up = rho_up.diagonal().array().sum();
	double n_dn = rho_dn.diagonal().array().sum();
	double op = (rho_up.diagonal().array()-rho_dn.diagonal().array()).square().sum();
	double n2 = (rho_up.diagonal().array()*rho_dn.diagonal().array()).sum();
	sign.add(psign*update_sign);
	density.add(s*(n_up+n_dn)/V);
	magnetization.add(s*(n_up-n_dn)/2.0/V);
	//magnetization_slow.add(s*(n_up-n_dn)/2.0/V);
	order_parameter.add(s*op/V);
	kinetic.add(s*(K_up-K_dn)/tx/V);
	interaction.add(s*g*n2/tx/V);
	//sign.add(svd_sign());
	//- (d1_up*d2_up).sum() - (d1_dn*d2_dn).sum();
	for (int i=0;i<V;i++) {
		d_up[i].add(s*rho_up(i, i));
		d_dn[i].add(s*rho_dn(i, i));
	}
	double d_wave_chi = pair_correlation(rho_up, rho_dn);
	Matrix_d F_up = svdA.inverse();
	Matrix_d F_dn = Matrix_d::Identity(V, V) - svdB.inverse();
	chi_d.add(s*d_wave_chi*beta);
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
		}
	}
	//if (staggered_field!=0.0) staggered_magnetization.add(s*(rho_up.diagonal().array()*staggering - rho_dn.diagonal().array()*staggering).sum()/V);
	get_green_function(s);
}

void Simulation::get_green_function (double s, int t0) {
}

void Simulation::write_green_function () {
	if (gf_name.empty()) return;
	std::ofstream out(gf_name);
	Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "{", "}", "{", "}");
	out << "beta = " << beta*tx << "\n";
	out << "dtau = " << dt*tx << "\n";
	out << "mu = " << mu/tx << "\n";
	out << "B = " << B/tx << "\n";
	out << "U = " << g/tx << "\n";
	out << "Lx = " << Lx << "\n";
	out << "Ly = " << Ly << "\n";
	out << "Lz = " << Lz << "\n";
	out << "N = " << N << "\n";
	out << "sign = " << sign.mean() << "\n";
	out << "Dsign = " << sign.error() << "\n";
	out << "G_up = {}\n";
	out << "G_dn = {}\n";
	out << "DG_up = {}\n";
	out << "DG_dn = {}\n\n";
	for (int t=0;t<=N;t++) {
		Eigen::ArrayXXd G = green_function_up[t].mean()/sign.mean();
		Eigen::ArrayXXd DG = G.abs()*(green_function_up[t].error()/green_function_up[t].mean().abs() + sign.error()/fabs(sign.mean()));
		out << "G_up[" << t << "] = " << G.format(HeavyFmt) << std::endl;
		out << "DG_up[" << t << "] = " << DG.format(HeavyFmt) << std::endl;
	}
	for (int t=0;t<=N;t++) {
		Eigen::ArrayXXd G = green_function_dn[t].mean()/sign.mean();
		Eigen::ArrayXXd DG = G.abs()*(green_function_dn[t].error()/green_function_dn[t].mean().abs() + sign.error()/fabs(sign.mean()));
		out << "G_dn[" << t << "] = " << G.format(HeavyFmt) << std::endl;
		out << "DG_dn[" << t << "] = " << DG.format(HeavyFmt) << std::endl;
	}
}



