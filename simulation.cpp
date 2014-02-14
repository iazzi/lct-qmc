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
	randomTime = std::uniform_int_distribution<int>(0, N-1);
	dt = beta/N;
	A = sqrt(exp(g*dt)-1.0);
	diagonals.insert(diagonals.begin(), N, Vector_d::Zero(V));
	distribution = std::bernoulli_distribution(0.5);
	for (size_t i=0;i<diagonals.size();i++) {
		diagonals[i] = Array_d::Constant(V, distribution(generator)?A:-A);
		for (int j=0;j<V;j++) {
			diagonals[i][j] = distribution(generator)?A:-A;
		}
	}
	v_x.setZero(V);
	v_p.setZero(V);

	positionSpace.setIdentity(V, V);
	momentumSpace.setIdentity(V, V);

	prepare_propagators();
	prepare_open_boundaries();

	valid_slices.clear();
	valid_slices.insert(valid_slices.begin(), nslices(), false);
	std::tie(plog, psign) = make_plain_inverse();
	std::tie(plog, psign) = make_svd_inverse();

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
	L << singlet;
	lua_setfield(L, -2, "singlet");
	L << order_parameter;
	lua_setfield(L, -2, "order_parameter");
	L << chi_af;
	lua_setfield(L, -2, "chi_af");
	//L << measured_sign;
	//lua_setfield(L, -2, "measured_sign");
	L << d_up;
	lua_setfield(L, -2, "d_up");
	L << d_dn;
	lua_setfield(L, -2, "d_dn");
	L << spincorrelation;
	lua_setfield(L, -2, "spincorrelation");
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
	for (int i=0;i<N;i++) {
		int t = oldN<N?i%oldN:i;
		for (int j=0;j<V;j++) {
			int x = j%oldV;
			lua_rawgeti(L, -1, t*oldV+x+1);
			diagonals[t][x] = lua_tonumber(L, -1)<0.0?-A:A;
			lua_pop(L, 1);
			//std::cerr << (diagonals[t][x]<0.0?'-':'+') << ' ';
		}
	}
	//std::cerr << std::endl;
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
	for (int i=0;i<N;i++) {
		for (int j=0;j<V;j++) {
			lua_pushnumber(L, diagonals[i][j]);
			lua_rawseti(L, -2, i*V+j+1);
			//std::cerr << (diagonals[i][j]<0.0?'-':'+') << ' ';
		}
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

bool Simulation::metropolis () {
	steps++;
	bool ret = false;
	int x = randomPosition(generator);
	std::pair<double, double> r1 = rank1_probability(x);
	ret = -trialDistribution(generator)<r1.first-update_prob;
	if (ret) {
		//std::cerr << "accepted " << x << ' ' << update_size << std::endl;
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
		out << (d.array()>Array_d::Zero(V)).transpose() << std::endl;
	}
	out << std::endl;
}

std::pair<double, double> Simulation::recheck () {
	const int prec = 2048;
	PreciseMatrix A(prec), W(prec), A1(prec), A2(prec);
	PreciseMatrix C(prec), Q(prec), wr(prec), wi(prec);
	A = Matrix_d::Identity(V, V);
	for (int i=0;i<N;i++) {
		//A.applyOnTheLeft(freePropagator_open*((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
	}
	for (int i=0;i<N;i++) {
		A.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
		A.applyOnTheLeft(freePropagator_open);
	}
	A2 = A1 = W = A;
	A1 *= std::exp(beta*B/2+beta*mu);
	A2 *= std::exp(-beta*B/2+beta*mu);
	A1 += Matrix_d::Identity(V, V);
	A2 += Matrix_d::Identity(V, V);
	std::vector<int> p;
	int s1 = A1.in_place_LU(p);
	int s2 = A2.in_place_LU(p);
	mpfr_t d;
	mpfr_init2(d, prec);
	mpfr_set_d(d, 1.0, A.rnd());
	for (int i=0;i<V;i++) {
		mpfr_mul(d, d, A1.coeff(i, i), A1.rnd());
		mpfr_mul(d, d, A2.coeff(i, i), A2.rnd());
	}
	int sign = s1*s2*mpfr_sgn(d);
	mpfr_abs(d, d, A.rnd());
	mpfr_log(d, d, A.rnd());
	std::cout << "SVDs: " << svd.S.transpose() << '\n';
	std::cout << "svd determinant: " << (psign<0.0?"-exp(":"exp(") << plog << ')';
	if (sign<0) {
		std::cout << ", exact probability: -exp(" << d << ')' << std::endl;
	} else {
		std::cout << ", exact probability: +exp(" << d << ')' << std::endl;
	}
	double r = 0.0;
	for (auto d : diagonals) {
		r += (d.array()>Array_d::Zero(V)).count();
	}
	std::cout << "positive points = " << r/N/V << std::endl;
	std::cout << "time_shift = " << time_shift << std::endl;
	std::ofstream out;
	if (svd_sign()*sign<0.0 && !isnan(plog)) {
		std::cerr << "different!" << std::endl;
		out.open("last_different.wf");
	} else {
		out.open("last_equal.wf");
		std::cerr << "equal." << std::endl;
	}
	out << "# steps = " << steps << '/' << N*V << " (" << int(100*steps/N/V) << "%)" << std::endl;
	out << "# time_shift = " << time_shift << std::endl;
	out << "# svd probability = " << (psign<0.0?"-exp(":"exp(") << plog << ')' << std::endl;
	if (sign<0) {
		out << "# exact determinant: -exp(" << d << ')' << std::endl;
	} else {
		out << "# exact determinant: +exp(" << d << ')' << std::endl;
	}
	write_wavefunction(out);
	out.flush();
	out.close();
	r = mpfr_get_d(d, A.rnd());
	mpfr_clear(d);
	return std::pair<double, double>(r, sign<0?-1.0:1.0);
	W.reduce_to_hessenberg();
	W.extract_hessenberg_H(C);
	C.reduce_to_ev(wr, wi);
	int n1 = -1, n2 = -1;
	for (int i=0;i<V;i++) {
		if (mpfr_zero_p(wi.coeff(i, 0)) && mpfr_sgn(wr.coeff(i, 0))<0) {
			if (n1==-1) {
				n1 = i;
			} else if (n2==-1) {
				n2 = i;
			} else {
			}
		}	
		std::cout << '(' << wr.coeff(i, 0) << ", " << wi.coeff(i, 0) << ") ";
	}
	std::cout << std::endl;
	A1 = A;
	mpfr_t lambda;
	mpfr_init2(lambda, prec);
	mpfr_mul_d(lambda, wr.coeff(n1, 0), 1.00, A1.rnd());
	for (int i=0;i<V;i++) {
		mpfr_sub(A1.coeff(i, i), A1.coeff(i, i), lambda, A1.rnd());
	}
	std::cout << "todo: LU" << std::endl;
	A1.in_place_LU(p);
	std::cout << "LU done" << std::endl;
	PreciseMatrix v(prec);
	v = Matrix_d::Random(V, 1);
	v.normalize();
	//v.transpose_in_place(); std::cout << v << std::endl; v.transpose_in_place(); 
	for (int i=0;i<150;i++) {
		A1.apply_inverse_LU_vector(v, p);
		v.normalize();
		//v.transpose_in_place(); std::cout << v << std::endl; v.transpose_in_place(); 
	}
	v.transpose_in_place(); std::cout << "=======\n" << v << std::endl; v.transpose_in_place(); 
	v.applyOnTheLeft(A);
	v.normalize();
	v.transpose_in_place(); std::cout << "=======\n" << v << std::endl; v.transpose_in_place(); 
	std::ofstream wf("wf.dat");
	wf.precision(18);
	wf << "return {\n N = "<< (N+1) << ",\n Lx = " << Lx << ",\n Ly = " << Ly << ",\n Lz = " << Lz << ",\n dt = "<< dt << ",\n";
	wf << " sigma = {\n";
	for (int i=0;i<=N;i++) {
		wf << "  {";
		for (int j=0;j<V;j++) wf << " " << diagonal(i)[j] << ",";
		wf << " },\n";
	}
	wf << " },\n";
	wf << " wf1 = {\n";
	mpfr_set_d(lambda, 1.0, A1.rnd());
	for (int i=0;i<=N;i++) {
		wf << "  {";
		for (int j=0;j<V;j++) wf << " " << v.coeff(j, 0) << ",";
		wf << " norm = " << lambda << ", },\n";
		if (i==N) break;
		v.applyOnTheLeft(freePropagator_open*((Vector_d::Constant(V, 1.0)+diagonal(i)).array()).matrix().asDiagonal());
		v.get_norm(lambda);
		v.normalize();
	}
	wf << " },\n";
	wf << " wf2 = {\n";
	wf << " },\n";
	wf << "}\n";
	wf.close();
	mpfr_clear(lambda);
	//throw "";
	const int myV = V-2;
	Matrix_d G0, ev0, ev, tmp;
	Matrix_d overlaps = Matrix_d::Identity(myV, myV);
	SVDHelper helper;
	Eigen::SelfAdjointEigenSolver<Matrix_d> solver;
	solver.compute(hamiltonian);
	G0 = solver.eigenvectors() * (-0.5*dt*solver.eigenvalues().array()).exp().matrix().asDiagonal() * solver.eigenvectors().transpose();
	for (int i=0;i<10;i++) {
		solver.compute(G0*(Vector_d::Constant(V, 1.0)+diagonal(i)).asDiagonal()*G0);
		if (i!=0) {
			tmp = ev.transpose()*solver.eigenvectors().leftCols(myV);
			helper.fullSVD(tmp);
			overlaps.applyOnTheRight(helper.Vt.transpose()*helper.U.transpose());
		}
		ev = solver.eigenvectors().leftCols(myV);
		if (i==0) ev0 = ev;
		//std::cerr << "i=" << i << "; overlaps = " << overlaps.transpose() << std::endl;
	}
	tmp = ev.transpose()*ev0;
	helper.fullSVD(tmp);
	overlaps.applyOnTheRight(helper.Vt.transpose()*helper.U.transpose());
	std::cerr << "===\n" << overlaps.transpose() << std::endl << std::endl;
	return std::pair<double, double>(r, sign<0?-1.0:1.0);
}

void Simulation::straighten_slices () {
	for (Vector_d& d : diagonals) {
		bool up = true;
		for (int i=0;i<V;i++) {
			if (d[i]<0) {
				up = false;
				break;
			}
		}
		if (up) {
			for (int i=0;i<V;i++) {
				d[i] = -d[i];
			}
		}
	}
}

void Simulation::measure_sign () {
	exact_sign.add(psign*update_sign*recheck().second);
}

void Simulation::measure_quick () {
	double s = svd_sign();
	double n_up = rho_up.diagonal().array().sum();
	double n_dn = rho_dn.diagonal().array().sum();
	sign.add(psign*update_sign);
	density.add(s*(n_up+n_dn)/V);
	magnetization.add(s*(n_up-n_dn)/2.0/V);
	for (int i=0;i<V;i++) {
		d_up[i].add(s*rho_up(i, i));
		d_dn[i].add(s*rho_dn(i, i));
	}
	double sum = 0.0;
	for (int j=0;j<V;j++) {
		double ssz = 0.0;
		int x = j;
		int y = shift_x(j, 1);
		ssz += rho_up(x, x)*rho_up(y, y) + rho_dn(x, x)*rho_dn(y, y);
		ssz -= rho_up(x, x)*rho_dn(y, y) + rho_dn(x, x)*rho_up(y, y);
		ssz -= rho_up(x, y)*rho_up(y, x) + rho_dn(x, y)*rho_dn(y, x);
		spincorrelation[j].add(s*0.25*ssz);
		sum += 0.25*ssz;
	}
	singlet.add(s*sum/V);
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
	double X = 1.0-A*A;
	Matrix_d rho_up = Matrix_d::Identity(V, V) - svdA.inverse();
	Matrix_d rho_dn = svdB.inverse();
	SVDHelper help, flist[N+1], blist[N+1];
	// spin up
	help.setIdentity(V);
	for (int t=0;t<=N;t++) {
		flist[t] = help;
		help.U.applyOnTheLeft(freePropagator_open);
		help.S *= std::exp(+dt*B*0.5+dt*mu);
		help.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonals[(t+t0)%N]).array()).matrix().asDiagonal());
		help.absorbU();
	}
	help.setIdentity(V);
	for (int t=0;t<=N;t++) {
		blist[t] = help;
		help.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)-diagonals[(t+t0)%N]).array()).matrix().asDiagonal());
		help.S *= std::exp(-dt*B*0.5-dt*mu)/X;
		help.U.applyOnTheLeft(freePropagator_inverse);
		help.absorbU();
	}
	for (int t=0;t<=N;t++) {
		help = blist[N-t];
		help.add_svd(flist[t]);
		green_function_up[t].add(s*help.inverse());
	}
	// spin down
	help.setIdentity(V);
	for (int t=0;t<=N;t++) {
		flist[t] = help;
		help.U.applyOnTheLeft(freePropagator_open);
		help.S *= std::exp(-dt*B*0.5+dt*mu);
		help.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonals[(t+t0)%N]).array()).matrix().asDiagonal());
		help.absorbU();
	}
	help.setIdentity(V);
	for (int t=0;t<=N;t++) {
		blist[t] = help;
		help.U.applyOnTheLeft(((Vector_d::Constant(V, 1.0)-diagonals[(t+t0)%N]).array()).matrix().asDiagonal());
		help.S *= std::exp(+dt*B*0.5-dt*mu)/X;
		help.U.applyOnTheLeft(freePropagator_inverse);
		help.absorbU();
	}
	for (int t=0;t<=N;t++) {
		help = flist[N-t];
		help.add_svd(blist[t]);
		green_function_dn[t].add(s*help.inverse());
	}
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


void Simulation::accumulate_forward (int start, int end, Matrix_d &G_up, Matrix_d &G_dn) {
	while (end>N) end -= N;
	for (int i=start;i<end;i++) {
		G_up.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonals[i]).array()).matrix().asDiagonal());
		if (false) {
			svdA.U.applyOnTheLeft(freePropagator_open);
		} else {
			G_up.applyOnTheLeft(freePropagator_x.asDiagonal());
			fftw_execute_dft_r2c(x2p_col, G_up.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
			momentumSpace.applyOnTheLeft((freePropagator/double(V)).asDiagonal());
			fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), G_up.data());
		}
	}
	for (int i=start;i<end;i++) {
		G_dn.applyOnTheLeft(((Vector_d::Constant(V, 1.0)+diagonals[i]).array()).matrix().asDiagonal());
		if (false) {
			G_dn.applyOnTheLeft(freePropagator_open);
		} else {
			G_dn.applyOnTheLeft(freePropagator_x.array().inverse().matrix().asDiagonal());
			fftw_execute_dft_r2c(x2p_col, G_dn.data(), reinterpret_cast<fftw_complex*>(momentumSpace.data()));
			momentumSpace.applyOnTheLeft((freePropagator.array().inverse().matrix()/double(V)).asDiagonal());
			fftw_execute_dft_c2r(p2x_col, reinterpret_cast<fftw_complex*>(momentumSpace.data()), G_dn.data());
		}
	}
}


