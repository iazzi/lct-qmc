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
				if (Lx>1 && x!=Lx-0) H(a, b) = H(b, a) = tx;
				if (Ly>1 && y!=Ly-0) H(a, c) = H(c, a) = ty;
				if (Lz>1 && z!=Lz-0) H(a, d) = H(d, a) = tz;
			}
		}
	}

	Eigen::SelfAdjointEigenSolver<Matrix_d> solver;
	solver.compute(H);
	freePropagator_open = solver.eigenvectors() * (-dt*solver.eigenvalues().array()).exp().matrix().asDiagonal() * solver.eigenvectors().transpose();
	//std::cerr << "() " << solver.eigenvalues().array().sum() << std::endl;
	//std::cout << H << std::endl << std::endl;
	//v_x.setRandom();
	//std::cout << (freePropagator_open*v_x).transpose() << std::endl;
	//apply_propagator_vector();
	//std::cout << v_x.transpose() << std::endl;
	//throw 1;
	hamiltonian = H;
}


void Simulation::prepare_propagators () {
	energies = Vector_d::Zero(V);
	freePropagator = Vector_d::Zero(V);
	freePropagator_b = Vector_d::Zero(V);
	//potential = Vector_d::Zero(V);
	//freePropagator_x = Vector_d::Zero(V);
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
		energies[i] += -2.0 * ( tx * cos(2.0*kx*pi/Kx) + ty * cos(2.0*ky*pi/Ky) + tz * cos(2.0*kz*pi/Kz) );
		freePropagator[i] = exp(-dt*energies[i]);
		freePropagator_b[i] = exp(dt*energies[i]);
		//potential[i] = (x+y+z)%2?-staggered_field:staggered_field;
		//freePropagator_x[i] = exp(-dt*potential[i]);
		//freePropagator_x_b[i] = exp(dt*potential[i]);
		staggering[i] = (x+y+z)%2?-1.0:1.0;
	}
}

void Simulation::init () {
	if (Lx<2) { Lx = 1; tx = 0.0; }
	if (Ly<2) { Ly = 1; ty = 0.0; }
	if (Lz<2) { Lz = 1; tz = 0.0; }
	V = Lx * Ly * Lz;
	mslices = mslices>0?mslices:N;
	mslices = mslices<N?mslices:N;
	time_shift = 0 * mslices;
	if (max_update_size<1) max_update_size = 1;
	if (flips_per_update<1) flips_per_update = max_update_size;
	randomPosition = std::uniform_int_distribution<int>(0, V-1);
	randomTime = std::uniform_int_distribution<int>(0, N-1);
	randomStep = std::uniform_int_distribution<int>(0, mslices-1);
	dt = beta/N;
	A = sqrt(exp(g*dt)-1.0);
	diagonals.insert(diagonals.begin(), N, Vector_d::Zero(V));
	distribution = std::bernoulli_distribution(0.5);
	for (size_t i=0;i<diagonals.size();i++) {
		diagonals[i] = Array_d::Constant(V, distribution(generator)?A:-A);
		for (int j=0;j<V;j++) {
			diagonals[i][j] = distribution(generator)?A:-A;
			//diagonals[i][j] = A;
			//diagonals[i][j] = i<N/4.7?-A:A;
			//diagonals[i][j] = A;
			//if ((i*3+j)%7==1) diagonals[i][j] *= -1.0;
		}
	}
	v_x.setZero(V);
	v_p.setZero(V);

	positionSpace.setIdentity(V, V);

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
	g = -config.U;
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
	//lua_getfield(L, index, "h");    staggered_field = lua_tonumber(L, -1);     lua_pop(L, 1);
	lua_getfield(L, index, "RESET");  reset = lua_toboolean(L, -1);            lua_pop(L, 1);
	//lua_getfield(L, index, "REWEIGHT");  reweight = lua_tointeger(L, -1);      lua_pop(L, 1);
	lua_getfield(L, index, "OUTPUT");  outfn = lua_tostring(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "SLICES");  mslices = lua_tointeger(L, -1);         lua_pop(L, 1);
	lua_getfield(L, index, "SVD");     msvd = lua_tointeger(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "max_update_size");     max_update_size = lua_tointeger(L, -1);            lua_pop(L, 1);
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
	lua_pushinteger(L, max_update_size); lua_setfield(L, index, "max_update_size");
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
	lua_getfield(L, -1, "sigma");
	for (int i=0;i<N;i++) {
		for (int j=0;j<V;j++) {
			lua_rawgeti(L, -1, i*V+j+1);
			diagonals[i][j] = lua_tonumber(L, -1);
			lua_pop(L, 1);
			//std::cerr << (diagonals[i][j]<0.0?'-':'+') << ' ';
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

std::pair<double, double> Simulation::rank1_probability (int x, int t) {
	compute_uv_f_short(x, t);
	//diagonal(t)[x] = -diagonal(t)[x];
	//accumulate_forward(0, mslices);
	//diagonal(t)[x] = -diagonal(t)[x];
	//std::cerr << (slices[0]+cache.u_smart*cache.v_smart.transpose()-positionSpace).norm() << std::endl;
	const int L = update_size;
	update_U.col(L) = first_slice_inverse * cache.u_smart;
	update_Vt.row(L) = cache.v_smart.transpose();
	double d1, d2;
	if (true) {
		d1 = (update_Vt.topRows(L+1)*update_U.leftCols(L+1) - (update_Vt.topRows(L+1)*svd_inverse_up.U) * svd_inverse_up.S.asDiagonal() * (svd_inverse_up.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)).determinant();
		d2 = (update_Vt.topRows(L+1)*update_U.leftCols(L+1) - (update_Vt.topRows(L+1)*svd_inverse_dn.U) * svd_inverse_dn.S.asDiagonal() * (svd_inverse_dn.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)).determinant();
	} else {
		d1 = ((update_Vt.topRows(L+1)*svd_inverse_up.U) * svd_inverse_up.S.asDiagonal() * (svd_inverse_up.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)).determinant();
		d2 = ((update_Vt.topRows(L+1)*svd_inverse_dn.U) * svd_inverse_dn.S.asDiagonal() * (svd_inverse_dn.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)).determinant();
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
	//double d1 = ( (update_Vt.topRows(L+1)*svdA.Vt.transpose())*svdA.S.array().inverse().matrix().asDiagonal()*(svdA.U.transpose()*update_U.leftCols(L+1))*std::exp(+beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L+1, L+1) ).determinant();
	//double d2 = ( (update_Vt.topRows(L+1)*svdB.Vt.transpose())*svdB.S.array().inverse().matrix().asDiagonal()*(svdB.U.transpose()*update_U.leftCols(L+1))*std::exp(-beta*B*0.5+beta*mu) + Eigen::MatrixXd::Identity(L+1, L+1) ).determinant();
	//std::cerr << "rank 1 update  size " << L <<  " (" << x << ", " << t << ')' << ' ' << d2 << " [" << svd.S.array().log().sum() << ']' << std::endl;
	//std::cerr << update_Vt.topRows(L+1) << std::endl << std::endl;
	//std::cerr << svd.S.transpose() << std::endl;
	//std::cerr << svd_inverse.S.transpose() << std::endl;
	//std::cerr << svd_inverse_dn.S.transpose() << std::endl;
	//std::cerr << ((update_Vt.topRows(L+1)*svd_inverse_dn.U) * svd_inverse_dn.S.asDiagonal() * (svd_inverse_dn.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)) << std::endl;
	//std::cerr << "rank 1 finished" << std::endl;
	return std::pair<double, double>(std::log(d1)+std::log(d2), s);
}

bool Simulation::metropolis () {
	//std::cerr << "start metropolis step " << svd.S.array().log().sum() << std::endl;
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
		//std::cerr << "accepted metropolis step" << std::endl;
	} else {
		//std::cerr << "rejected metropolis step" << std::endl;
	}
	return ret;
}

double Simulation::ising_energy (int x, int t) {
	double sum = 0.0;
	sum += diagonal((t+1)%N)[x]<0?-1.0:1.0;
	sum += diagonal((t+N-1)%N)[x]<0?-1.0:1.0;
	sum += diagonal(t)[shift_x(x, +1)]<0?-1.0:1.0;
	sum += diagonal(t)[shift_x(x, -1)]<0?-1.0:1.0;
	sum += diagonal(t)[shift_y(x, +1)]<0?-1.0:1.0;
	sum += diagonal(t)[shift_y(x, -1)]<0?-1.0:1.0;
	return diagonal(t)[x]<0?sum:-sum;
}

bool Simulation::metropolis_ising () {
	//std::cerr << "start metropolis step " << svd.S.array().log().sum() << std::endl;
	steps++;
	bool ret = false;
	int x = randomPosition(generator);
	int t = randomStep(generator);
	std::pair<double, double> r1 = rank1_probability(x, t);
	double en = ising_energy(x, t);
	ret = -trialDistribution(generator)<en;
	if (ret && r1.second==update_sign) {
		//std::cerr << "accepted" << std::endl;
		diagonal(t)[x] = -diagonal(t)[x];
		slices[t/mslices] += cache.u_smart*cache.v_smart.transpose();
		update_size++;
		update_prob = r1.first;
		update_sign = r1.second;
		std::cerr << "accepted ising metropolis step sign=" << r1.second << std::endl;
	} else {
		std::cerr << "rejected ising metropolis step sign=" << r1.second << std::endl;
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

bool Simulation::anneal_ising () {
	bool ret = false;
	for (int t=0;t<mslices;t++)
		for (int x=0;x<V;x++) {
			collapse_updates();
			if (diagonal(t)[x]<=0) continue;
			std::pair<double, double> r1 = rank1_probability(x, t);
			double en = ising_energy(x, t);
			ret = 0.0 < en;
			if (r1.second==update_sign) {
				//std::cerr << "accepted" << std::endl;
				diagonal(t)[x] = -diagonal(t)[x];
				slices[t/mslices] += cache.u_smart*cache.v_smart.transpose();
				update_size++;
				update_prob = r1.first;
				update_sign = r1.second;
				std::cerr << "accepted ising anneal step sign=" << r1.second << std::endl;
			} else {
				std::cerr << "rejected ising anneal step sign=" << r1.second << std::endl;
			}
		}
	return ret;
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
	for (auto s:slices) {
		A.applyOnTheLeft(s);
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
	const double dtau = beta/slices.size();
	for (const Matrix_d& U : slices) {
		//F_up.applyOnTheLeft(U*std::exp(+dtau*B*0.5+dtau*mu));
		//F_dn.applyOnTheLeft(U*std::exp(-dtau*B*0.5+dtau*mu));
		//d_wave_chi += pair_correlation(F_up, F_dn);
	}
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
	//if (staggered_field!=0.0) staggered_magnetization.add(s*(rho_up.diagonal().array()*staggering - rho_dn.diagonal().array()*staggering).sum()/V);
}



