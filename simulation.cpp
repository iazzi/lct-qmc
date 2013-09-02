#include "simulation.hpp"

// FIXME only works in 2D
void Simulation::prepare_open_boundaries () {
	std::uniform_real_distribution<double> d;
	Matrix_d H = Matrix_d::Zero(V, V);
	for (int x=0;x<Lx;x++) {
		for (int y=0;y<Ly;y++) {
			for (int z=0;z<Lz;z++) {
				int a = x*Ly*Lz + y*Lz + z;
				int b = ((x+1)%Lx)*Ly*Lz + y*Lz + z;
				int c = x*Ly*Lz + ((y+1)%Ly)*Lz + z;
				int d = x*Ly*Lz + y*Lz + (z+1)%Lz;
				if (Lx>1 && x!=Lx-1) H(a, b) = H(b, a) = tx;
				if (Ly>1 && y!=Ly-1) H(a, c) = H(c, a) = ty;
				if (Lz>1 && z!=Lz-1) H(a, d) = H(d, a) = tz;
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
}


void Simulation::prepare_propagators () {
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
}

void Simulation::load (lua_State *L, int index) {
	lua_getfield(L, index, "SEED");
	if (lua_isnumber(L, -1)) {
		generator.seed(lua_tointeger(L, -1));
	} else if (lua_isstring(L, -1)) {
		std::stringstream in(std::string(lua_tostring(L, -1)));
		in >> generator;
	}
	lua_pop(L, 1);
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
	lua_getfield(L, index, "flips_per_update");     flips_per_update = lua_tointeger(L, -1);            lua_pop(L, 1);
	lua_getfield(L, index, "open_boundary");     open_boundary = lua_toboolean(L, -1);            lua_pop(L, 1);
	//lua_getfield(L, index, "update_start");     update_start = lua_tointeger(L, -1);         lua_pop(L, 1);
	//lua_getfield(L, index, "update_end");       update_end = lua_tointeger(L, -1);           lua_pop(L, 1);
	//lua_getfield(L, index, "LOGFILE");  logfile.open(lua_tostring(L, -1));     lua_pop(L, 1);
	init();
}

void Simulation::save (lua_State *L, int index) {
	if (index<1) index = lua_gettop(L)+index;
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
	lua_pushnumber(L, Vx); lua_setfield(L, index, "Vx");
	lua_pushnumber(L, Vy); lua_setfield(L, index, "Vy");
	lua_pushnumber(L, Vz); lua_setfield(L, index, "Vz");
	lua_pushnumber(L, -g); lua_setfield(L, index, "U");
	lua_pushnumber(L, mu); lua_setfield(L, index, "mu");
	lua_pushnumber(L, B); lua_setfield(L, index, "B");
	lua_pushnumber(L, staggered_field); lua_setfield(L, index, "h");
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
	L << measured_sign;
	lua_setfield(L, -2, "measured_sign");
	L << chi_d;
	lua_setfield(L, -2, "chi_d");
	lua_setfield(L, index, "results");
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
	double d1 = ((update_Vt.topRows(L+1)*svd_inverse_up.U) * svd_inverse_up.S.asDiagonal() * (svd_inverse_up.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)).determinant();
	double d2 = ((update_Vt.topRows(L+1)*svd_inverse_dn.U) * svd_inverse_dn.S.asDiagonal() * (svd_inverse_dn.Vt*update_U.leftCols(L+1)) + Matrix_d::Identity(L+1, L+1)).determinant();
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

void Simulation::write_wavefunction (std::ostream &out) {
}




