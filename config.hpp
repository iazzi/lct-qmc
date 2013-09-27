#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "named.hpp"

namespace config {
	named_type(int, Lx);
	named_type(int, Ly);
	named_type(int, Lz);
	named_type(int, N);
	named_type(double, beta);
	named_type(double, U);
	named_type(double, mu);
	named_type(double, B);
	named_type(double, tx);
	named_type(double, ty);
	named_type(double, tz);
	named_type(std::string, type);
	//double Vx, Vy, Vz; // trap strength
	//double staggered_field;
	typedef named_tuple<Lx_t, Ly_t, Lz_t, N_t, beta_t, U_t, mu_t, B_t, tx_t, ty_t, tz_t, type_t> hubbard_config;
}

#endif // CONFIG_HPP

