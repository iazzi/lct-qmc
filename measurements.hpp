#ifndef __MEASUREMENTS_HPP
#define __MEASUREMENTS_HPP

#include <vector>
#include <iostream>

#include <cmath>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

template <typename T, bool Log = false>
class measurement {
	private:
		std::vector<T> sums_;
		std::vector<T> squared_sums_;
		std::vector<T> x_;
		std::vector<int> n_;
		std::string name_;
	public:
		const std::string &name () const { return name_; }
		void set_name (const std::string &name) { name_ = name; }

		void add (const T &x) {
			T nx = x;
			for (size_t i=0;;i++) {
				if (i==n_.size()) {
					sums_.push_back(T());
					squared_sums_.push_back(T());
					x_.push_back(T());
					n_.push_back(0);
					if (Log) break;
				}
				if (n_[i]==0) {
					sums_[i] = nx;
					squared_sums_[i] = nx * nx;
				} else {
					sums_[i] += nx;
					squared_sums_[i] += nx * nx;
				}
				n_[i] += 1;
				if (n_[i]%2==1) {
					x_[i] = nx;
					break;
				} else {
					nx = (nx + x_[i]) / 2.0;
					x_[i] = nx;
				}
			}
		}

		void repeat () { add(x_[0]); }

		T mean (int i = 0) const {
			return sums_[i] / double(n_[i]);
		}

		T variance (int i = 0) const {
			T m = mean(i);
			T m2 = squared_sums_[i] / double(n_[i]);
			return m2 - m*m;
		}

		T error (int i = 0) const {
			return sqrt( variance(i) / double(n_[i]) );
		}

		size_t bins() const { return n_.size(); }
		int samples (int i = 0) const { if (n_.size()==0) return 0; else return n_[i]; }

		double time (int i = 0) const {
			return (variance(i)*n_[0]/n_[i]/variance(0)-1.0)*0.5;
		}

		measurement () : name_("Result") {}

	protected:
};

template <typename T, bool Log> std::ostream& operator<< (std::ostream& out, const measurement<T, Log>& m) {
	if (m.samples()==0) {
		out << m.name() << ": Empty." << std::endl;
	} else {
		int N = m.bins()-6;
		N = N>0?N:0;
		out << m.name() << ": " << m.mean() << " +- " << m.error(N) << std::endl;
		if (N<2 || 2*m.error(N-1)<(m.error(N)+m.error(N-2))) {
			out << "NOT CONVERGING" << std::endl;
		}
		out << "Bins: " << N << std::endl;
		for (int i=0;i<N;i++) {
			out << "#" << i+1 << ": samples = " << m.samples(i) << ", value = " << m.mean(i) << " +- " << m.error(i) << ", autocorrelation time = " << m.time(i) << std::endl;
		}
	}
	return out;
}

template <typename T, bool Log> lua_State* operator<< (lua_State *L, const measurement<T, Log>& m) {
	int t = lua_gettop(L) + 1;
	lua_newtable(L);
	lua_pushlstring(L, m.name().c_str(), m.name().length());
	lua_setfield(L, t, "name");
	lua_pushinteger(L, m.bins());
	lua_setfield(L, t, "bins");
	lua_newtable(L);
	for (size_t i=0;i<m.bins();i++) {
		lua_pushinteger(L, m.samples(i));
		lua_rawseti(L, -2, i+1);
	}
	lua_setfield(L, t, "samples");
	lua_newtable(L);
	for (size_t i=0;i<m.bins();i++) {
		lua_pushnumber(L, m.mean(i));
		lua_rawseti(L, -2, i+1);
	}
	lua_setfield(L, t, "averages");
	lua_newtable(L);
	for (size_t i=0;i<m.bins();i++) {
		lua_pushnumber(L, m.error(i));
		lua_rawseti(L, -2, i+1);
	}
	lua_setfield(L, t, "errors");
	lua_newtable(L);
	for (size_t i=0;i<m.bins();i++) {
		lua_pushnumber(L, m.time(i));
		lua_rawseti(L, -2, i+1);
	}
	lua_setfield(L, t, "time");
	return L;
}

template <typename T, bool Log> lua_State* operator>> (lua_State *L, measurement<T, Log>& m) {
	int t = lua_gettop(L);
	lua_getfield(L, t, "name");
	m.set_name(lua_tostring(L, -1));
	lua_pop(L, 1);
	return L;
}
#endif // __MEASUREMENTS_HPP

