#include "named.hpp"

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

template <class T>
void lua_get (lua_State *L, T& t);

template <class ... Args>
void lua_get (lua_State *L, named_tuple<Args...> &t) {
	int dummy[sizeof...(Args)] = {
		(lua_getfield(L, -1, Args::name(t)), lua_get(L, Args::ref(t)), lua_pop(L, 1), 0)...
	};
	(void)dummy;
}

template <class T>
void lua_set (lua_State *L, const T& t);

template <class ... Args>
void lua_set (lua_State *L, const named_tuple<Args...> &t) {
	lua_newtable(L);
	int dummy[sizeof...(Args)] = {
		(lua_set(L, Args::value(t)), lua_setfield(L, -2, Args::name(t)), 0)...
	};
	(void)dummy;
}

template<> void lua_get<int> (lua_State *L, int &v) { v = lua_tointeger(L, -1); }
template<> void lua_get<size_t> (lua_State *L, size_t &v) { v = lua_tointeger(L, -1); }
template<> void lua_get<float> (lua_State *L, float &v) { v = lua_tonumber(L, -1); }
template<> void lua_get<double> (lua_State *L, double &v) { v = lua_tonumber(L, -1); }
template<> void lua_get<char> (lua_State *L, char &v) { v = lua_tostring(L, -1)[0]; }
template<> void lua_get<std::string> (lua_State *L, std::string &v) { v = lua_isstring(L, -1)?lua_tostring(L, -1):""; } // FIXME: embedded '\0'

template<> void lua_set<int> (lua_State *L, const int &v) { lua_pushinteger(L, v); }
template<> void lua_set<size_t> (lua_State *L, const size_t &v) { lua_pushinteger(L, v); }
template<> void lua_set<float> (lua_State *L, const float &v) { lua_pushnumber(L, v); }
template<> void lua_set<double> (lua_State *L, const double &v) { lua_pushnumber(L, v); }
template<> void lua_set<char> (lua_State *L, const char &v) { lua_pushlstring(L, &v, 1); }
template<> void lua_set<std::string> (lua_State *L, const std::string &v) { lua_pushlstring(L, v.data(), v.size()); }

template <class T>
void operator<< (lua_State *L, const std::vector<T>& v) {
	lua_newtable(L);
	for (size_t i=0;i<v.size();i++) {
		L << v[i];
		lua_rawseti(L, -2, i+1);
	}
}

