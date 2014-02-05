#include "named.hpp"

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

static void lua_get (lua_State *L, int &v) { v = lua_tointeger(L, -1); }
static void lua_get (lua_State *L, size_t &v) { v = lua_tointeger(L, -1); }
static void lua_get (lua_State *L, float &v) { v = lua_tonumber(L, -1); }
static void lua_get (lua_State *L, double &v) { v = lua_tonumber(L, -1); }
static void lua_get (lua_State *L, char &v) { v = lua_tostring(L, -1)[0]; }
static void lua_get (lua_State *L, std::string &v) { v = lua_isstring(L, -1)?lua_tostring(L, -1):""; } // FIXME: embedded '\0'

static void lua_set (lua_State *L, int v) { lua_pushinteger(L, v); }
static void lua_set (lua_State *L, size_t v) { lua_pushinteger(L, v); }
static void lua_set (lua_State *L, float v) { lua_pushnumber(L, v); }
static void lua_set (lua_State *L, double v) { lua_pushnumber(L, v); }
static void lua_set (lua_State *L, char v) { lua_pushlstring(L, &v, 1); }
static void lua_set (lua_State *L, std::string v) { lua_pushlstring(L, v.data(), v.size()); }

template <class ... Args>
static void lua_get (lua_State *L, named_tuple<Args...> &t) {
	int dummy[sizeof...(Args)] = {
		(lua_getfield(L, -1, Args::name(t)), lua_get(L, Args::ref(t)), lua_pop(L, 1), 0)...
	};
	(void)dummy;
}

template <class ... Args>
static void lua_set (lua_State *L, named_tuple<Args...> &t) {
	lua_newtable(L);
	int dummy[sizeof...(Args)] = {
		(lua_set(L, Args::value(t)), lua_setfield(L, -2, Args::name(t)), 0)...
	};
	(void)dummy;
}

template <class T>
void operator<< (lua_State *L, const std::vector<T>& v) {
	lua_newtable(L);
	for (size_t i=0;i<v.size();i++) {
		L << v[i];
		lua_rawseti(L, -2, i+1);
	}
}

