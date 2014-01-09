#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <complex>

extern "C" {
#include <fftw3.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#define PI atan2(0.0, -1.0)

using namespace std;

int N, Lx, Ly;
double beta;

void average (fftw_complex &a, fftw_complex &b) {
	a[0] += b[0];
	a[0] /= 2.0;
	b[0] = a[0];
	a[1] += b[1];
	a[1] /= 2.0;
	b[1] = a[1];
}

void load_gf (lua_State *L, fftw_complex *G, int Lx, int Ly) {
	int V = Lx*Ly;
	for (int t=0;t<N;t++) {
		lua_rawgeti(L, -1, t);
		for (int x=0;x<V;x++) {
			lua_rawgeti(L, -1, x+1);
			for (int y=0;y<V;y++) {
				lua_rawgeti(L, -1, y+1);
				if (lua_isnumber(L, -1)) {
					G[t*V*V+x*V+y][0] = lua_tonumber(L, -1);
					G[t*V*V+x*V+y][1] = 0.0;
				} else {
					lua_rawgeti(L, -1, 1);
					G[t*V*V+x*V+y][0] = lua_tonumber(L, -1);
					lua_pop(L, 1);
					lua_rawgeti(L, -1, 2);
					G[t*V*V+x*V+y][1] = lua_tonumber(L, -1);
					lua_pop(L, 1);
				}
				//cerr << G_up[t*V*V+x*V+y] << ", ";
				lua_pop(L, 1);
			}
			//cerr << '\n';
			lua_pop(L, 1);
		}
		//cerr << '\n';
		lua_pop(L, 1);
	}
}

void transl_symm (fftw_complex* G, int N, int Lx, int Ly) {
	int V = Lx*Ly;
	for (int t=0;t<N;t++) {
		for (int x=1;x<V;x++) {
			for (int y=0;y<V;y++) {
				int x_a = x/Ly;
				int x_b = x%Ly;
				int y_a = y/Ly;
				int y_b = y%Ly;
				int z_a = (x_a+y_a)%Lx;
				int z_b = (x_b+y_b)%Ly;
				int z = z_a*Ly + z_b;
				G[t*V*V+y][0] += G[t*V*V+x*V+z][0];
				G[t*V*V+y][1] += G[t*V*V+x*V+z][1];
			}
		}
		for (int y=0;y<V;y++) {
			G[t*V*V+y][0] /= V;
			G[t*V*V+y][1] /= V;
		}
		for (int x=1;x<V;x++) {
			for (int y=0;y<V;y++) {
				int x_a = x/Ly;
				int x_b = x%Ly;
				int y_a = y/Ly;
				int y_b = y%Ly;
				int z_a = (x_a+y_a)%Lx;
				int z_b = (x_b+y_b)%Ly;
				int z = z_a*Ly + z_b;
				G[t*V*V+x*V+z][0] = G[t*V*V+y][0];
				G[t*V*V+x*V+z][1] = G[t*V*V+y][1];
			}
		}
	}
}

void symm (fftw_complex* G, int N, int Lx, int Ly) {
	int V = Lx*Ly;
	for (int t=0;t<N;t++) {
		for (int x=0;x<V;x++) {
			for (int y=0;y<V;y++) {
				int x_a = x/Ly;
				int x_b = x%Ly;
				int z_a = (Lx-x_a)%Lx;
				int z_b = (Ly-x_b)%Ly;
				int z = z_a*Ly + z_b;
				int y_a = y/Ly;
				int y_b = y%Ly;
				int w_a = (Lx-y_a)%Lx;
				int w_b = (Ly-y_b)%Ly;
				int w = w_a*Ly + w_b;
				average(G[t*V*V+x*V+y], G[t*V*V+z*V+w]);
			}
		}
	}
}

void invert (fftw_complex &y) {
	complex<double> z(y[0], y[1]);
	z = 1.0/z;
	y[0] = z.real();
	y[1] = z.imag();
}

void invert (fftw_complex* G, int N, int Lx, int Ly) {
	int V = Lx*Ly;
	for (int t=0;t<N;t++) {
		for (int x=0;x<V;x++) {
			for (int y=0;y<V;y++) {
				invert(G[t*V*V+x*V+y]);
			}
		}
	}
}

void flip_row (fftw_complex* G, int N, int Lx, int Ly) {
	int V = Lx*Ly;
	for (int t=0;t<N;t++) {
		for (int x=0;x<V;x++) {
			for (int y=0;y<V;y++) {
				int y_a = y/Ly;
				int y_b = y%Ly;
				int z_a = (Lx-y_a)%Lx;
				int z_b = (Ly-y_b)%Ly;
				int z = z_a*Ly + z_b;
				if (y<z) std::swap(G[t*V*V+x*V+y][0], G[t*V*V+x*V+z][0]);
				if (y<z) std::swap(G[t*V*V+x*V+y][1], G[t*V*V+x*V+z][1]);
			}
		}
	}
}

int main (int argc, char **argv) {
	ofstream out(argv[2]);
	lua_State *L = luaL_newstate();

	luaL_dofile(L, argv[1]);

	lua_getglobal(L, "N");
	N = lua_tointeger(L, -1);
	lua_pop(L, 1);

	lua_getglobal(L, "Lx");
	Lx = lua_tointeger(L, -1);
	lua_pop(L, 1);

	lua_getglobal(L, "Ly");
	Ly = lua_tointeger(L, -1);
	lua_pop(L, 1);

	lua_getglobal(L, "beta");
	beta = lua_tonumber(L, -1);
	lua_pop(L, 1);

	int V = Lx*Ly;

	int size[5] = { Lx, Ly, Lx, Ly };
	fftw_complex *G_up_position = fftw_alloc_complex(N*V*V);
	fftw_complex *G_up_momentum = fftw_alloc_complex(N*V*V);
	fftw_complex *G_dn_position = fftw_alloc_complex(N*V*V);
	fftw_complex *G_dn_momentum = fftw_alloc_complex(N*V*V);
	fftw_plan g_up_plan = fftw_plan_many_dft(4, size, N, G_up_position, NULL, 1, V*V, G_up_momentum, NULL, 1, V*V, FFTW_FORWARD, FFTW_PATIENT);
	fftw_plan g_dn_plan = fftw_plan_many_dft(4, size, N, G_dn_position, NULL, 1, V*V, G_dn_momentum, NULL, 1, V*V, FFTW_FORWARD, FFTW_PATIENT);

	lua_getglobal(L, "G_up");
	load_gf(L, G_up_position, Lx, Ly);
	lua_pop(L, 1);

	lua_getglobal(L, "G_dn");
	load_gf(L, G_dn_position, Lx, Ly);
	lua_pop(L, 1);


	transl_symm(G_up_position, N, Lx, Ly);
	symm(G_up_position, N, Lx, Ly);
	fftw_execute(g_up_plan);
	flip_row(G_up_momentum, N, Lx, Ly);
	for (int x=0;x<V;x++) {
		complex<double> v[N];
		for (int n=0;n<N;n++) {
			v[n] = 0.0;
			complex<double> w = PI*(2*n+1)/beta;
			double dt = beta / N;
			for (int t=0;t<N;t++) {
				complex<double> f(G_up_momentum[t*V*V+x*V+x][0], G_up_momentum[t*V*V+x*V+x][1]);
				double tau = dt*t;
				v[n] += dt * exp(w*tau) * f;
			}
		}
	}
	//invert(G_up_momentum, N, Lx, Ly);

	out.precision(12);
	out << "G_up = {}\n";
	out << "G_dn = {}\n\n";
	for (int t=0;t<N;t++) {
		out << "G_up[" << t << "] = {";
		for (int x=0;x<V;x++) {
			out << " {";
			for (int y=0;y<V;y++) {
				G_up_momentum[t*V*V+x*V+y][0] /= V;
				G_up_momentum[t*V*V+x*V+y][1] /= V;
				out << " { " << G_up_momentum[t*V*V+x*V+y][0] << ", " << G_up_momentum[t*V*V+x*V+y][1] << " }, ";
			}
			out << "}\n";
		}
		out << "}\n";
		//cerr << G_up_momentum[t*V*V][0] << ' ' << G_up_momentum[t*V*V][1] << endl;
	}

	double dt = beta / N;
	for (int t=0;t<N;t++) {
		double tau = dt*t;
		cerr << "# " << t << "\n";
		for (int x=0;x<V;x++) {
			double k_x = 2*PI*(x/Ly)/Lx;
			double k_y = 2*PI*(x%Ly)/Ly;
			if (k_x>PI) k_x -= 2.0*PI;
			if (k_y>PI) k_y -= 2.0*PI;
			double e = -2*(cos(k_x)+cos(k_y))-4;
			cerr << k_x << " " << k_y << " "
				<< G_up_momentum[t*V*V+x*V+x][0]/G_up_momentum[x*V+x][0] << " "
				<< log(G_up_momentum[t*V*V+x*V+x][0]/G_up_momentum[x*V+x][0])/tau << " "
				<< G_up_momentum[t*V*V+x*V+x][0] << ' ' << exp(-tau*e)/(1.0+exp(-beta*e)) << "\n";
				//<< G_up_momentum[x*V+x][0] << ' ' << 1.0/(1.0+exp(-beta*e)) << "\n";
			if ((x+1)%Ly==0) cerr << '\n';
		}
		//cerr << "\n";
	}

#if 0
	out << "G_0 = {}\n\n";
	for (int t=0;t<N;t++) {
		out << "G_0[" << t << "] = {";
		for (int x=0;x<V;x++) {
			int y = x;
			complex<double> z;
			double k_x = 2*PI*(x/Ly)/Lx;
			double k_y = 2*PI*(x%Ly)/Ly;
			z.real(-2*(cos(k_x)+cos(k_y)));
			z.imag(t);
			z = 1.0/z;
			out << " { " << z.real() << ", " << z.imag() << " }, ";
			cerr << z.real() << " " << z.imag() << " ";
		}
		out << "}\n";
		cerr << '\n';
	}
#endif
	fftw_free(g_up_plan);
	lua_close(L);

	return 0;
}
