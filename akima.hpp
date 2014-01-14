#ifndef AKIMA_HPP
#define AKIMA_HPP

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

template <typename T>
class Akima {
	private:
		std::vector<double> x;
		std::vector<T> y;
		std::vector<T> t;
		int N;
	public:
		std::vector<T> p0, p1, p2, p3;
		Akima (const std::vector<double>& X, const std::vector<T> &Y) : x(X), y(Y), N(X.size()) {
			t.resize(N);
			for (int i=0;i<N;i++) {
				t[i] = slope(i);
			}
			p0.resize(N-1);
			p1.resize(N-1);
			p2.resize(N-1);
			p3.resize(N-1);
			for (int i=0;i<N-1;i++) {
				p0[i] = y[i];
				p1[i] = t[i];
				p2[i] = ( 3.0*(y[i+1]-y[i])/(x[i+1]-x[i])-2.0*t[i]-t[i+1] )/(x[i+1]-x[i]);
				p3[i] = ( t[i]+t[i+1]-2.0*(y[i+1]-y[i])/(x[i+1]-x[i]) )/(x[i+1]-x[i])/(x[i+1]-x[i]);
			}
		}

		T operator() (double z) {
			int j = index(z);
			if (j<0) {
				std::cerr << z << ' ' << j << ' ' << x[0] << ' ' << x[N-1] << std::endl;
				throw "outside bounds";
			}
			double dx = z-x[j];
			return p0[j] + p1[j]*dx + p2[j]*dx*dx + p3[j]*dx*dx*dx;
		}

		int index (double z) {
			if (z<x[0] || z>x[N-1]) return -1;
			int j = 0;
			while (j<N-1 && z>=x[j+1]) j++;
			return j;
		}

		T slope (int j) {
			T m1, m2, m3, m4;
			if (j==0) {
				m3 = (y[j+1]-y[j])/(x[j+1]-x[j]);
				m4 = (y[j+2]-y[j+1])/(x[j+2]-x[j+1]);
				m2 = 2.0*m3-m4;
				m1 = 2.0*m2-m3;
			} else if (j==1) {
				m3 = (y[j+1]-y[j])/(x[j+1]-x[j]);
				m4 = (y[j+2]-y[j+1])/(x[j+2]-x[j+1]);
				m2 = (y[j]-y[j-1])/(x[j]-x[j-1]);
				m1 = 2.0*m2-m3;
			} else if (j==N-2) {
				m1 = (y[j-1]-y[j-2])/(x[j-1]-x[j-2]);
				m2 = (y[j]-y[j-1])/(x[j]-x[j-1]);
				m3 = (y[j+1]-y[j])/(x[j+1]-x[j]);
				m4 = 2.0*m3-m2;
			} else if (j==N-1) {
				m1 = (y[j-1]-y[j-2])/(x[j-1]-x[j-2]);
				m2 = (y[j]-y[j-1])/(x[j]-x[j-1]);
				m3 = 2.0*m2-m1;
				m4 = 2.0*m3-m2;
			} else {
				m1 = (y[j-1]-y[j-2])/(x[j-1]-x[j-2]);
				m2 = (y[j]-y[j-1])/(x[j]-x[j-1]);
				m3 = (y[j+1]-y[j])/(x[j+1]-x[j]);
				m4 = (y[j+2]-y[j+1])/(x[j+2]-x[j+1]);
			}
			T u;
			if (m1==m2 && m3!=m4) u = m2;
			else if (m3==m4 && m1!=m2) u = m3;
			else if (m1==m2 && m3==m4) u = 0.5*(m2+m3);
			else {
				T w1 = std::abs(m2-m1);
				T w2 = std::abs(m4-m3);
				u = (w1*m3+w2*m2)/(w1+w2);
			}
			return u;
		}
};

#endif // AKIMA_HPP

