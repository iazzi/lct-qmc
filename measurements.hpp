#ifndef __MEASUREMENTS_HPP
#define __MEASUREMENTS_HPP

#include <vector>
#include <iostream>

#include <cmath>

template <typename T>
class mymeasurement {
	private:
		std::vector<T> sums_;
		std::vector<T> squared_sums_;
		std::vector<T> x_;
		std::vector<int> n_;
		std::string name_;
	public:
		const std::string &name () const { return name_; }
		void set_name (const std::string &name) { name_ = name; }

		void add_rec (const T &x, size_t i = 0) {
			if (i==n_.size()) {
				sums_.push_back(x);
				squared_sums_.push_back(x*x);
				x_.push_back(T());
				n_.push_back(0);
			} else {
				sums_[i] += x;
				squared_sums_[i] += x * x;
			}
			n_[i] += 1;
			if (n_[i]%2==1) {
				x_[i] = x;
			} else {
				T nx = (x + x_[i]) / 2.0;
				x_[i] = x;
				add(nx, i+1);
			}
		}

		void add (const T &x) {
			T nx = x;
			for (size_t i=0;;i++) {
				if (i==n_.size()) {
					sums_.push_back(nx);
					squared_sums_.push_back(nx*nx);
					x_.push_back(T());
					n_.push_back(0);
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

		int bins() const { return n_.size(); }
		int samples (int i = 0) const { if (n_.size()==0) return 0; else return n_[i]; }

		double time (int i = 0) const {
			return (variance(i)*n_[0]/n_[i]/variance(0)-1.0)*0.5;
		}

		mymeasurement () : name_("Result") {}

	protected:
};

template <typename T> std::ostream& operator<< (std::ostream& out, const mymeasurement<T>& m) {
	if (m.samples()==0) {
		out << m.name() << ": Empty." << std::endl;
	} else {
		int N = m.bins()-7;
		N = N>0?N:0;
		out << m.name() << ": " << m.mean() << " +- " << m.error(N) << std::endl;
		if (N<2 || 2*m.error(N-1)<(m.error(N)+m.error(N-2))) {
			out << "NOT CONVERGING" << std::endl;
		}
		out << "Bins: " << N << std::endl;
		for (int i=0;i<N;i++) {
			out << "#" << i+1 << ": number = " << m.samples(i) << ", error = " << m.error(i) << ", autocorrelation time = " << m.time(i) << std::endl;
		}
	}
	return out;
}

#endif // __MEASUREMENTS_HPP

