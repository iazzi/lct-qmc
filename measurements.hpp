#ifndef __MEASUREMENTS_HPP
#define __MEASUREMENTS_HPP

#include <vector>
#include <iostream>

#include <cmath>

class mymeasurement {
	private:
		std::vector<double> x_;
		std::vector<double> w_;
		std::vector<double> sums;
		std::vector<double> squared_sums;
		std::vector<double> weights;
	public:
		void add (double x, double w = 1.0, int i = 0) {
			if (i==x_.size()) {
				x_.push_back(0.0);
				w_.push_back(0.0);
				sums.push_back(0.0);
				squared_sums.push_back(0.0);
				weights.push_back(0.0);
			}
			sums[i] += w * x;
			squared_sums[i] += w * w * x * x;
			weights[i] += w;
			if (w_[i]==0.0) {
				w_[i] = w;
				x_[i] = x;
			} else {
				double nw = w + w_[i];
				double nx = (x*w + x_[i]*w_[i]) / nw;
				x_[i] = 0.0;
				w_[i] = 0.0;
				add(nx, 0.5*nw, i+1);
			}
		}

		void print (std::ostream& out) {
			int N = x_.size();
			out << "Bins: " << N << std::endl;
			for (int i=0;i<N;i++) {
				//out << "#" << i+1 << ": sum = " << sums[i] << ", ssum = " << squared_sums[i] << ", weights = " << weights[i] << " (" << w_[i] << ")" << std::endl;
				out << "#" << i+1 << ": mean = " << mean(i) << ", error = " << error(i) << std::endl;
			}
		}

		double mean (int i) {
			return sums[i] / weights[i];
		}

		double error (int i) {
			double m = mean(i);
			double m2 = squared_sums[i] / weights[i];
			return sqrt( (m2 - m*m) / weights[i] );
		}

	protected:
};

#endif // __MEASUREMENTS_HPP

