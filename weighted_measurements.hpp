#ifndef __WEIGHTED_MEASUREMENTS_HPP
#define __WEIGHTED_MEASUREMENTS_HPP

#include <vector>
#include <iostream>

#include <cmath>

template<typename T>
class weighted_measurement {
	private:
		T weight;
		T sum;
		T squared_sum;
	public:
		void add (T x, T w) {
			weight += w;
			sum += w * x;
			squared_sum += w * x * x;
		}

		void print (std::ostream& out) {
			out << "mean = " << mean() << ", variance = " << variance() << std::endl;
		}

		T mean () {
			return sum / weight;
		}

		T variance () {
			double m = mean();
			double m2 = squared_sum;
			return sqrt( m2/weight - m*m );
		}

	protected:
};

#endif // __WEIGHTED_MEASUREMENTS_HPP


