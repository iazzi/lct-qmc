#include <Eigen/Core>
#include "measurements.hpp"
#include <random>
#include <cmath>

using namespace std;
using namespace Eigen;
int main (int argc, char** argv) {
	mymeasurement<ArrayXXd> m;
	mymeasurement<ArrayXXd> n;
	for (int i=0;i<10000;i++) {
		ArrayXXd a = ArrayXXd::Random(1, 2);
		m.add(a);
		n.add_plain(a);
	}
	cout << m << endl;
	cout << n << endl;
	return 0;
}

