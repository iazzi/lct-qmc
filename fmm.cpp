#include <Eigen/Dense>
#include <iostream>

extern "C" {
#include <fftw3.h>
}

using namespace std;
using namespace Eigen;

MatrixXd fancy_mm (const MatrixXd &A, const MatrixXd &B) {
	const int N = A.rows();
	ArrayXcd a = ArrayXcd::Zero(N*N*N, 1);
	ArrayXcd b = ArrayXcd::Zero(N*N*N, 1);
	ArrayXcd c = ArrayXcd::Zero(N*N*N, 1);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x = -i + j*N;
			if (x<0) x += N*N*N;
			a[x] = A(i, j);
			int y = -i*N + j*N*N;
			if (y<0) y += N*N*N;
			b[y] = B(i, j);
		}
	}
	fftw_plan forwardA = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(a.data()), reinterpret_cast<fftw_complex*>(a.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan forwardB = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(b.data()), reinterpret_cast<fftw_complex*>(b.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan backward = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(c.data()), reinterpret_cast<fftw_complex*>(c.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(forwardA);
	fftw_execute(forwardB);
	c = a*b;
	fftw_execute(backward);
	fftw_destroy_plan(forwardA);
	fftw_destroy_plan(forwardB);
	fftw_destroy_plan(backward);
	c /= N*N*N;
	MatrixXd C = MatrixXd::Zero(N, N);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x = -i + j*N*N;
			if (x<0) x += N*N*N;
			C(i, j) = c[x].real();
		}
	}
	return C;
}

int main (int argc, char **argv) {
	const int N = 3;
	MatrixXd A = MatrixXd::Random(N, N);
	MatrixXd B = MatrixXd::Random(N, N);

	MatrixXd C = A*B;

	cout << C << endl << endl;

	MatrixXd D =  fancy_mm(A, B);

	cout << D << endl;

	return 0;
}

