#include <Eigen/Dense>
#include <iostream>

extern "C" {
#include <fftw3.h>
}

using namespace std;
using namespace Eigen;

void move_to_fft (const MatrixXd &A, ArrayXcd &a, int step) {
	const int N = A.rows();
	a = ArrayXcd::Zero(N*N*N, 1);
	fftw_plan forward = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(a.data()), reinterpret_cast<fftw_complex*>(a.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	a = ArrayXcd::Zero(N*N*N, 1);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x;
		       	switch (step%6) {
				case 0:
					x = i + j*N;
					break;
				case 1:
					x = -i*N - j*N*N;
					break;
				case 2:
					x = i*N*N + j;
					break;
				case 3:
					x = -i - j*N;
					break;
				case 4:
					x = i*N + j*N*N;
					break;
				case 5:
					x = -i*N*N - j;
					break;
			}
			if (x<0) x += N*N*N;
			a[x] = A(i, j);
		}
	}
	fftw_execute(forward);
	fftw_destroy_plan(forward);
}

void move_from_fft (MatrixXd &A, const ArrayXcd &a, int step) {
	const int N = A.rows();
	ArrayXcd b = ArrayXcd::Zero(N*N*N, 1);
	fftw_plan backward = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(b.data()), reinterpret_cast<fftw_complex*>(b.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
	b = a;
	fftw_execute(backward);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x;
		       	switch (step%6) {
				case 0:
					x = i + j*N;
					break;
				case 1:
					x = -i*N - j*N*N;
					break;
				case 2:
					x = i*N*N + j;
					break;
				case 3:
					x = -i - j*N;
					break;
				case 4:
					x = i*N + j*N*N;
					break;
				case 5:
					x = -i*N*N - j;
					break;
			}
			if (x<0) x += N*N*N;
			A(i, j) = b[x].real();
		}
	}
	A /= N*N*N;
	fftw_destroy_plan(backward);
}

MatrixXd fancy_mm2 (const MatrixXd &A, const MatrixXd &B) {
	const int N = A.rows();

	ArrayXcd a = ArrayXcd::Zero(N*N*N, 1);
	ArrayXcd b = ArrayXcd::Zero(N*N*N, 1);
	ArrayXcd c = ArrayXcd::Zero(N*N*N, 1);

	fftw_plan forwardA = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(a.data()), reinterpret_cast<fftw_complex*>(a.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan forwardB = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(b.data()), reinterpret_cast<fftw_complex*>(b.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan backward = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(c.data()), reinterpret_cast<fftw_complex*>(c.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

	a = ArrayXcd::Zero(N*N*N, 1);
	b = ArrayXcd::Zero(N*N*N, 1);
	c = ArrayXcd::Zero(N*N*N, 1);

	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x = i + j*N;
			if (x<0) x += N*N*N;
			a[x] = A(i, j);
			int y = -i*N - j*N*N;
			if (y<0) y += N*N*N;
			b[y] = B(i, j);
		}
	}

	fftw_execute(forwardA);
	fftw_execute(forwardB);

	c = a*b;

	fftw_execute(backward);
	c /= N*N*N;

	fftw_destroy_plan(forwardA);
	fftw_destroy_plan(forwardB);
	fftw_destroy_plan(backward);

	MatrixXd C = MatrixXd::Zero(N, N);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x = i + j*N*N;
			if (x<0) x += N*N*N;
			C(i, j) = c[x].real();
		}
	}

	return C;
}

int main (int argc, char **argv) {
	const int N = 3;
	const int L = 12;

	MatrixXd A = MatrixXd::Random(N, N);
	cout << A << endl << endl;
	ArrayXcd a;


	for (int i=0;i<L;i++) {
		move_to_fft(A, a, i);
		move_from_fft(A, a, i);
		cout << A << endl << endl;
	}

	return 0;
}

