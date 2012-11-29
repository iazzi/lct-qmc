#include <Eigen/Dense>
#include <iostream>

extern "C" {
#include <fftw3.h>
}

using namespace std;
using namespace Eigen;

enum Strides {
	N01, N02,
	N10, N12,
	N20, N21,
};

enum Signs {
	Spp, Spm,
	Smp, Smm,
};

void fill_steps (int N, Strides strides, Signs signs, int &step1, int &step2) {
	switch (strides) {
		case N01:
			step1 = 1;
			step2 = N;
			break;
		case N02:
			step1 = 1;
			step2 = N*N;
			break;
		case N10:
			step1 = N;
			step2 = 1;
			break;
		case N12:
			step1 = N;
			step2 = N*N;
			break;
		case N20:
			step1 = N*N;
			step2 = 1;
			break;
		case N21:
			step1 = N*N;
			step2 = N;
			break;
	}
	if (signs==Smp || signs==Smm) {
		step1 = -step1;
	}
	if (signs==Spm || signs==Smm) {
		step2 = -step2;
	}
}

void move_to_fft (const MatrixXd &A, ArrayXcd &a, Strides strides, Signs signs) {
	const int N = A.rows();
	int step1, step2;
	fill_steps(N, strides, signs, step1, step2);
	a = ArrayXcd::Zero(N*N*N, 1);
	fftw_plan forward = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(a.data()), reinterpret_cast<fftw_complex*>(a.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	a = ArrayXcd::Zero(N*N*N, 1);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x = i*step1 + j*step2;
			if (x<0) x += N*N*N;
			a[x] = A(i, j);
		}
	}
	fftw_execute(forward);
	fftw_destroy_plan(forward);
}

void move_from_fft (MatrixXd &A, const ArrayXcd &a, Strides strides, Signs signs) {
	const int N = A.rows();
	int step1, step2;
	fill_steps(N, strides, signs, step1, step2);
	ArrayXcd b = ArrayXcd::Zero(N*N*N, 1);
	fftw_plan backward = fftw_plan_dft_1d(N*N*N, reinterpret_cast<fftw_complex*>(b.data()), reinterpret_cast<fftw_complex*>(b.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
	b = a;
	fftw_execute(backward);
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			int x = i*step1 + j*step2;
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
	int N = 3;
	int L = 1;

	cin >> N >> L;

	MatrixXd A = MatrixXd::Random(N, N);
	//cout << A << endl << endl;
	ArrayXcd a;


	for (int i=0;i<24;i++) {
		//move_to_fft(A, a, i%6, i/6);
		//move_from_fft(A, a, i%6, i/6);
		//cout << A << endl << endl;
	}

	MatrixXd B = MatrixXd::Identity(N, N);
	MatrixXd C = B;
	ArrayXcd c;
	move_to_fft(C, c, N20, Smm);

	MatrixXd X = MatrixXd::Random(N, N) + 1 * MatrixXd::Identity(N, N);
	ArrayXcd xev = X.eigenvalues();
	for (int i=0;i<L;i++) {
		ArrayXcd x;
		B *= X;
		if (i%2==0) {
			move_to_fft(X, x, N01, Spm);
			c *= x;
			//cout << c.transpose() << endl;
			//ArrayXcd d = c;
			//d.block(0, 0, N*N, 1) = c.block(0, 0, N*N*N/2, 1) + c.block(N*N*N/2, 0, N*N*N/2, 1);
			//d.block(N*N*N/2, 0, N*N*N/2, 1) = d.block(0, 0, N*N*N/2, 1);
			//d /= 2.0;
			move_from_fft(C, c, N21, Smm);
			move_to_fft(C, c, N21, Smm);
			//cout << c.transpose() << endl;
			//cout << d.transpose() << endl;
		} else {
			move_to_fft(X, x, N10, Spm);
			c *= x;
			move_from_fft(C, c, N20, Smm);
			move_to_fft(C, c, N20, Smm);
		}
		cout << (xev.pow(i+1)).transpose().real() << " " << (xev.pow(i+1)).transpose().imag() << " ";
		cout << (B.eigenvalues().array()).transpose().real() << " " << (B.eigenvalues().array()).transpose().imag() << " ";
		cout << (C.eigenvalues().array()).transpose().real() << " " << (C.eigenvalues().array()).transpose().imag() << endl;
	}

	if (L%2==0) {
		move_from_fft(C, c, N20, Smm);
	} else {
		move_from_fft(C, c, N21, Smm);
	}

	//cout << "standard" << endl << B << endl << endl << B.eigenvalues() << endl << endl;
	//cout << "fancy" << endl << C << endl << endl << C.eigenvalues() << endl << endl;

	return 0;
}

