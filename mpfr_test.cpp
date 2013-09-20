#include "mpfr.hpp"
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include <cstdlib>

using namespace std;
using namespace Eigen;

int main (int argc, char **argv) {
	const int n = argc>1?atoi(argv[1]):4;
	const int PREC = argc>2?atoi(argv[2]):32;
	Eigen::MatrixXd R = Eigen::MatrixXd::Random(n, n), T1;
	PreciseMatrix A(PREC), B(PREC), U(PREC), V(PREC), Q(PREC), T2(PREC);
	PreciseMatrix wr(PREC), wi(PREC);
	Eigen::EigenSolver<MatrixXd> solver(R);
	A = R;
	//cout << A << endl;
	//cout << solver.eigenvalues().transpose() << endl;
	//A.solve_eigenproblem(wr, wi, U, V);
	//wr.transpose_in_place();
	//wi.transpose_in_place();
	//cout << wr << wi << endl;
	//cout << A << endl;
	//V.applyOnTheLeft(A);
	//V.applyOnTheLeft(U);
	//cout << V << endl;

	//A.reduce_to_hessenberg();
	//A.extract_hessenberg_H(B);
	//A.extract_hessenberg_UV(U, V);
	//cout << B << endl;
	//B.reduce_to_ev_verbose(wr, wi, Q);
	//wr.transpose_in_place(); wi.transpose_in_place(); cout << wr << wi << endl;
	//cout << B << endl;
	//B = Q;
	//B.applyOnTheLeft(Q);
	//cout << B << endl;
	//Q.copy_transpose(U);
	//U.applyOnTheLeft(B);
	//U.applyOnTheLeft(Q);
	//cout << U << endl;
	
	std::vector<int> perm;
	cout << A << endl;
	A.in_place_LU(perm);
	U = MatrixXd::Zero(n, n);
	A.extract_bands(U, 0, n);
	cout << U << endl;
	B = MatrixXd::Identity(n, n);
	A.extract_bands(B, 0, -n);
	cout << B << endl;
	for (auto i : perm) cout << i << ' ';
	cout << endl;
	wr = MatrixXd::Random(n, 1);
	wr.transpose_in_place(); cout << wr << endl; wr.transpose_in_place();
	A.apply_inverse_LU_vector(wr, perm);
	wr.transpose_in_place(); cout << wr << endl; wr.transpose_in_place();
	wr.applyOnTheLeft(U);
	wr.applyOnTheLeft(B);
	wr.permute_rows(perm);
	wr.transpose_in_place(); cout << wr << endl; wr.transpose_in_place();
}

