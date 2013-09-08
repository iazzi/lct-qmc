#include "mpfr.hpp"
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#define PREC 32

using namespace std;
using namespace Eigen;

int main () {
	Eigen::MatrixXd R = Eigen::MatrixXd::Random(6, 6);
	//Eigen::PartialPivLU<Eigen::MatrixXd> LU(R);
	PreciseMatrix A(PREC), B(PREC), U(PREC), V(PREC);
	Eigen::HessenbergDecomposition<MatrixXd> hd(R);
	Eigen::EigenSolver<MatrixXd> solver(R);
	//R = hd.matrixH();
	A = R;
	//B = Eigen::MatrixXd::Identity(3, 4);
	//cout << A << endl;
	//cout << B << endl;
	//A.applyOnTheLeft(Eigen::MatrixXd::Identity(3, 4));
	//A.inPlaceLU();
	//cout << A << endl;
	//cout << LU.matrixLU() << endl << endl;
	//R += R.triangularView<Eigen::StrictlyUpper>();
	//R += R.triangularView<Eigen::StrictlyUpper>();
	//R += R.triangularView<Eigen::StrictlyUpper>();
	//R += R.triangularView<Eigen::StrictlyUpper>();
	//cout << R << endl << endl;
	//A = R;
	cout << A << endl;
	//A.balance();
	cout << solver.eigenvalues().transpose() << endl;
	A.reduce_to_hessenberg();
	A.extract_hessenberg_H(B);
	A.extract_hessenberg_UV(U, V);
	V.applyOnTheLeft(U);
	cout << V << endl;
	A.extract_hessenberg_UV(U, V);
	V.applyOnTheLeft(B);
	V.applyOnTheLeft(U);
	cout << V << endl;
	B.copy_into(R);
	PreciseMatrix wr(PREC), wi(PREC);
	B.reduce_to_ev(wr, wi);
	cout << wr << endl;
	cout << wi << endl;
	cout << R << endl << endl;
	cout << R.eigenvalues().transpose() << endl;
	//cout << hd.matrixH() << endl << endl;
}

