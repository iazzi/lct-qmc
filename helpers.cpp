#include "helpers.hpp"

#include <cassert>
#include <iostream>

#include <Eigen/QR>
#include <Eigen/SVD>

extern void dggev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta);

Eigen::MatrixXd reduce_f (const std::vector<Eigen::MatrixXd>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(V, V);
	for (auto X : vec) {
		ret.applyOnTheLeft(X);
	}
	return ret;
}

Eigen::MatrixXd reduce_b (const std::vector<Eigen::MatrixXd>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(V, V);
	for (auto X : vec) {
		ret.applyOnTheRight(X);
	}
	return ret;
}

Eigen::MatrixXd reduceSVD_f (std::vector<Eigen::MatrixXd>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner> svd;
	Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(V, V);
	Eigen::MatrixXd Y = Eigen::MatrixXd::Identity(V, V);
	Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(V, V);
	Eigen::ArrayXd D;
	for (auto X : vec) {
		svd.compute(X*Y*Z, Eigen::ComputeFullU | Eigen::ComputeFullV);
		ret.applyOnTheLeft(svd.matrixV().transpose());
		Y = svd.matrixU();
		Z = svd.singularValues().asDiagonal();
		D = svd.singularValues();
	}
	std::cout << Z.diagonal().transpose() << std::endl;
	Eigen::VectorXcd eva;
	Eigen::VectorXd evb;
	dggev(Z, Y.transpose()*ret.transpose(), eva, evb);
	std::cout << (eva.array()/evb.cast<std::complex<double>>().array()).transpose() << std::endl;
	dggev(Y.transpose()*ret.transpose(), Z, eva, evb);
	std::cout << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
	dggev(D.inverse().matrix().asDiagonal(), Y.transpose()*ret.transpose(), eva, evb);
	std::cout << (eva.array()/evb.cast<std::complex<double>>().array()).transpose() << std::endl;
	dggev(Y.transpose()*ret.transpose(), D.inverse().matrix().asDiagonal(), eva, evb);
	std::cout << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
	return Y*Z*ret;
}

Eigen::MatrixXd reduceSVD_b (std::vector<Eigen::MatrixXd>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner> svd;
	Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(V, V);
	Eigen::MatrixXd Y = Eigen::MatrixXd::Identity(V, V);
	Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(V, V);
	for (auto X : vec) {
		svd.compute(Z*(Y*X).eval(), Eigen::ComputeFullU | Eigen::ComputeFullV);
		ret.applyOnTheRight(svd.matrixU());
		Y = svd.matrixV().transpose();
		Z = svd.singularValues().asDiagonal();
	}
	std::cout << Z.diagonal().array().inverse().transpose() << std::endl;
	Eigen::VectorXcd eva;
	Eigen::VectorXd evb;
	dggev(Z, ret.transpose()*Y.transpose(), eva, evb);
	std::cout << (eva.array()/evb.cast<std::complex<double>>().array()).transpose() << std::endl;
	dggev(ret.transpose()*Y.transpose(), Z, eva, evb);
	std::cout << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
	return ret*Z*Y;
}

void test_sequences (std::vector<Eigen::MatrixXd>& fvec, std::vector<Eigen::MatrixXd>& bvec) {
	assert(fvec.size() == bvec.size());
	const int N = fvec.size();
	assert(fvec[0].rows()==fvec[0].cols());
	assert(bvec[0].rows()==bvec[0].cols());
	assert(fvec[0].rows()==bvec[0].rows());
	const int V = fvec[0].rows();
	for (int i=0;i<N;i++) {
		std::cout << i << "th pair are inverse? " << (fvec[i]*bvec[i]).eval().isIdentity() << ", " << (bvec[i]*fvec[i]).eval().isIdentity() << std::endl;
	}
	Eigen::MatrixXd fp = reduce_f(fvec);
	Eigen::MatrixXd bp = reduce_b(bvec);
	std::cout << "straight products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	std::cout << fp << std::endl << std::endl << bp << std::endl << std::endl;
	std::cout << "SVDs: " << std::endl << fp.jacobiSvd().singularValues().transpose() << std::endl << bp.jacobiSvd().singularValues().array().inverse().transpose() << std::endl;
	std::cout << "EVs: " << std::endl << fp.eigenvalues().transpose() << std::endl << bp.eigenvalues().array().transpose() << std::endl;
	fp.setIdentity(V, V);
	bp.setIdentity(V, V);
	for (int i=0;i<N;i++) {
		fp.applyOnTheLeft(fvec[i]);
		bp.applyOnTheRight(bvec[i]);
		std::cout << i << "th accumulated products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	}
	fp = reduceSVD_f(fvec);
	bp = reduceSVD_b(bvec);
	std::cout << "SVD products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	std::cout << fp << std::endl << std::endl << bp << std::endl << std::endl << fp*bp << std::endl << std::endl;
}

