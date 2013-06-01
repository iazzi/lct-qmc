#include "helpers.hpp"

#include <cassert>
#include <iostream>

#include <Eigen/QR>
#include <Eigen/SVD>

extern "C" void dggev_ (const char *jobvl, const char *jobvr, const int &N,
			double *A, const int &lda, double *B, const int &ldb,
			double *alphar, double *alphai, double *beta,
			double *vl, const int &ldvl, double *vr, const int &ldvr,
			double *work, const int &lwork, int &info);

void dggev (const Matrix_d &A, const Matrix_d &B, Vector_cd &alpha, Vector_d &beta) {
	Eigen::MatrixXd a = A.cast<double>(), b = B.cast<double>();
	int info = 0;
	int N = a.rows();
	Eigen::VectorXd alphar = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd alphai = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd betar = Eigen::VectorXd::Zero(N);
	alpha = Vector_cd::Zero(N);
	beta = Vector_d::Zero(N);
	//Vector_d vl = Vector_d::Zero(1);
	//Vector_d vr = Vector_d::Zero(1);
	Eigen::VectorXd work = Eigen::VectorXd::Zero(8*N);
	dggev_("N", "N", N, a.data(), N, b.data(), N,
			alphar.data(), alphai.data(), betar.data(),
			NULL, 1, NULL, 1,
			work.data(), work.size(), info);
	if (info == 0) {
		alpha.real() = alphar.cast<Real>();
		alpha.imag() = alphai.cast<Real>();
		beta = betar.cast<Real>();
	} else if (info<0) {
		std::cerr << "dggev_: error at argument " << -info << std::endl;
	} else if (info<=N) {
		std::cerr << "QZ iteration failed at step " << info << std::endl;
	} else {
	}
}

extern "C" void dgesvd_ (const char *jobu, const char *jobvt,
		const int &M, const int &N,
		double *A, const int &lda,
		double *S,
		double *U, const int &ldu,
		double *VT, const int &ldvt,
		double *work, const int &lwork, int &info);

void dgesvd (const Eigen::MatrixXd &A, Eigen::VectorXd &S, Eigen::MatrixXd &U, Eigen::MatrixXd &V) {
	const int N = A.rows();
	Eigen::MatrixXd a = A.cast<double>();
	//Eigen::VectorXd s = Eigen::VectorXd::Zero(N);
	//Eigen::MatrixXd u = Eigen::MatrixXd::Zero(N, N);
	//Eigen::MatrixXd vt = Eigen::MatrixXd::Zero(N, N);
	Eigen::ArrayXd work = Eigen::ArrayXd::Zero(5*N);
	//S = Eigen::VectorXd::Zero(N);
	//U = Eigen::MatrixXd::Zero(N, N);
	//V = Eigen::MatrixXd::Zero(N, N);
	int info = 0;
	dgesvd_("A", "A", N, N, a.data(), N,
			S.data(), U.data(), N, V.data(), N,
			work.data(), 5*N, info);
	if (info == 0) {
		//S = s.cast<Real>();
		//U = u.cast<Real>();
		//V = vt.cast<Real>();
		V.transposeInPlace();
		//std::cerr << "SVD of matrix\n" << A << "\nU*S*V^t\n" << u*s.asDiagonal()*vt << std::endl;
	} else if (info<0) {
		std::cerr << "dgesvd_: error at argument " << -info << std::endl;
	} else if (info<=N) {
		std::cerr << "DBDSQR iteration failed at superdiagonal " << info << std::endl;
	} else {
	}
}

void sort_vector (Vector_cd &v) {
	const int N = v.size();
	for (int i=0;i<N;i++) {
		for (int j=i+1;j<N;j++) {
			if (std::norm(v[i])<std::norm(v[j])) {
				Complex x = v[j];
				v[j] = v[i];
				v[i] = x;
			}
		}
	}
}

void reverse_vector (Vector_cd &v) {
	const int N = v.size();
	for (int i=0;i<N/2;i++) {
		const int j = N-i-1;
		Complex x = v[j];
		v[j] = v[i];
		v[i] = x;
	}
}

void sort_vector (Vector_cd &v, Vector_d &u) {
	const int N = v.size();
	for (int i=0;i<N;i++) {
		for (int j=i+1;j<N;j++) {
			if (std::norm(u[i]*v[j])<std::norm(u[j]*v[i])) {
				Complex x = v[j];
				v[j] = v[i];
				v[i] = x;
				Real y = u[j];
				u[j] = u[i];
				u[i] = y;
			}
		}
	}
}

void reverse_vector (Vector_cd &v, Vector_d &u) {
	const int N = v.size();
	for (int i=0;i<N/2;i++) {
		const int j = N-i-1;
		Complex x = v[j];
		v[j] = v[i];
		v[i] = x;
		Real y = u[j];
		u[j] = u[i];
		u[i] = y;
	}
}

Matrix_d reduce_f (const std::vector<Matrix_d>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Matrix_d ret = Matrix_d::Identity(V, V);
	for (auto X : vec) {
		ret.applyOnTheLeft(X);
	}
	return ret;
}

Matrix_d reduce_b (const std::vector<Matrix_d>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Matrix_d ret = Matrix_d::Identity(V, V);
	for (auto X : vec) {
		ret.applyOnTheRight(X);
	}
	return ret;
}

Vector_cd merge_ev (Vector_cd ev1, Vector_cd ev2) {
	assert(ev1.size()==ev2.size());
	const int V = ev1.size();
	sort_vector(ev1);
	sort_vector(ev2);
	reverse_vector(ev2);
	Vector_cd ret = Vector_cd::Zero(V);
	std::complex<double> lnr = 0.0;
	for (int i=0;i<V;i++) {
		if (std::norm(ev1[i]/ev1[0])<std::norm(ev2[i]/ev2[V-1])) {
			ret[i] = ((Real)1.0)/ev2[i];
			lnr -= std::log(ev2[i]);
		} else {
			ret[i] = ev1[i];
			lnr += std::log(ev1[i]);
		}
	}
	std::cerr << "guess evs = " << ret.transpose() << std::endl;
	std::cerr << "guess log = " << lnr << std::endl;
	return ret;
}

Vector_cd merge_ev_g (Vector_cd eva1, Vector_d evb1, Vector_cd eva2, Vector_d evb2) {
	assert(eva1.size()==eva2.size());
	assert(eva1.size()==evb1.size());
	assert(eva2.size()==evb2.size());
	const int V = eva1.size();
	sort_vector(eva1, evb1);
	sort_vector(eva2, evb2);
	reverse_vector(eva2, evb2);
	Vector_cd ret = Vector_cd::Zero(V);
	std::complex<double> lnr = 0.0;
	for (int i=0;i<V;i++) {
		if (std::norm(eva1[i]*evb1[0]/eva1[0]/evb1[i])>std::norm(eva2[i]*evb2[V-1]/eva2[V-1]/evb2[i])) {
			ret[i] = evb2[i]/eva2[i];
			lnr -= std::log(eva2[i]) - std::log(evb2[i]);
		} else {
			ret[i] = eva1[i]/evb1[i];
			lnr += std::log(eva1[i]) - std::log(evb1[i]);
		}
	}
	std::cerr << "guess evs = " << ret.transpose() << std::endl;
	std::cerr << "guess log = " << lnr << std::endl;
	return ret;
}

Matrix_d reduceSVD_f (std::vector<Matrix_d>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Eigen::JacobiSVD<Matrix_d, Eigen::NoQRPreconditioner> svd;
	Matrix_d ret = Matrix_d::Identity(V, V);
	Matrix_d X = Matrix_d::Identity(V, V);
	Matrix_d Y = Matrix_d::Identity(V, V);
	Matrix_d Z = Matrix_d::Identity(V, V);
	Vector_d D;
	for (auto X : vec) {
		dgesvd((X*Y).eval()*Z, D, Y, X);
		//svd.compute((X*Y).eval()*Z, Eigen::ComputeFullU | Eigen::ComputeFullV);
		ret.applyOnTheLeft(X.transpose());
		//Y = svd.matrixU();
		//D = svd.singularValues();
		Z = D.asDiagonal();
	}
	Vector_cd ev1, ev2;
	std::cerr << Z.diagonal().transpose() << std::endl;
	std::cerr << "alt guess = " << D.array().log().sum() << std::endl;
	Vector_cd eva1, eva2;
	Vector_d evb1, evb2;

	//dggev(Y.transpose()*ret.transpose(), Z, eva1, evb1);
	//ev1 = evb1.cast<std::complex<double>>().array()/eva1.array();
	//std::cerr << (evb1.cast<std::complex<double>>().array()/eva1.array()).transpose() << std::endl;
	//dggev(Y.transpose()*ret.transpose(), D.inverse().matrix().asDiagonal(), eva2, evb2);
	//ev2 = evb2.cast<std::complex<double>>().array()/eva2.array();
	//std::cerr << (evb2.cast<std::complex<double>>().array()/eva2.array()).transpose() << std::endl;

	dggev(Z, Y.transpose()*ret.transpose(), eva1, evb1);
	std::cerr << (eva1.array()/evb1.cast<Complex>().array()).transpose() << std::endl;
	dggev(D.array().inverse().matrix().asDiagonal(), Y.transpose()*ret.transpose(), eva2, evb2);
	std::cerr << (eva2.array()/evb2.cast<Complex>().array()).transpose() << std::endl;

	//Vector_cd ev3 = merge_ev_g(eva1, evb1, eva2, evb2);
	//Vector_cd ev3 = merge_ev(evb1.cast<std::complex<double>>().array()/eva1.array(), evb2.cast<std::complex<double>>().array()/eva2.array());

	Eigen::EigenSolver<Matrix_d> solver;
	Vector_cd ev3 = merge_ev(eva1.array()/evb1.cast<Complex>().array(), eva2.array()/evb2.cast<Complex>().array());
	//std::complex<double> p1 = 1.0;
	//std::complex<double> p2 = 1.0;
	//std::complex<double> p3 = 1.0;
	//double s = 1.0;
	//for (int i=0;i<ev3.size()/2;i++) {
		//p1 *= ev1[i] * ev1[ev1.size()-1-i];
		//p2 *= ev2[i] * ev2[ev2.size()-1-i];
		//p3 *= ev3[i] * ev3[ev3.size()-1-i];
		//s *= D[i] * D[D.size()-1-i];
	//}
	//std::cerr << p1 << ' ' << p2 << ' ' << p3 << ' ' << s << std::endl;
	return Y*Z*ret;
}

void collapseSVD (std::vector<Matrix_d>& vec, Vector_d &S, Matrix_d &U, Matrix_d &V) {
	assert(vec[0].rows()==vec[0].cols());
	const int N = vec[0].rows();
	Matrix_d ret = Matrix_d::Identity(N, N);
	Matrix_d W = Matrix_d::Identity(N, N);
	Matrix_d Y = Matrix_d::Identity(N, N);
	Matrix_d Z = Matrix_d::Identity(N, N);
	Vector_d D = Vector_d::Ones(N);
	for (auto X : vec) {
		dgesvd((X*Y).eval()*D.asDiagonal(), D, Y, W);
		ret.applyOnTheLeft(W.transpose());
		//Z = D.asDiagonal();
		//std::cerr << "add determinant: " << X.determinant() << " -> " << D.array().log().sum() << std::endl;
	}
	//std::cerr << Z.diagonal().transpose() << std::endl;
	//std::cerr << "collapseSVD guess = " << D.array().log().sum() << std::endl;
	S = D.cast<Real>();
	U = Y.cast<Real>();
	V = ret.cast<Real>().transpose();
	ret = Matrix_d::Identity(N, N);
	//for (auto X : vec) {
		//ret.applyOnTheLeft(X);
		//std::cerr << "add determinant: " << X.determinant() << " -> " << ret.determinant() << std::endl;
	//}
}

Matrix_d reduceSVD_b (std::vector<Matrix_d>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	Eigen::JacobiSVD<Matrix_d, Eigen::NoQRPreconditioner> svd;
	Matrix_d ret = Matrix_d::Identity(V, V);
	Matrix_d Y = Matrix_d::Identity(V, V);
	Matrix_d Z = Matrix_d::Identity(V, V);
	for (auto X : vec) {
		svd.compute(Z*(Y*X).eval(), Eigen::ComputeFullU | Eigen::ComputeFullV);
		ret.applyOnTheRight(svd.matrixU());
		Y = svd.matrixV().transpose();
		Z = svd.singularValues().asDiagonal();
	}
	std::cerr << Z.diagonal().array().inverse().transpose() << std::endl;
	Vector_cd eva;
	Vector_d evb;
	dggev(Z, ret.transpose()*Y.transpose(), eva, evb);
	std::cerr << (eva.array()/evb.cast<Complex>().array()).transpose() << std::endl;
	dggev(ret.transpose()*Y.transpose(), Z, eva, evb);
	std::cerr << (evb.cast<Complex>().array()/eva.array()).transpose() << std::endl;
	return ret*Z*Y;
}

void test_sequences (std::vector<Matrix_d>& fvec, std::vector<Matrix_d>& bvec) {
	assert(fvec.size() == bvec.size());
	const int N = fvec.size();
	assert(fvec[0].rows()==fvec[0].cols());
	assert(bvec[0].rows()==bvec[0].cols());
	assert(fvec[0].rows()==bvec[0].rows());
	const int V = fvec[0].rows();
	for (int i=0;i<N;i++) {
		//std::cerr << i << "th pair are inverse? " << (fvec[i]*bvec[i]).eval().isIdentity() << ", " << (bvec[i]*fvec[i]).eval().isIdentity() << std::endl;
	}
	Matrix_d fp = reduce_f(fvec);
	Matrix_d bp = reduce_b(bvec);
	std::cerr << "straight products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	//std::cerr << fp << std::endl << std::endl << bp << std::endl << std::endl;
	std::cerr << "SVDs: " << std::endl << fp.jacobiSvd().singularValues().transpose() << std::endl << bp.jacobiSvd().singularValues().array().inverse().transpose() << std::endl;
	std::cerr << "EVs: " << std::endl << fp.eigenvalues().transpose() << std::endl << bp.eigenvalues().transpose() << std::endl;
	fp.setIdentity(V, V);
	bp.setIdentity(V, V);
	for (int i=0;i<N;i++) {
		//fp.applyOnTheLeft(fvec[i]);
		//bp.applyOnTheRight(bvec[i]);
		//std::cerr << i << "th accumulated products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	}
	fp = reduceSVD_f(fvec);
	bp = reduceSVD_b(bvec);
	std::cerr << "SVD products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	//std::cerr << fp << std::endl << std::endl << bp << std::endl << std::endl << fp*bp << std::endl << std::endl;
}

