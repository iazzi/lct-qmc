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

void dggev (const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::VectorXcd &alpha, Eigen::VectorXd &beta) {
	Eigen::MatrixXd a = A, b = B;
	int info = 0;
	int N = a.rows();
	Eigen::VectorXd alphar = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd alphai = Eigen::VectorXd::Zero(N);
	alpha = Eigen::VectorXcd::Zero(N);
	beta = Eigen::VectorXd::Zero(N);
	//Eigen::VectorXd vl = Eigen::VectorXd::Zero(1);
	//Eigen::VectorXd vr = Eigen::VectorXd::Zero(1);
	Eigen::VectorXd work = Eigen::VectorXd::Zero(8*N);
	dggev_("N", "N", N, a.data(), N, b.data(), N,
			alphar.data(), alphai.data(), beta.data(),
			NULL, 1, NULL, 1,
			work.data(), work.size(), info);
	if (info == 0) {
		alpha.real() = alphar;
		alpha.imag() = alphai;
	} else if (info<0) {
		std::cerr << "dggev_: error at argument " << -info << std::endl;
	} else if (info<=N) {
		std::cerr << "QZ iteration failed at step " << info << std::endl;
	} else {
	}
}

void sort_vector (Eigen::VectorXcd &v) {
	const int N = v.size();
	for (int i=0;i<N;i++) {
		for (int j=i+1;j<N;j++) {
			if (std::norm(v[i])<std::norm(v[j])) {
				std::complex<double> x = v[j];
				v[j] = v[i];
				v[i] = x;
			}
		}
	}
}

void reverse_vector (Eigen::VectorXcd &v) {
	const int N = v.size();
	for (int i=0;i<N/2;i++) {
		const int j = N-i-1;
		std::complex<double> x = v[j];
		v[j] = v[i];
		v[i] = x;
	}
}

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

void order_evs (Eigen::VectorXcd &ev1, Eigen::VectorXcd &ev2) {
	assert(ev1.size()==ev2.size());
	const int N = ev1.size();
	for (int i=0;i<N;i++) {
		for (int j=i+1;j<N;j++) {
			if (std::norm(ev1[i])<std::norm(ev1[j])) {
				std::complex<double> x = ev1[j];
				ev1[j] = ev1[i];
				ev1[i] = x;
			}
			if (std::norm(ev2[i])>std::norm(ev2[j])) {
				std::complex<double> x = ev2[j];
				ev2[j] = ev2[i];
				ev2[i] = x;
			}
		}
	}
}

Eigen::VectorXcd get_ev_from_qd (const Eigen::MatrixXd &Q, const Eigen::VectorXd &D) {
	assert(Q.rows()==Q.cols());
	assert(D.rows()==Q.cols());
	const int V = Q.rows();
	Eigen::VectorXcd ev1, ev2;
	Eigen::VectorXcd eva;
	Eigen::VectorXd evb;
	dggev(Q, D.asDiagonal(), eva, evb);
	ev1 = eva.array()/evb.cast<std::complex<double>>().array();
	dggev(Q.transpose(), D.array().inverse().matrix().asDiagonal(), eva, evb);
	ev2 = eva.array()/evb.cast<std::complex<double>>().array();
	order_evs(ev1, ev2);
	return ev1;
}

Eigen::VectorXcd merge_ev (Eigen::VectorXcd ev1, Eigen::VectorXcd ev2) {
	assert(ev1.size()==ev2.size());
	const int V = ev1.size();
	sort_vector(ev1);
	sort_vector(ev2);
	reverse_vector(ev2);
	Eigen::VectorXcd ret = Eigen::VectorXcd::Zero(V);
	std::complex<double> lnr = 0.0;
	for (int i=0;i<V;i++) {
		if (std::norm(ev1[i]/ev1[0])<std::norm(ev2[i]/ev2[V-1])) {
			ret[i] = 1.0/ev2[i];
			lnr -= std::log(ev2[i]);
		} else {
			ret[i] = ev1[i];
			lnr += std::log(ev1[i]);
		}
	}
	std::cerr << "guess log = " << lnr << std::endl;
	return ret;
}

std::vector<Eigen::VectorXcd> evlist (std::vector<Eigen::MatrixXd>& vec) {
	assert(vec[0].rows()==vec[0].cols());
	const int V = vec[0].rows();
	std::vector<Eigen::VectorXcd> retlist;
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
		{
			Eigen::VectorXcd eva;
			Eigen::VectorXd evb;
			dggev(Y.transpose()*ret.transpose(), Z, eva, evb);
			std::cerr << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
			dggev(Y.transpose()*ret.transpose(), D.inverse().matrix().asDiagonal(), eva, evb);
			std::cerr << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
		}
	}
	std::cerr << Z.diagonal().transpose() << std::endl;
	return retlist;
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
	Eigen::VectorXcd ev1, ev2;
	std::cerr << Z.diagonal().transpose() << std::endl;
	std::cerr << "alt guess = " << D.array().log().sum() << std::endl;
	Eigen::VectorXcd eva;
	Eigen::VectorXd evb;
	dggev(Z, Y.transpose()*ret.transpose(), eva, evb);
	std::cerr << (eva.array()/evb.cast<std::complex<double>>().array()).transpose() << std::endl;
	dggev(Y.transpose()*ret.transpose(), Z, eva, evb);
	ev1 = evb.cast<std::complex<double>>().array()/eva.array();
	std::cerr << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
	dggev(D.inverse().matrix().asDiagonal(), Y.transpose()*ret.transpose(), eva, evb);
	std::cerr << (eva.array()/evb.cast<std::complex<double>>().array()).transpose() << std::endl;
	dggev(Y.transpose()*ret.transpose(), D.inverse().matrix().asDiagonal(), eva, evb);
	ev2 = evb.cast<std::complex<double>>().array()/eva.array();
	std::cerr << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
	Eigen::VectorXcd ev3 = merge_ev(ev1, ev2);
	std::complex<double> p1 = 1.0;
	std::complex<double> p2 = 1.0;
	std::complex<double> p3 = 1.0;
	double s = 1.0;
	for (int i=0;i<ev3.size()/2;i++) {
		p1 *= ev1[i] * ev1[ev1.size()-1-i];
		p2 *= ev2[i] * ev2[ev2.size()-1-i];
		p3 *= ev3[i] * ev3[ev3.size()-1-i];
		s *= D[i] * D[D.size()-1-i];
	}
	std::cerr << p1 << ' ' << p2 << ' ' << p3 << ' ' << s << std::endl;
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
	std::cerr << Z.diagonal().array().inverse().transpose() << std::endl;
	Eigen::VectorXcd eva;
	Eigen::VectorXd evb;
	dggev(Z, ret.transpose()*Y.transpose(), eva, evb);
	std::cerr << (eva.array()/evb.cast<std::complex<double>>().array()).transpose() << std::endl;
	dggev(ret.transpose()*Y.transpose(), Z, eva, evb);
	std::cerr << (evb.cast<std::complex<double>>().array()/eva.array()).transpose() << std::endl;
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
		//std::cerr << i << "th pair are inverse? " << (fvec[i]*bvec[i]).eval().isIdentity() << ", " << (bvec[i]*fvec[i]).eval().isIdentity() << std::endl;
	}
	Eigen::MatrixXd fp = reduce_f(fvec);
	Eigen::MatrixXd bp = reduce_b(bvec);
	std::cerr << "straight products are inverse? " << (fp*bp).eval().isIdentity(1e-3) << ", " << (bp*fp).eval().isIdentity(1e-3) << std::endl;
	//std::cerr << fp << std::endl << std::endl << bp << std::endl << std::endl;
	std::cerr << "SVDs: " << std::endl << fp.jacobiSvd().singularValues().transpose() << std::endl << bp.jacobiSvd().singularValues().array().inverse().transpose() << std::endl;
	std::cerr << "EVs: " << std::endl << fp.eigenvalues().transpose() << std::endl << bp.eigenvalues().array().transpose() << std::endl;
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

