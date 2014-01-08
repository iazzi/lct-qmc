#ifndef SVD_HPP
#define SVD_HPP

#include <Eigen/QR>
#include <Eigen/SVD>

#include <iostream>

extern "C" void dgesvd_ (const char *jobu, const char *jobvt,
		const int &M, const int &N,
		double *A, const int &lda,
		double *S,
		double *U, const int &ldu,
		double *VT, const int &ldvt,
		double *work, const int &lwork, int &info);

extern "C" void dggev_ (const char *jobvl, const char *jobvr,
		const int &N,
		double *A, const int &lda,
		double *B, const int &ldb,
		double *alphar,
		double *alphai,
		double *beta,
		double *VL, const int &ldvl,
		double *VR, const int &ldvr,
		double *work, const int &lwork, int &info);

struct SVDHelper {
	typedef Eigen::VectorXd Vector;
	typedef Eigen::MatrixXd Matrix;
	typedef Eigen::ArrayXd Array;
	Array work;
	Matrix U;
	Vector S;
	Matrix Vt;
	Matrix other;

	void setSize (int outer, int inner) {
		U.resize(outer, inner);
		S.resize(inner);
		Vt.resize(inner, outer);
		other.resize(inner, inner);
		work.resize(5*inner+outer);
	}

	void setIdentity (int N) {
		U.setIdentity(N, N);
		S.setOnes(N);
		Vt.setIdentity(N, N);
	}

	void reserve (int N) {
		if (work.size()<N) work.resize(N);
	}

	void check_info (int info) {
		if (info == 0) {
			//std::cerr << "reserving working space " << work[0] << std::endl;
			reserve(work[0]);
		} else if (info<0) {
			std::cerr << "dgesvd_: error at argument " << -info << std::endl;
		} else {
			std::cerr << "DBDSQR iteration failed at superdiagonal " << info << std::endl;
		}
	}

	// this leaves a state that is inconsistent with the rest of the class since one of V and U cannot be multiplied by S
	void fullSVD (const Matrix &A) {
		const int M = A.rows();
		const int N = A.cols();
		const int inner = M<N?M:N;
		const int outer = M<N?N:M;
		Matrix B = A;
		int info = 0;
		U.resize(M, M);
		S.resize(inner);
		Vt.resize(N, N);
		reserve(5*inner+outer);
		dgesvd_("A", "A", M, N, B.data(), M, S.data(), U.data(), M, Vt.data(), N, work.data(), work.size(), info);
		check_info(info);
	}

	void thinSVD (const Matrix &A) {
		const int M = A.rows();
		const int N = A.cols();
		const int inner = M<N?M:N;
		const int outer = M<N?N:M;
		Matrix B = A;
		int info = 0;
		U.resize(M, inner);
		S.resize(inner);
		Vt.resize(inner, N);
		reserve(5*inner+outer);
		dgesvd_("S", "S", M, N, B.data(), M, S.data(), U.data(), M, Vt.data(), inner, work.data(), work.size(), info);
		check_info(info);
	}

	void inPlaceSVD (const Matrix &A) {
		const int M = A.rows();
		const int N = A.cols();
		int info = 0;
		if (M>N) {
			const int inner = M<N?M:N;
			const int outer = M<N?N:M;
			U = A;
			S.resize(inner);
			Vt.resize(inner, N);
			reserve(5*inner+outer);
			dgesvd_("O", "S", M, N, U.data(), M, S.data(), U.data(), M, Vt.data(), inner, work.data(), work.size(), info);
		} else {
			const int inner = M<N?M:N;
			const int outer = M<N?N:M;
			U.resize(M, inner);
			S.resize(inner);
			Vt = A;
			reserve(5*inner+outer);
			dgesvd_("S", "O", M, N, Vt.data(), M, S.data(), U.data(), M, Vt.data(), inner, work.data(), work.size(), info);
		}
		check_info(info);
	}

	// this will only work if M>=N
	void absorbU () {
		const int M = U.rows();
		const int N = U.cols();
		int info = 0;
		U.applyOnTheRight(S.asDiagonal());
		const int inner = M<N?M:N;
		const int outer = M<N?N:M;
		other.resize(inner, inner);
		reserve(5*inner+outer);
		dgesvd_("O", "S", M, N, U.data(), M, S.data(), U.data(), M, other.data(), inner, work.data(), work.size(), info);
		check_info(info);
		Vt.applyOnTheLeft(other);
	}

	// this will only work if M<=N
	void absorbVt () {
		const int M = Vt.rows();
		const int N = Vt.cols();
		int info = 0;
		Vt.applyOnTheLeft(S.asDiagonal());
		const int inner = M<N?M:N;
		const int outer = M<N?N:M;
		S.resize(inner);
		other.resize(inner, inner);
		reserve(5*inner+outer);
		if (M<=N) {
			dgesvd_("S", "O", M, N, Vt.data(), M, S.data(), other.data(), M, Vt.data(), inner, work.data(), work.size(), info);
			check_info(info);
			U.applyOnTheRight(other);
		} else {
			dgesvd_("O", "S", M, N, Vt.data(), M, S.data(), Vt.data(), M, other.data(), inner, work.data(), work.size(), info);
			check_info(info);
			U.applyOnTheRight(Vt);
			Vt = other;
		}
	}

	// TODO size constraints!
	void rank1_update (const Vector &u, const Vector &v, double lambda = 1.0) {
		const int N = S.size();
		int info = 0;
		Matrix A = U;
		Matrix B = Vt;
		other = lambda * (U.transpose()*u) * (v.transpose() * Vt.transpose());
		other.diagonal() += S;
		dgesvd_("A", "A", N, N, other.data(), N, S.data(), U.data(), N, Vt.data(), N, work.data(), work.size(), info);
		check_info(info);
		U.applyOnTheLeft(A);
		Vt.applyOnTheRight(B);
	}


	// TODO size constraints!
	void add_identity (double lambda = 1.0) {
		const int N = S.size();
		int info = 0;
		Matrix A = U;
		Matrix B = Vt;
		other = U.transpose() * Vt.transpose();
		other.diagonal() += S * lambda;
		reserve(6*N);
		dgesvd_("A", "A", N, N, other.data(), N, S.data(), U.data(), N, Vt.data(), N, work.data(), work.size(), info);
		check_info(info);
		U.applyOnTheLeft(A);
		Vt.applyOnTheRight(B);
	}

	// TODO size constraints!
	void add_svd (const SVDHelper &s) {
		const int N = S.size();
		int info = 0;
		Matrix A = U;
		Matrix B = s.Vt;
		other = (U.transpose()*s.U) * s.S.asDiagonal();
		other += S.asDiagonal() * (Vt*s.Vt.transpose());
		dgesvd_("A", "A", N, N, other.data(), N, S.data(), U.data(), N, Vt.data(), N, work.data(), work.size(), info);
		check_info(info);
		U.applyOnTheLeft(A);
		Vt.applyOnTheRight(B);
	}

	Matrix matrix () {
		return U.block(0, 0, U.rows(), S.size()) * S.asDiagonal() * Vt.block(0, 0, S.size(), Vt.cols());
	}

	Matrix inverse () {
		return Vt.transpose() * S.array().inverse().matrix().asDiagonal() * U.transpose();
	}

	void invertInPlace () {
		S = S.array().inverse().matrix();
		S.reverseInPlace();
		other = U.transpose().colwise().reverse();
		U = Vt.transpose().rowwise().reverse();
		Vt = other;
		//other.setZero(S.size(), S.size());
		//for (int i=0;i<S.size();i++) {
			//other(i, S.size()-i-1) = 1.0;
		//}
		//U.applyOnTheRight(other);
		//Vt.applyOnTheLeft(other);
		//std::cerr << "Vt " << Vt << std::endl << std::endl;
	}

	void diagonalize () {
		const int N = S.size();
		Matrix A = Matrix::Zero(N, N);
		Matrix B = Matrix::Zero(N, N);
		A.diagonal() = S;
		B = (Vt*U).transpose();
		Vector alphar = Vector::Zero(N);
		Vector alphai = Vector::Zero(N);
		Vector beta = Vector::Zero(N);
		//Matrix VR = Matrix::Zero(N, N);
		int info;
		dggev_("N", "N", N, A.data(), N, B.data(), N, alphar.data(), alphai.data(), beta.data(), NULL, 1, NULL, 1, work.data(), work.size(), info);
		std::cerr << "sv: " << S.transpose() << std::endl;
		std::cerr << "dggev: " << info << std::endl;
		std::cerr << alphar.transpose() << std::endl;
		std::cerr << alphai.transpose() << std::endl;
		std::cerr << beta.transpose() << std::endl;
		int x = 0;
		for (int i=0;i<N;i++) if (alphai[i]==0.0 && alphar[i]/beta[i]<0.0) x++;
		std::cerr << x << " negatives" << std::endl;
		std::cerr << S.array().log().sum() << " vs. " << alphar.array().abs().log().sum()-beta.array().log().sum() << std::endl;
	}

	void printout () {
		std::cerr << "U=" << U << std::endl << std::endl;
		std::cerr << "S=" << S << std::endl << std::endl;
		std::cerr << "V=" << Vt << std::endl << std::endl;
		std::cerr << "B=" << matrix() << std::endl << std::endl;
	}
};

#endif // SVD_HPP

