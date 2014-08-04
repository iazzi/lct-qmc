#ifndef ACCUMULATOR_HPP
#define ACCUMULATOR_HPP
#include "svd.hpp"

class Accumulator {
	using SVDMatrix = SVDHelper;
	SVDMatrix svd;
	double total_logdet;
	double current_logdet;
	double dist;

	class AssertionFailed {};

	typedef typename SVDMatrix::Matrix Matrix;

	public:
	void start (const Matrix& M) {
		svd.inPlaceSVD(M);
		total_logdet = 0.0;
		current_logdet = 0.0;
		dist = 0.0;
	}

	void reset (size_t V) {
		svd.setIdentity(V);
		total_logdet = 0.0;
		current_logdet = 0.0;
	}

	Matrix &matrixU () { return svd.U; }
	Matrix &matrixVt () { return svd.Vt; }

	double logdet () const { return svd.S.array().log().sum(); }
	double distance () const { return dist; }

	void increase_logdet (double x) {
		total_logdet += x;
		current_logdet += x;
	}

	void increase_distance (double d) {
		dist += d;
	}

	void decomposeU () {
		svd.absorbU();
		current_logdet = 0.0;
		dist = 0.0;
	}

	bool testLogDet (double prec = 1.0e-6) const {
		return std::fabs(svd.S.array().log().sum()-total_logdet)<prec;
	}

	void assertLogDet (double prec = 1.0e-6) const {
		if (!testLogDet(prec)) throw AssertionFailed();
	}

	const SVDMatrix& SVD () const { return svd;	}
	SVDMatrix& SVD () { return svd;	}
};

#endif // ACCUMULATOR_HPP

