#ifndef MPFR_HPP
#define MPFR_HPP

#include <iostream>
#include <cstdio>
#include <gmp.h>
#include <mpfr.h>

#include <Eigen/Core>

#include <algorithm>
#include <utility>

class PreciseMatrix {
	private:
	size_t rows_;
	size_t cols_;
	mpfr_t *data_;
	mpfr_prec_t prec_;
	mpfr_rnd_t rnd_;
	void cleanup ();
	public:
	PreciseMatrix (mpfr_prec_t precision = 64);
	~PreciseMatrix ();
	void resize (size_t newrows, size_t newcols);
	size_t size () const { return rows()*cols(); }
	size_t rows () const { return rows_; }
	size_t cols () const { return cols_; }
	mpfr_t* data () const { return data_; }
	mpfr_rnd_t rnd () const { return rnd_; }
	mpfr_prec_t precision () const { return prec_; }
	const mpfr_t& coeff (size_t row, size_t col) const { return data_[row+rows()*col]; }
	mpfr_t& coeff (size_t row, size_t col) { return data_[row+rows()*col]; }
	void set_coeff (size_t row, size_t col, double x) { mpfr_set_d(data_[row+rows()*col], x, rnd_); }
	const PreciseMatrix& operator= (const PreciseMatrix& B);
	const PreciseMatrix& operator= (const Eigen::MatrixXd& B);
	const PreciseMatrix& operator*= (double x);
	const PreciseMatrix& operator+= (const Eigen::MatrixXd& B);
	void applyOnTheLeft (const Eigen::MatrixXd& B);
	void applyOnTheLeft (const PreciseMatrix& B);
	void swap (PreciseMatrix& other);
	int inPlaceLU ();
	void balance ();
	void reduce_to_hessenberg ();
	void extract_hessenberg_H (PreciseMatrix& other);
	void extract_hessenberg_UV (PreciseMatrix& U, PreciseMatrix& V);
	void reduce_to_ev (PreciseMatrix& wr, PreciseMatrix& wi);
	void copy_into (Eigen::MatrixXd& other);
};

std::ostream& operator<< (std::ostream& out, const PreciseMatrix& A);
std::ostream& operator<< (std::ostream& out, const mpfr_t& x);



#endif // MPFR_HPP

