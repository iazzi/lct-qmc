#include "mpfr.hpp"

#include <Eigen/LU>

using namespace std;

PreciseMatrix::PreciseMatrix (mpfr_prec_t precision) : rows_(0), cols_(0), data_(NULL), prec_(precision), rnd_(MPFR_RNDN) {}
PreciseMatrix::~PreciseMatrix () { cleanup(); }

void PreciseMatrix::cleanup () {
	const size_t V = size();
	for (size_t i=0;i<V;i++) {
		mpfr_clear(data_[i]);
	}
	if (data_!=NULL) delete[] data_;
	data_ = NULL;
}

void PreciseMatrix::resize (size_t newrows, size_t newcols) {
	if (size()!=newrows*newcols) {
		cleanup();
		const size_t V = newrows*newcols;
		data_ = new mpfr_t[V];
		for (size_t i=0;i<V;i++) {
			mpfr_init2(data_[i], prec_);
		}
	}
	rows_ = newrows;
	cols_ = newcols;
}

const PreciseMatrix& PreciseMatrix::operator= (const PreciseMatrix& B) {
	resize(B.rows(), B.cols());
	for (size_t i=0;i<rows();i++)
		for (size_t j=0;j<cols();j++) {
			mpfr_set(coeff(i, j), B.coeff(i, j), rnd_);
		}
	return *this;
}

const PreciseMatrix& PreciseMatrix::operator= (const Eigen::MatrixXd& B) {
	resize(B.rows(), B.cols());
	for (size_t i=0;i<rows();i++)
		for (size_t j=0;j<cols();j++) {
			mpfr_set_d(coeff(i, j), B(i, j), rnd_);
		}
	return *this;
}

const PreciseMatrix& PreciseMatrix::operator*= (double x) {
	for (size_t j=0;j<rows();j++) {
		for (size_t k=0;k<cols();k++) {
			mpfr_mul_d(coeff(j, k), coeff(j, k), x, rnd());
		}
	}
	return *this;
}

const PreciseMatrix& PreciseMatrix::operator+= (const Eigen::MatrixXd& B) {
	for (size_t j=0;j<rows();j++) {
		for (size_t k=0;k<cols();k++) {
			mpfr_add_d(coeff(j, k), coeff(j, k), B(j, k), rnd());
		}
	}
	return *this;
}

void PreciseMatrix::applyOnTheRight (const Eigen::MatrixXd& B) {
	PreciseMatrix C(prec_);
	C = Eigen::MatrixXd::Zero(rows(), B.cols());
	mpfr_t c;
	mpfr_init2(c, prec_);
	for (size_t i=0;i<rows();i++){
		for (size_t j=0;j<cols();j++) {
			for (size_t k=0;k<size_t(B.cols());k++) {
				// C(i, k) += A(i, j)*B(j, k)
				mpfr_mul_d(c, coeff(i, j), B(j, k), rnd());
				mpfr_add(C.coeff(i, k), C.coeff(i, k), c, rnd());
			}
		}
	}
	mpfr_clear(c);
	swap(C);
}

void PreciseMatrix::applyOnTheRight (const PreciseMatrix& B) {
	PreciseMatrix C(prec_);
	C = Eigen::MatrixXd::Zero(rows(), B.cols());
	mpfr_t c;
	mpfr_init2(c, prec_);
	for (size_t i=0;i<rows();i++){
		for (size_t j=0;j<cols();j++) {
			for (size_t k=0;k<B.cols();k++) {
				// C(i, k) += A(i, j)*B(j, k)
				mpfr_mul(c, B.coeff(j, k), coeff(i, j), rnd());
				mpfr_add(C.coeff(i, k), C.coeff(i, k), c, rnd());
			}
		}
	}
	mpfr_clear(c);
	swap(C);
}

void PreciseMatrix::applyOnTheLeft (const Eigen::MatrixXd& B) {
	PreciseMatrix C(prec_);
	C = Eigen::MatrixXd::Zero(B.rows(), cols());
	mpfr_t c;
	mpfr_init2(c, prec_);
	for (int i=0;i<B.rows();i++){
		for (size_t j=0;j<rows();j++) {
			for (size_t k=0;k<cols();k++) {
				// C(i, k) += B(i, j)*A(j, k)
				mpfr_mul_d(c, coeff(j, k), B(i, j), rnd());
				mpfr_add(C.coeff(i, k), C.coeff(i, k), c, rnd());
			}
		}
	}
	mpfr_clear(c);
	swap(C);
}

void PreciseMatrix::applyOnTheLeft (const PreciseMatrix& B) {
	PreciseMatrix C(prec_);
	C = Eigen::MatrixXd::Zero(B.rows(), cols());
	mpfr_t c;
	mpfr_init2(c, prec_);
	for (size_t i=0;i<B.rows();i++){
		for (size_t j=0;j<rows();j++) {
			for (size_t k=0;k<cols();k++) {
				// C(i, k) += B(i, j)*A(j, k)
				mpfr_mul(c, coeff(j, k), B.coeff(i, j), rnd());
				mpfr_add(C.coeff(i, k), C.coeff(i, k), c, rnd());
			}
		}
	}
	mpfr_clear(c);
	swap(C);
}

PreciseMatrix operator* (const Eigen::MatrixXd& A, const PreciseMatrix& B) {
	PreciseMatrix C(B.precision());
	C = Eigen::MatrixXd::Zero(A.rows(), B.cols());
	mpfr_t c;
	mpfr_init2(c, B.precision());
	for (size_t i=0;i<size_t(A.rows());i++){
		for (size_t j=0;j<B.rows();j++) {
			for (size_t k=0;k<B.cols();k++) {
				// C(i, k) += A(i, j)*B(j, k)
				mpfr_mul_d(c, B.coeff(j, k), A(i, j), B.rnd());
				mpfr_add(C.coeff(i, k), C.coeff(i, k), c, B.rnd());
			}
		}
	}
	mpfr_clear(c);
	return C;
}

void PreciseMatrix::get_norm (mpfr_t& n) const {
	mpfr_t tmp;
	mpfr_init2(tmp, prec_);
	mpfr_set_zero(n, +1);
	for (size_t i=0;i<rows_*cols_;i++) {
		mpfr_sqr(tmp, data_[i], rnd_);
		mpfr_add(n, n, tmp, rnd_);
	}
	mpfr_sqrt(n, n, rnd_);
	mpfr_clear(tmp);
}

void PreciseMatrix::normalize () {
	mpfr_t tmp;
	mpfr_init2(tmp, prec_);
	get_norm(tmp);
	for (size_t i=0;i<rows_*cols_;i++) {
		mpfr_div(data_[i], data_[i], tmp, rnd_);
	}
	mpfr_clear(tmp);
}

ostream& operator<< (ostream& out, const PreciseMatrix& A) {
	for (size_t i=0;i<A.rows();i++) {
		for (size_t j=0;j<A.cols();j++) {
			out << A.coeff(i, j) << (j==A.cols()-1?'\n':' ');
		}
	}
	return out;
}

void PreciseMatrix::extract_bands (PreciseMatrix& other, int n, int m) {
	if (m<n) std::swap(n, m);
	for (int i=0;i<std::min(int(rows_), int(cols_)-n);i++) {
		for (int j=std::max(0, i+n);j<std::min(int(cols_), i+m);j++) {
			mpfr_set(other.coeff(i, j), coeff(i, j), rnd_);
		}
	}
}

void PreciseMatrix::swap (PreciseMatrix& other) {
	std::swap(prec_, other.prec_);
	std::swap(rows_, other.rows_);
	std::swap(cols_, other.cols_);
	std::swap(rnd_, other.rnd_);
	std::swap(data_, other.data_);
}

int PreciseMatrix::permute_rows (const std::vector<int> &perm) {
	int ret = 1;
	for (int j=rows_-1;j>=0;j--) {
		if (j!=perm[j]) {
			ret = -ret;
			for (size_t k=0;k<cols_;k++) {
				mpfr_swap(coeff(j, k), coeff(perm[j], k));
			}
		}
	}
	return ret;
}

int PreciseMatrix::permute_rows_inv (const std::vector<int> &perm) {
	int ret = 1;
	for (int j=0;j<rows_;j++) {
		if (j!=perm[j]) {
			ret = -ret;
			for (size_t k=0;k<cols_;k++) {
				mpfr_swap(coeff(j, k), coeff(perm[j], k));
			}
		}
	}
	return ret;
}

void PreciseMatrix::apply_inverse_LU_vector (PreciseMatrix& v, const std::vector<int> &perm) {
	mpfr_t sum, temp;
	mpfr_inits2(prec_, sum, temp, (mpfr_ptr) 0);
	v.permute_rows_inv(perm);
	for (size_t i=0;i<rows_;i++) {
		mpfr_set_zero(sum, +1);
		for (size_t j=0;j<i;j++) {
			mpfr_mul(temp, coeff(i, j), v.coeff(j, 0), rnd_);
			mpfr_add(sum, sum, temp, rnd_);
		}
		mpfr_sub(v.coeff(i, 0), v.coeff(i, 0), sum, rnd_);
	}
	for (size_t i=rows_;i-->0;) {
		mpfr_set_zero(sum, +1);
		for (size_t j=rows_-1;j>i;j--) {
			mpfr_mul(temp, coeff(i, j), v.coeff(j, 0), rnd_);
			mpfr_add(sum, sum, temp, rnd_);
		}
		mpfr_sub(sum, v.coeff(i, 0), sum, rnd_);
		mpfr_div(v.coeff(i, 0), sum, coeff(i, i), rnd_);
	}
	mpfr_clears(sum, temp, (mpfr_ptr) 0);
}

int PreciseMatrix::in_place_LU (std::vector<int> &perm) {
	const int N = rows();
	int d = 1;
	int i,imax,j,k;
	mpfr_t big, dum, sum, temp;
	mpfr_inits2(prec_, big, dum, sum, temp, (mpfr_ptr) 0);
	PreciseMatrix vv; //   vv stores the implicit scaling of each row.
	vv = Eigen::VectorXd::Zero(N);
	perm.resize(N);
	for (i=1;i<=N;i++) { // Loop over rows to get the implicit scaling information.
		mpfr_set_zero(big, +1);
		for (j=1;j<=N;j++)
			if (mpfr_cmpabs(coeff(i-1, j-1), big)>0) mpfr_set(big, coeff(i-1, j-1), rnd_);
		//No nonzero largest element.
		if (mpfr_zero_p(big)) throw("Singular matrix in routine ludcmp");
		//Save the scaling.
		mpfr_d_div(vv.coeff(i-1, 0), 1.0, big, rnd_);
		mpfr_abs(vv.coeff(i-1, 0), vv.coeff(i-1, 0), rnd_);
	}
	for (j=1;j<=N;j++) { // This is the loop over columns of Crout’s method.
		for (i=1;i<j;i++) {
			//This is equation (2.3.12) except for i = j.
			mpfr_set(sum, coeff(i-1, j-1), rnd_);
			for (k=1;k<i;k++) {
				//sum -= a[i][k]*a[k][j];
				mpfr_mul(temp, coeff(i-1, k-1), coeff(k-1, j-1), rnd_);
				mpfr_sub(sum, sum, temp, rnd_);
			}
			mpfr_set(coeff(i-1, j-1), sum, rnd_);
		}
		//Initialize for the search for largest pivot element.
		mpfr_set_zero(big, +1);
		for (i=j;i<=N;i++) {
			//This is i = j of equation (2.3.12) and i = j + 1 . . . N
			mpfr_set(sum, coeff(i-1, j-1), rnd_);
			//of equation (2.3.13).
			for (k=1;k<j;k++) {
				//sum -= a[i][k]*a[k][j];
				mpfr_mul(temp, coeff(i-1, k-1), coeff(k-1, j-1), rnd_);
				mpfr_sub(sum, sum, temp, rnd_);
			}
			mpfr_set(coeff(i-1, j-1), sum, rnd_);
			mpfr_mul(dum, vv.coeff(i-1, 0), sum, rnd_);
			if (mpfr_cmpabs(dum, big)>=0) {
				//Is the figure of merit for the pivot better than the best so far?
				mpfr_abs(big, dum, rnd_);
				imax=i;
			}
		}
		//Do we need to interchange rows?
		if (j!=imax) {
			//Yes, do so...
			for (k=1;k<=N;k++) {
				mpfr_swap(coeff(imax-1, k-1), coeff(j-1, k-1));
			}
			//...and change the parity of d.
			d = -d;
			//Also interchange the scale factor.
			mpfr_set(vv.coeff(imax-1, 0), vv.coeff(j-1, 0), rnd_);
		}
		perm[j-1]=imax-1;
		//If the pivot element is zero the matrix is singular (at least to the precision of the
		//algorithm). For some applications on singular matrices, it is desirable to substitute
		//TINY for zero.
		// if (a[j][j] == 0.0) a[j][j]=TINY;
		if (j!=N) // Now, finally, divide by the pivot element.
		{
			mpfr_d_div(dum, 1.0, coeff(j-1, j-1), rnd_);
			for (i=j+1;i<=N;i++) mpfr_mul(coeff(i-1, j-1), coeff(i-1, j-1), dum, rnd_);
		}
	}
	//Go back for the next column in the reduction.
	mpfr_clears(big, dum, sum, temp, (mpfr_ptr) 0);
	//vv.transpose_in_place();
	//cout << vv << endl;
	return d;
}

void PreciseMatrix::balance () {
	const int n = rows();
	const double RADIX = 2.0;
	int last,j,i;
	mpfr_t s, r, g, f, c, sqrdx;
	mpfr_inits2(prec_, s, r, g, f, c, sqrdx, (mpfr_ptr) 0);
	mpfr_set_d(sqrdx, RADIX*RADIX, rnd_);
	last = 0;
	while (last==0) {
		last = 1;
		for (i=1;i<=n;i++) {
			//Calculate row and column norms.
			mpfr_set_zero(r, +1);
			mpfr_set_zero(c, +1);
			for (j=1;j<=n;j++) {
				if (j != i) {
					//c += fabs(a[j][i]);
					//r += fabs(a[i][j]);
					if (mpfr_sgn(coeff(j-1, i-1))<0) {
						mpfr_sub(c, c, coeff(j-1, i-1), rnd_);
					} else {
						mpfr_add(c, c, coeff(j-1, i-1), rnd_);
					}
					if (mpfr_sgn(coeff(i-1, j-1))<0) {
						mpfr_sub(r, r, coeff(i-1, j-1), rnd_);
					} else {
						mpfr_add(r, r, coeff(i-1, j-1), rnd_);
					}
				}
			}
			if (!(mpfr_zero_p(c) || mpfr_zero_p(r))) {
				//If both are nonzero,
				mpfr_div_d(g, r, RADIX, rnd_);
				mpfr_set_d(f, 1.0, rnd_);
				mpfr_add(s, c, r, rnd_);
				//find the integer power of the machine radix that
				//comes closest to balancing the matrix.
				while (mpfr_cmp(c, g)<0) {
					mpfr_mul_d(f, f, RADIX, rnd_);
					mpfr_mul(c, c, sqrdx, rnd_);
				}
				mpfr_mul_d(g, r, RADIX, rnd_);
				while (mpfr_cmp(c, g)>0) {
					mpfr_div_d(f, f, RADIX, rnd_);
					mpfr_div(c, c, sqrdx, rnd_);
				}
				//if ((c+r)/f < 0.95*s) {
				mpfr_add(c, c, r, rnd_);
				mpfr_mul(s, s, f, rnd_);
				mpfr_mul_d(s, s, 0.95, rnd_);
				if (mpfr_cmp(c, s)<0) {
					last=0;
					mpfr_d_div(g, 1.0, f, rnd_);
					//Apply similarity transformation
					for (j=1;j<=n;j++) mpfr_mul(coeff(i-1, j-1), coeff(i-1, j-1), g, rnd_);
					for (j=1;j<=n;j++) mpfr_mul(coeff(j-1, i-1), coeff(j-1, i-1), f, rnd_);
				}
			}
		}
	}
	mpfr_clears(s, r, g, f, c, sqrdx, (mpfr_ptr) 0);
}

void PreciseMatrix::reduce_to_hessenberg () {
	bool pivoting = false;
	const int n = rows();
	int m, j, i;
	mpfr_t y, x, z;
	mpfr_inits2(prec_, y, x, z, (mpfr_ptr) 0);
	for (m=2;m<n;m++) {
		// m is called r + 1 in the text.
		mpfr_set_zero(x, +1);
		i = m;
		for (j=m;j<=(pivoting?n:m);j++) {
			// Find the pivot.
			if (mpfr_cmpabs(coeff(j-1, m-2), x)>0) {
				mpfr_set(x, coeff(j-1, m-2), rnd_);
				i = j;
			}
		}
		if (i!=m) {
			// Interchange rows and columns.
			for (j=m-1;j<=n;j++) mpfr_swap(coeff(i-1, j-1), coeff(m-1, j-1));
			for (j=1;j<=n;j++) mpfr_swap(coeff(j-1, i-1), coeff(j-1, m-1));
		}
		if (!mpfr_zero_p(x)) {
			// Carry out the elimination.
			//cout << "carrying out the elimination on row " << m << endl;
			for (i=m+1;i<=n;i++) {
				if (!mpfr_zero_p(coeff(i-1, m-2))) {
					mpfr_div(y, coeff(i-1, m-2), x, rnd_);
					mpfr_set(coeff(i-1, m-2), y, rnd_);
					for (j=m;j<=n;j++) {
						//a[i][j] -= y*a[m][j];
						mpfr_mul(z, y, coeff(m-1, j-1), rnd_);
						mpfr_sub(coeff(i-1, j-1), coeff(i-1, j-1), z, rnd_);
					}
					for (j=1;j<=n;j++) {
						//a[j][m] += y*a[j][i];
						mpfr_mul(z, y, coeff(j-1, i-1), rnd_);
						mpfr_add(coeff(j-1, m-1), coeff(j-1, m-1), z, rnd_);
					}
				}
			}
		}
	}
	mpfr_clears(y, x, z, (mpfr_ptr) 0);
}

void PreciseMatrix::extract_hessenberg_H (PreciseMatrix& other) {
	other = Eigen::MatrixXd::Zero(rows_, cols_);
	for (size_t i=0;i<cols_;i++)
		for (size_t j=0;j<std::min(rows_, i+2);j++) {
		mpfr_set(other.coeff(j, i), coeff(j, i), rnd_);
		}
}

void PreciseMatrix::copy_into (Eigen::MatrixXd& other) {
	other.resize(rows_, cols_);
	for (size_t i=0;i<rows_;i++)
		for (size_t j=0;j<cols_;j++) {
			other(i, j) = mpfr_get_d(coeff(i, j), rnd_);
		}
}

void PreciseMatrix::copy_transpose (PreciseMatrix& other) {
	other.resize(cols_, rows_);
	for (size_t i=0;i<rows_;i++)
		for (size_t j=0;j<cols_;j++) {
			mpfr_set(other.coeff(j, i), coeff(i, j), rnd_);
		}
}

void PreciseMatrix::extract_hessenberg_UV (PreciseMatrix& U, PreciseMatrix& V) {
	//const size_t n = rows_;
	U = Eigen::MatrixXd::Identity(rows_, rows_);
	V = Eigen::MatrixXd::Identity(cols_, cols_);
	mpfr_t x, y, z;
	mpfr_inits2(prec_, x, y, z, (mpfr_ptr) 0);
	for (size_t i=0;i<cols_;i++) {
		for (size_t j=i+2;j<rows_;j++) {
			mpfr_set(y, coeff(j, i), rnd_);
			for (size_t k=0;k<rows_;k++) {
				mpfr_mul(z, y, U.coeff(k, j), rnd_);
				mpfr_add(U.coeff(k, i+1), U.coeff(k, i+1), z, rnd_);
			}
			for (size_t k=0;k<cols_;k++) {
				mpfr_mul(z, y, V.coeff(i+1, k), rnd_);
				mpfr_sub(V.coeff(j, k), V.coeff(j, k), z, rnd_);
			}
		}
	}
	mpfr_clears(x, y, z, (mpfr_ptr) 0);
}

void PreciseMatrix::transpose_in_place () {
	std::swap(rows_, cols_);
	for (size_t i=0;i<rows_;i++)
		for (size_t j=0;j<cols_;j++) {
			mpfr_swap(data_[i*cols_+j], data_[j*rows_+i]);
		}
}

std::ostream& operator<< (std::ostream& out, const mpfr_t& x) {
	mpfr_exp_t e;
	char *s = mpfr_get_str(NULL, &e, 10, 0, x, MPFR_RNDN);
	char *t = s;
	if (mpfr_sgn(x)<0) out << *(t++);
	if (e==0) {
		out << "0." << t;
	} else {
		out << t[0] << '.' << &t[1];
		if (e!=1) out << 'e' << e-1;
	}
	mpfr_free_str(s);
	return out;
}

void PreciseMatrix::reduce_to_ev (PreciseMatrix &wr, PreciseMatrix& wi) {
	//PreciseMatrix wr(prec_), wi(prec_);
	//PreciseMatrix Q(prec_);
	const int n = rows_;
	wr.resize(n, 1);
	wi.resize(n, 1);
	int nn, m, l, k, j, its, i, mmin;
	mpfr_t z, y, x, w, v, u, t, s, r, q, p, anorm, temp;
	mpfr_inits2(prec_, z, y, x, w, v, u, t, s, r, q, p, anorm, temp, (mpfr_ptr) 0);
	mpfr_set_zero(anorm, +1);;
	//Compute matrix norm for possible use in locating single small subdiagonal element.
	for (i=1;i<=n;i++)
		for (j=std::max(i-1,1);j<=n;j++)
			if (mpfr_sgn(coeff(i-1, j-1))<0) {
				mpfr_sub(anorm, anorm, coeff(i-1, j-1), rnd_);
			} else {
				mpfr_add(anorm, anorm, coeff(i-1, j-1), rnd_);
			}
	nn = n;
	mpfr_set_zero(t, +1);
	//Gets changed only by an exceptional shift.
	while (nn>=1) {
		//Begin search for next eigenvalue.
		its = 0;
		do {
			for (l=nn;l>=2;l--) {
				//Begin iteration: look for single small subdiagonal element.
				mpfr_abs(s, coeff(l-2, l-2), rnd_);
				if (mpfr_sgn(coeff(l-1, l-1))<0) {
					mpfr_sub(s, s, coeff(l-1, l-1), rnd_);
				} else {
					mpfr_add(s, s, coeff(l-1, l-1), rnd_);
				}
				//s=fabs(a[l-1][l-1])+fabs(a[l][l]);
				if (mpfr_zero_p(s)) mpfr_set(s, anorm, rnd_);
				mpfr_add(temp, s, coeff(l-1, l-2), rnd_);
				mpfr_abs(temp, temp, rnd_);
				if (mpfr_cmp(temp, s)==0) {
					//std::cerr << coeff(l-1, l-2) << " found to be much less than " << s << std::endl;
					mpfr_set_zero(coeff(l-1, l-2), +1);
					break;
				}
			}
			mpfr_set(x, coeff(nn-1, nn-1), rnd_);
			if (l==nn) {
				//One root found.
				mpfr_add(wr.coeff(nn-1, 0), x, t, rnd_);
				mpfr_set_zero(wi.coeff(nn-1, 0), +1);
				//std::cerr << "one root found at " << l << ": " << wr.coeff(nn-1, 0) << std::endl;
				nn--;
			} else {
				mpfr_set(y, coeff(nn-2, nn-2), rnd_);
				mpfr_mul(w, coeff(nn-1, nn-2), coeff(nn-2, nn-1), rnd_);
				if (l==(nn-1)) {
					// Two roots found...
					//p = 0.5*(y-x);
					mpfr_sub(p, y, x, rnd_);
					mpfr_mul_d(p, p, 0.5, rnd_);
					//q=p*p+w;
					mpfr_mul(q, p, p, rnd_);
					mpfr_add(q, q, w, rnd_);
					//z=sqrt(fabs(q));
					mpfr_abs(z, q, rnd_);
					mpfr_sqrt(z, z, rnd_);
					//x += t;
					mpfr_add(x, x, t, rnd_);
					if (mpfr_sgn(q)>=0) {
						//std::cerr << "found a real pair" << std::endl;
						//...a real pair.
						// z=p+SIGN(z,p); // #define SIGN(a,b) (b>=0?+fabs(a):-fabs(a))
						mpfr_abs(z, z, rnd_);
						if (mpfr_sgn(p)>=0) {
							mpfr_add(z, p, z, rnd_);
						} else {
							mpfr_sub(z, p, z, rnd_);
						}
						//wr[nn-1]=wr[nn]=x+z;
						mpfr_add(wr.coeff(nn-2, 0), x, z, rnd_);
						//if (z) wr[nn]=x-w/z;
						if (mpfr_zero_p(z)) {
							mpfr_add(wr.coeff(nn-1, 0), x, z, rnd_);
						} else {
							mpfr_div(wr.coeff(nn-1, 0), w, z, rnd_);
							mpfr_sub(wr.coeff(nn-1, 0), x, wr.coeff(nn-1, 0), rnd_);
						}
						//wi[nn-1]=wi[nn]=0.0;
						mpfr_set_zero(wi.coeff(nn-2, 0), +1);
						mpfr_set_zero(wi.coeff(nn-1, 0), +1);
					} else {
						//std::cerr << "found a complex pair" << std::endl;
						//...a complex pair.
						//wr[nn-1]=wr[nn]=x+p;
						mpfr_add(wr.coeff(nn-1, 0), x, p, rnd_);
						mpfr_set(wr.coeff(nn-2, 0), wr.coeff(nn-1, 0), rnd_);
						//wi[nn-1]= -(wi[nn]=z);
						mpfr_set(wi.coeff(nn-1, 0), z, rnd_);
						mpfr_mul_d(wi.coeff(nn-2, 0), z, -1.0, rnd_);
					}
					nn -= 2;
				} else {
					//No roots found. Continue iteration.
					if (its == 30) throw("Too many iterations in hqr");
					if (its == 10 || its == 20) {
						// Form exceptional shift.
						mpfr_add(t, t, x, rnd_);
						for (i=1;i<=nn;i++) mpfr_sub(coeff(i-1, i-1), coeff(i-1, i-1), x, rnd_);
						//s=fabs(a[nn][nn-1])+fabs(a[nn-1][nn-2]);
						mpfr_abs(s, coeff(nn-1, nn-2), rnd_);
						if (mpfr_sgn(coeff(nn-2, nn-1))>=0) {
							mpfr_add(s, s, coeff(nn-2, nn-1), rnd_);
						} else {
							mpfr_sub(s, s, coeff(nn-2, nn-1), rnd_);
						}
						//y=x=0.75*s;
						mpfr_mul_d(x, s, 0.75, rnd_);
						mpfr_set(y, x, rnd_);
						//w = -0.4375*s*s;
						mpfr_sqr(w, s, rnd_);
						mpfr_mul_d(w, w, -0.4375, rnd_);
					}
					++its;
					for (m=(nn-2);m>=l;m--) {
						//Form shift and then look for 2 consecutive small subdiagonal elements.
						mpfr_set(z, coeff(m-1, m-1), rnd_); // z=a[m][m];
						mpfr_sub(r, x, z, rnd_); // r=x-z;
						mpfr_sub(s, y, z, rnd_); // s=y-z;
						//Equation (11.6.23).
						//p=(r*s-w)/a[m+1][m]+a[m][m+1];
						mpfr_mul(p, r, s, rnd_);
						mpfr_sub(p, p, w, rnd_);
						mpfr_div(p, p, coeff(m, m-1), rnd_);
						mpfr_add(p, p, coeff(m-1, m), rnd_);
						// q=a[m+1][m+1]-z-r-s;
						mpfr_sub(q, coeff(m, m), z, rnd_);
						mpfr_sub(q, q, r, rnd_);
						mpfr_sub(q, q, s, rnd_);
						//r=a[m+2][m+1];
						mpfr_set(r, coeff(m+1, m), rnd_);
						//s=fabs(p)+fabs(q)+fabs(r);
						mpfr_abs(s, p, rnd_);
						if (mpfr_sgn(q)>=0) mpfr_add(s, s, q, rnd_); else mpfr_sub(s, s, q, rnd_);
						if (mpfr_sgn(r)>=0) mpfr_add(s, s, r, rnd_); else mpfr_sub(s, s, r, rnd_);
						//Scale to prevent overflow or underflow.
						mpfr_div(p, p, s, rnd_); // p /= s;
						mpfr_div(q, q, s, rnd_); // q /= s;
						mpfr_div(r, r, s, rnd_); // r /= s;
						if (m == l) break;
						//u=fabs(a[m][m-1])*(fabs(q)+fabs(r));
						mpfr_abs(u, q, rnd_);
						if (mpfr_sgn(r)>=0) mpfr_add(u, u, r, rnd_); else mpfr_sub(u, u, r, rnd_);
						mpfr_mul(u, u, coeff(m-1, m-2), rnd_);
						mpfr_abs(u, u, rnd_);
						//v=fabs(p)*(fabs(a[m-1][m-1])+fabs(z)+fabs(a[m+1][m+1]));
						mpfr_abs(v, coeff(m-2, m-2), rnd_);
						mpfr_abs(temp, z, rnd_);
						mpfr_add(v, v, temp, rnd_);
						mpfr_abs(temp, coeff(m, m), rnd_);
						mpfr_add(v, v, temp, rnd_);
						mpfr_abs(temp, p, rnd_);
						mpfr_mul(v, v, temp, rnd_);
						mpfr_add(temp, u, v, rnd_);
						if (mpfr_cmp(u, v)==0) break;
						//Equation (11.6.26).
					}
					for (i=m+2;i<=nn;i++) {
						mpfr_set_zero(coeff(i-1, i-3), +1);
						if (i!=(m+2)) mpfr_set_zero(coeff(i-1, i-4), +1);
					}
					for (k=m;k<=nn-1;k++) {
						//Double QR step on rows l to nn and columns m to nn.
						if (k != m) {
							mpfr_set(p, coeff(k-1, k-2), rnd_); // p=a[k][k-1];
							//Begin setup of Householder
							mpfr_set(q, coeff(k, k-2), rnd_); // q=a[k+1][k-1];
							//vector.
							mpfr_set_zero(r, +1); // r=0.0;
							if (k != (nn-1)) mpfr_set(r, coeff(k+1, k-2), rnd_); // r=a[k+2][k-1];
							mpfr_abs(x, p, rnd_);
							mpfr_abs(temp, q, rnd_);
							mpfr_add(x, x, temp, rnd_);
							mpfr_abs(temp, r, rnd_);
							mpfr_add(x, x, temp, rnd_);
							if (!mpfr_zero_p(x)) {
								//Scale to prevent overflow or underflow.
								mpfr_div(p, p, x, rnd_); // p /= x;
								mpfr_div(q, q, x, rnd_); // q /= x;
								mpfr_div(r, r, x, rnd_); // r /= x;
							}
						}
						//if ((s=SIGN(sqrt(p*p+q*q+r*r),p)) != 0.0) {
						mpfr_sqr(s, p, rnd_);
						mpfr_sqr(temp, q, rnd_);
						mpfr_add(s, s, temp, rnd_);
						mpfr_sqr(temp, r, rnd_);
						mpfr_add(s, s, temp, rnd_);
						mpfr_sqrt(s, s, rnd_);
						if (mpfr_sgn(p)<0) mpfr_mul_d(s, s, -1.0, rnd_);
						if (!mpfr_zero_p(s)) {
							if (k == m) {
								if (l != m)
									mpfr_mul_d(coeff(k-1, k-2), coeff(k-1, k-2), -1.0, rnd_); // a[k][k-1] = -a[k][k-1];
							} else {
								mpfr_mul(coeff(k-1, k-2), s, x, rnd_); // a[k][k-1] = -s*x;
								mpfr_mul_d(coeff(k-1, k-2), coeff(k-1, k-2), -1.0, rnd_);
							}
							mpfr_add(p, p, s, rnd_); // p += s;
							//Equations (11.6.24).
							mpfr_div(x, p, s, rnd_); // x=p/s;
							mpfr_div(y, q, s, rnd_); // y=q/s;
							mpfr_div(z, r, s, rnd_); // z=r/s;
							mpfr_div(q, q, p, rnd_); // q /= p;
							mpfr_div(r, r, p, rnd_); // r /= p;
							for (j=k;j<=nn;j++) {
								//std::cerr << "row mod" << std::endl;
								//Row modification.
								//p=a[k][j]+q*a[k+1][j];
								mpfr_mul(p, q, coeff(k, j-1), rnd_);
								mpfr_add(p, p, coeff(k-1, j-1), rnd_);
								if (k != (nn-1)) {
									//p += r*a[k+2][j];
									mpfr_mul(temp, r, coeff(k+1, j-1), rnd_);
									mpfr_add(p, p, temp, rnd_);
									//a[k+2][j] -= p*z;
									mpfr_mul(temp, p, z, rnd_);
									mpfr_sub(coeff(k+1, j-1), coeff(k+1, j-1), temp, rnd_);
								}
								// a[k+1][j] -= p*y;
								mpfr_mul(temp, p, y, rnd_);
								mpfr_sub(coeff(k, j-1), coeff(k, j-1), temp, rnd_);
								//a[k][j] -= p*x;
								mpfr_mul(temp, p, x, rnd_);
								mpfr_sub(coeff(k-1, j-1), coeff(k-1, j-1), temp, rnd_);
							}
							mmin = nn<k+3 ? nn : k+3;
							//std::cerr << k << ' ' << nn << ' ' << x << ' ' << y << ' ' << z << ' ' << q << ' ' << r << endl;
							for (i=l;i<=mmin;i++) {
								//std::cerr << "col mod" << std::endl;
								// Column modification.
								// p=x*a[i][k]+y*a[i][k+1];
								mpfr_mul(p, x, coeff(i-1, k-1), rnd_);
								mpfr_mul(temp, y, coeff(i-1, k), rnd_);
								mpfr_add(p, p, temp, rnd_);
								if (k != (nn-1)) {
									//p += z*a[i][k+2];
									mpfr_mul(temp, z, coeff(i-1, k+1), rnd_);
									mpfr_add(p, p, temp, rnd_);
									//a[i][k+2] -= p*r;
									mpfr_mul(temp, p, r, rnd_);
									mpfr_sub(coeff(i-1, k+1), coeff(i-1, k+1), temp, rnd_);
								}
								//a[i][k+1] -= p*q;
								mpfr_mul(temp, p, q, rnd_);
								mpfr_sub(coeff(i-1, k), coeff(i-1, k), temp, rnd_);
								//a[i][k] -= p;
								mpfr_sub(coeff(i-1, k-1), coeff(i-1, k-1), p, rnd_);
							}
							//std::cerr << "iteration with k=" << k << "\n" << *this << std::endl;
						}
					}
				}
			}
		} while (l < nn-1);
	}
	mpfr_clears(z, y, x, w, v, u, t, s, r, q, p, anorm, temp, (mpfr_ptr) 0);
}

void PreciseMatrix::reduce_to_ev_verbose (PreciseMatrix &wr, PreciseMatrix& wi, PreciseMatrix& Q) {
	//PreciseMatrix wr(prec_), wi(prec_);
	//PreciseMatrix Q(prec_);
	const int n = rows_;
	Q = Eigen::MatrixXd::Identity(n, n);
	wr.resize(n, 1);
	wi.resize(n, 1);
	int nn, m, l, k, j, its, i, mmin;
	mpfr_t z, y, x, w, v, u, t, s, r, q, p, anorm, temp;
	mpfr_inits2(prec_, z, y, x, w, v, u, t, s, r, q, p, anorm, temp, (mpfr_ptr) 0);
	mpfr_set_zero(anorm, +1);;
	//Compute matrix norm for possible use in locating single small subdiagonal element.
	for (i=1;i<=n;i++)
		for (j=std::max(i-1,1);j<=n;j++)
			if (mpfr_sgn(coeff(i-1, j-1))<0) {
				mpfr_sub(anorm, anorm, coeff(i-1, j-1), rnd_);
			} else {
				mpfr_add(anorm, anorm, coeff(i-1, j-1), rnd_);
			}
	nn = n;
	mpfr_set_zero(t, +1);
	//Gets changed only by an exceptional shift.
	while (nn>=1) {
		std::cerr << "nn=" << nn << std::endl;
		//Begin search for next eigenvalue.
		its = 0;
		do {
			for (l=nn;l>=2;l--) {
				//Begin iteration: look for single small subdiagonal element.
				mpfr_abs(s, coeff(l-2, l-2), rnd_);
				if (mpfr_sgn(coeff(l-1, l-1))<0) {
					mpfr_sub(s, s, coeff(l-1, l-1), rnd_);
				} else {
					mpfr_add(s, s, coeff(l-1, l-1), rnd_);
				}
				//s=fabs(a[l-1][l-1])+fabs(a[l][l]);
				if (mpfr_zero_p(s)) mpfr_set(s, anorm, rnd_);
				mpfr_add(temp, s, coeff(l-1, l-2), rnd_);
				mpfr_abs(temp, temp, rnd_);
				if (mpfr_cmp(temp, s)==0) {
					std::cerr << "l=" << l << "; " << coeff(l-1, l-2) << " found to be much less than " << s << std::endl;
					mpfr_set_zero(coeff(l-1, l-2), +1);
					break;
				}
			}
			mpfr_set(x, coeff(nn-1, nn-1), rnd_);
			if (l==nn) {
				//One root found.
				mpfr_add(wr.coeff(nn-1, 0), x, t, rnd_);
				mpfr_set_zero(wi.coeff(nn-1, 0), +1);
				std::cerr << "one root found at " << l << ": " << wr.coeff(nn-1, 0) << std::endl;
				nn--;
			} else {
				mpfr_set(y, coeff(nn-2, nn-2), rnd_);
				mpfr_mul(w, coeff(nn-1, nn-2), coeff(nn-2, nn-1), rnd_);
				if (l==(nn-1)) {
					// Two roots found...
					//p = 0.5*(y-x);
					mpfr_sub(p, y, x, rnd_);
					mpfr_mul_d(p, p, 0.5, rnd_);
					//q=p*p+w;
					mpfr_mul(q, p, p, rnd_);
					mpfr_add(q, q, w, rnd_);
					//z=sqrt(fabs(q));
					mpfr_abs(z, q, rnd_);
					mpfr_sqrt(z, z, rnd_);
					//x += t;
					mpfr_add(x, x, t, rnd_);
					if (mpfr_sgn(q)>=0) {
						//std::cerr << "found a real pair" << std::endl;
						//...a real pair.
						// z=p+SIGN(z,p); // #define SIGN(a,b) (b>=0?+fabs(a):-fabs(a))
						mpfr_abs(z, z, rnd_);
						if (mpfr_sgn(p)>=0) {
							mpfr_add(z, p, z, rnd_);
						} else {
							mpfr_sub(z, p, z, rnd_);
						}
						//wr[nn-1]=wr[nn]=x+z;
						mpfr_add(wr.coeff(nn-2, 0), x, z, rnd_);
						//if (z) wr[nn]=x-w/z;
						if (mpfr_zero_p(z)) {
							mpfr_add(wr.coeff(nn-1, 0), x, z, rnd_);
						} else {
							mpfr_div(wr.coeff(nn-1, 0), w, z, rnd_);
							mpfr_sub(wr.coeff(nn-1, 0), x, wr.coeff(nn-1, 0), rnd_);
						}
						//wi[nn-1]=wi[nn]=0.0;
						mpfr_set_zero(wi.coeff(nn-2, 0), +1);
						mpfr_set_zero(wi.coeff(nn-1, 0), +1);
					} else {
						//std::cerr << "found a complex pair" << std::endl;
						//...a complex pair.
						//wr[nn-1]=wr[nn]=x+p;
						mpfr_add(wr.coeff(nn-1, 0), x, p, rnd_);
						mpfr_set(wr.coeff(nn-2, 0), wr.coeff(nn-1, 0), rnd_);
						//wi[nn-1]= -(wi[nn]=z);
						mpfr_set(wi.coeff(nn-1, 0), z, rnd_);
						mpfr_mul_d(wi.coeff(nn-2, 0), z, -1.0, rnd_);
					}
					nn -= 2;
				} else {
					std::cerr << "starting double QR iteration" << std::endl;
					//No roots found. Continue iteration.
					if (its == 30) throw("Too many iterations in hqr");
					if (its == 10 || its == 20) {
						// Form exceptional shift.
						mpfr_add(t, t, x, rnd_);
						for (i=1;i<=nn;i++) mpfr_sub(coeff(i-1, i-1), coeff(i-1, i-1), x, rnd_);
						//s=fabs(a[nn][nn-1])+fabs(a[nn-1][nn-2]);
						mpfr_abs(s, coeff(nn-1, nn-2), rnd_);
						if (mpfr_sgn(coeff(nn-2, nn-1))>=0) {
							mpfr_add(s, s, coeff(nn-2, nn-1), rnd_);
						} else {
							mpfr_sub(s, s, coeff(nn-2, nn-1), rnd_);
						}
						//y=x=0.75*s;
						mpfr_mul_d(x, s, 0.75, rnd_);
						mpfr_set(y, x, rnd_);
						//w = -0.4375*s*s;
						mpfr_sqr(w, s, rnd_);
						mpfr_mul_d(w, w, -0.4375, rnd_);
					}
					++its;
					for (m=(nn-2);m>=l;m--) {
						//Form shift and then look for 2 consecutive small subdiagonal elements.
						mpfr_set(z, coeff(m-1, m-1), rnd_); // z=a[m][m];
						mpfr_sub(r, x, z, rnd_); // r=x-z;
						mpfr_sub(s, y, z, rnd_); // s=y-z;
						//Equation (11.6.23).
						//p=(r*s-w)/a[m+1][m]+a[m][m+1];
						mpfr_mul(p, r, s, rnd_);
						mpfr_sub(p, p, w, rnd_);
						mpfr_div(p, p, coeff(m, m-1), rnd_);
						mpfr_add(p, p, coeff(m-1, m), rnd_);
						// q=a[m+1][m+1]-z-r-s;
						mpfr_sub(q, coeff(m, m), z, rnd_);
						mpfr_sub(q, q, r, rnd_);
						mpfr_sub(q, q, s, rnd_);
						//r=a[m+2][m+1];
						mpfr_set(r, coeff(m+1, m), rnd_);
						//s=fabs(p)+fabs(q)+fabs(r);
						mpfr_abs(s, p, rnd_);
						if (mpfr_sgn(q)>=0) mpfr_add(s, s, q, rnd_); else mpfr_sub(s, s, q, rnd_);
						if (mpfr_sgn(r)>=0) mpfr_add(s, s, r, rnd_); else mpfr_sub(s, s, r, rnd_);
						//Scale to prevent overflow or underflow.
						mpfr_div(p, p, s, rnd_); // p /= s;
						mpfr_div(q, q, s, rnd_); // q /= s;
						mpfr_div(r, r, s, rnd_); // r /= s;
						if (m == l) break;
						//u=fabs(a[m][m-1])*(fabs(q)+fabs(r));
						mpfr_abs(u, q, rnd_);
						if (mpfr_sgn(r)>=0) mpfr_add(u, u, r, rnd_); else mpfr_sub(u, u, r, rnd_);
						mpfr_mul(u, u, coeff(m-1, m-2), rnd_);
						mpfr_abs(u, u, rnd_);
						//v=fabs(p)*(fabs(a[m-1][m-1])+fabs(z)+fabs(a[m+1][m+1]));
						mpfr_abs(v, coeff(m-2, m-2), rnd_);
						mpfr_abs(temp, z, rnd_);
						mpfr_add(v, v, temp, rnd_);
						mpfr_abs(temp, coeff(m, m), rnd_);
						mpfr_add(v, v, temp, rnd_);
						mpfr_abs(temp, p, rnd_);
						mpfr_mul(v, v, temp, rnd_);
						mpfr_add(temp, u, v, rnd_);
						if (mpfr_cmp(u, v)==0) break;
						//Equation (11.6.26).
					}
					for (i=m+2;i<=nn;i++) {
						mpfr_set_zero(coeff(i-1, i-3), +1);
						if (i!=(m+2)) mpfr_set_zero(coeff(i-1, i-4), +1);
					}
					for (k=m;k<=nn-1;k++) {
						//Double QR step on rows l to nn and columns m to nn.
						if (k != m) {
							mpfr_set(p, coeff(k-1, k-2), rnd_); // p=a[k][k-1];
							//Begin setup of Householder
							mpfr_set(q, coeff(k, k-2), rnd_); // q=a[k+1][k-1];
							//vector.
							mpfr_set_zero(r, +1); // r=0.0;
							if (k != (nn-1)) mpfr_set(r, coeff(k+1, k-2), rnd_); // r=a[k+2][k-1];
							mpfr_abs(x, p, rnd_);
							mpfr_abs(temp, q, rnd_);
							mpfr_add(x, x, temp, rnd_);
							mpfr_abs(temp, r, rnd_);
							mpfr_add(x, x, temp, rnd_);
							if (!mpfr_zero_p(x)) {
								//Scale to prevent overflow or underflow.
								mpfr_div(p, p, x, rnd_); // p /= x;
								mpfr_div(q, q, x, rnd_); // q /= x;
								mpfr_div(r, r, x, rnd_); // r /= x;
							}
						}
						//if ((s=SIGN(sqrt(p*p+q*q+r*r),p)) != 0.0) {
						mpfr_sqr(s, p, rnd_);
						mpfr_sqr(temp, q, rnd_);
						mpfr_add(s, s, temp, rnd_);
						mpfr_sqr(temp, r, rnd_);
						mpfr_add(s, s, temp, rnd_);
						mpfr_sqrt(s, s, rnd_);
						if (mpfr_sgn(p)<0) mpfr_mul_d(s, s, -1.0, rnd_);
						if (!mpfr_zero_p(s)) {
							if (k == m) {
								if (l != m)
									mpfr_mul_d(coeff(k-1, k-2), coeff(k-1, k-2), -1.0, rnd_); // a[k][k-1] = -a[k][k-1];
							} else {
								mpfr_mul(coeff(k-1, k-2), s, x, rnd_); // a[k][k-1] = -s*x;
								mpfr_mul_d(coeff(k-1, k-2), coeff(k-1, k-2), -1.0, rnd_);
							}
							mpfr_add(p, p, s, rnd_); // p += s;
							//Equations (11.6.24).
							mpfr_div(x, p, s, rnd_); // x=p/s;
							mpfr_div(y, q, s, rnd_); // y=q/s;
							mpfr_div(z, r, s, rnd_); // z=r/s;
							mpfr_div(q, q, p, rnd_); // q /= p;
							mpfr_div(r, r, p, rnd_); // r /= p;
							for (j=k;j<=nn;j++) {
								//std::cerr << "row mod" << std::endl;
								//Row modification.
								//p=a[k][j]+q*a[k+1][j];
								mpfr_mul(p, q, coeff(k, j-1), rnd_);
								mpfr_add(p, p, coeff(k-1, j-1), rnd_);
								if (k != (nn-1)) {
									//p += r*a[k+2][j];
									mpfr_mul(temp, r, coeff(k+1, j-1), rnd_);
									mpfr_add(p, p, temp, rnd_);
									//a[k+2][j] -= p*z;
									mpfr_mul(temp, p, z, rnd_);
									mpfr_sub(coeff(k+1, j-1), coeff(k+1, j-1), temp, rnd_);
								}
								// a[k+1][j] -= p*y;
								mpfr_mul(temp, p, y, rnd_);
								mpfr_sub(coeff(k, j-1), coeff(k, j-1), temp, rnd_);
								//a[k][j] -= p*x;
								mpfr_mul(temp, p, x, rnd_);
								mpfr_sub(coeff(k-1, j-1), coeff(k-1, j-1), temp, rnd_);
							}
							for (j=k;j<=nn;j++) {
								//std::cerr << "row mod" << std::endl;
								//Row modification.
								//p=a[k][j]+q*a[k+1][j];
								mpfr_mul(p, q, Q.coeff(k, j-1), rnd_);
								mpfr_add(p, p, Q.coeff(k-1, j-1), rnd_);
								if (k != (n-1)) {
									//p += r*a[k+2][j];
									mpfr_mul(temp, r, Q.coeff(k+1, j-1), rnd_);
									mpfr_add(p, p, temp, rnd_);
									//a[k+2][j] -= p*z;
									mpfr_mul(temp, p, z, rnd_);
									mpfr_sub(Q.coeff(k+1, j-1), Q.coeff(k+1, j-1), temp, rnd_);
								}
								// a[k+1][j] -= p*y;
								mpfr_mul(temp, p, y, rnd_);
								//std::cerr << k << ' ' << j-1 << ' ' << Q.coeff(k, j-1) << ' ' << temp;
								mpfr_sub(Q.coeff(k, j-1), Q.coeff(k, j-1), temp, rnd_);
								//std::cerr << " -> "<< Q.coeff(k, j-1) << std::endl;
								//a[k][j] -= p*x;
								mpfr_mul(temp, p, x, rnd_);
								mpfr_sub(Q.coeff(k-1, j-1), Q.coeff(k-1, j-1), temp, rnd_);
							}
							mmin = nn<k+3 ? nn : k+3;
							//std::cerr << k << ' ' << nn << ' ' << x << ' ' << y << ' ' << z << ' ' << q << ' ' << r << endl;
							for (i=l;i<=mmin;i++) {
								//std::cerr << "col mod" << std::endl;
								// Column modification.
								// p=x*a[i][k]+y*a[i][k+1];
								mpfr_mul(p, x, coeff(i-1, k-1), rnd_);
								mpfr_mul(temp, y, coeff(i-1, k), rnd_);
								mpfr_add(p, p, temp, rnd_);
								if (k != (nn-1)) {
									//p += z*a[i][k+2];
									mpfr_mul(temp, z, coeff(i-1, k+1), rnd_);
									mpfr_add(p, p, temp, rnd_);
									//a[i][k+2] -= p*r;
									mpfr_mul(temp, p, r, rnd_);
									mpfr_sub(coeff(i-1, k+1), coeff(i-1, k+1), temp, rnd_);
								}
								//a[i][k+1] -= p*q;
								mpfr_mul(temp, p, q, rnd_);
								mpfr_sub(coeff(i-1, k), coeff(i-1, k), temp, rnd_);
								//a[i][k] -= p;
								mpfr_sub(coeff(i-1, k-1), coeff(i-1, k-1), p, rnd_);
							}
							std::cerr << "iteration with k=" << k << "\n" << *this << std::endl;
						}
					}
					}
				}
			} while (l < nn-1);
		}
		mpfr_clears(z, y, x, w, v, u, t, s, r, q, p, anorm, temp, (mpfr_ptr) 0);
	}

void PreciseMatrix::extract_right_eigenvectors_real (PreciseMatrix& V) {
	const size_t n = rows_;
	mpfr_t sum1, sum2, d, e, tmp;
	mpfr_inits2(prec_, sum1, sum2, d, e, tmp, (mpfr_ptr) 0);
	V = Eigen::MatrixXd::Identity(n, n);
	for (size_t i=1;i<n;i++) {
		for (int j=i-1;j>=0;j--) {
			if (j==0 || mpfr_zero_p(coeff(j, j-1))) {
				mpfr_set_zero(sum1, +1);
				for (int k=i;k>j;k--) {
					mpfr_mul(tmp, V.coeff(k, i), coeff(j, k), rnd_);
					mpfr_add(sum1, sum1, tmp, rnd_);
				}
				mpfr_sub(tmp, coeff(i, i), coeff(j, j), rnd_);
				mpfr_div(V.coeff(j, i), sum1, tmp, rnd_);
			} else {
				mpfr_sub(d, coeff(j, j), coeff(i, i), rnd_);
				mpfr_sub(tmp, coeff(j-1, j-1), coeff(i, i), rnd_);
				mpfr_mul(d, d, tmp, rnd_);
				mpfr_mul(tmp, coeff(j, j-1), coeff(j-1, j), rnd_);
				mpfr_sub(d, tmp, d, rnd_);
				mpfr_set_zero(sum1, +1);
				mpfr_set_zero(sum2, +1);
				for (int k=i;k>j;k--) {
					mpfr_mul(tmp, V.coeff(k, i), coeff(j, k), rnd_);
					mpfr_add(sum1, sum1, tmp, rnd_);
					mpfr_mul(tmp, V.coeff(k, i), coeff(j-1, k), rnd_);
					mpfr_add(sum2, sum2, tmp, rnd_);
				}
				mpfr_sub(e, coeff(j-1, j-1), coeff(i, i), rnd_);
				mpfr_mul(e, e, sum1, rnd_);
				mpfr_mul(tmp, coeff(j, j-1), sum2, rnd_);
				mpfr_sub(e, e, tmp, rnd_);
				mpfr_div(V.coeff(j, i), e, d, rnd_);
				mpfr_sub(e, coeff(j, j), coeff(i, i), rnd_);
				mpfr_mul(e, e, sum2, rnd_);
				mpfr_mul(tmp, coeff(j-1, j), sum1, rnd_);
				mpfr_sub(e, e, tmp, rnd_);
				mpfr_div(V.coeff(j-1, i), e, d, rnd_);
				j--;
			}
		}
	}
	mpfr_clears(sum1, sum2, tmp, d, e, (mpfr_ptr) 0);
}

//void PreciseMatrix::extract_right_eigenvectors_complex (PreciseMatrix& VR, PreciseMatrix& VI) {
	//const int n = rows_;
	//mpfr_t sum, temp;
	//mpfr_inits2(prec_, sum, temp, (mpfr_ptr) 0);
	//VR = Eigen::MatrixXd::Identity(n, n);
	//VI = Eigen::MatrixXd::Zero(n, n);
	//for (size_t i=1;i<n;i++) {
		//for (int j=i-1;j>=0;j--) {
			//mpfr_set_zero(sum, +1);
			//for (int k=i;k>j;k--) {
				//mpfr_mul(temp, V.coeff(k, i), coeff(j, k), rnd_);
				//mpfr_add(sum, sum, temp, rnd_);
			//}
			//mpfr_sub(temp, coeff(i, i), coeff(j, j), rnd_);
			//mpfr_div(V.coeff(j, i), sum, temp, rnd_);
		//}
	//}
	//mpfr_clears(sum, temp, (mpfr_ptr) 0);
//}


void PreciseMatrix::split_real_eigenvalues (PreciseMatrix& U, PreciseMatrix& V) {
	mpfr_t tmp, a, b;
	mpfr_inits2(prec_, tmp, a, b, (mpfr_ptr) 0);
	for (size_t i=rows_-1;i>0;i--) {
		if (!mpfr_zero_p(coeff(i, i-1))) {
			//std::cerr << "eigenvalue pair found at " << i << std::endl;
			mpfr_sub(a, coeff(i-1, i-1), coeff(i, i), rnd_); // (a-d)
			mpfr_mul(tmp, coeff(i, i-1), coeff(i-1, i), rnd_); // (bc)
			mpfr_mul_d(b, tmp, 4.0, rnd_); // 4bc
			mpfr_sqr(tmp, a, rnd_); // (a-d)^2
			mpfr_add(b, b, tmp, rnd_); // (a-d)^2+4bc
			if (mpfr_sgn(b)>=0) {
				//std::cerr << "splitting eigenvalues" << std::endl;
				mpfr_sqrt(b, b, rnd_); // sqrt( (a-d)^2+4bc )
				mpfr_add(a, a, b, rnd_); // a+sqrt( (a-d)^2+4bc )
				mpfr_div_d(a, a, 2.0, rnd_); // ( a+sqrt( (a-d)^2+4bc ) )/2
				mpfr_div(a, a, coeff(i-1, i), rnd_); // ( a+sqrt( (a-d)^2+4bc ) )/2b
				apply_gaussian_elimination_left(i-1, i, a);
				V.apply_gaussian_elimination_left(i-1, i, a);
				mpfr_mul_d(a, a, -1.0, rnd_);
				apply_gaussian_elimination_right(i-1, i, a);
				U.apply_gaussian_elimination_right(i-1, i, a);
				mpfr_set_zero(coeff(i, i-1), +1);
			}
		}
	}
	mpfr_clears(tmp, a, b, (mpfr_ptr) 0);
}

// apply matrix
// ( 1, 0 ) -- row i
// ( g, 1 ) -- row j
// to the rows i and j
void PreciseMatrix::apply_gaussian_elimination_left (size_t i, size_t j, mpfr_t g) {
	mpfr_t tmp, a1, a2;
	mpfr_inits2(prec_, tmp, a1, a2, (mpfr_ptr) 0);
	for (size_t k=0;k<cols_;k++) {
		mpfr_mul(tmp, coeff(i, k), g, rnd_);
		mpfr_add(coeff(j, k), coeff(j, k), tmp, rnd_);
	}
	mpfr_clears(tmp, a1, a2, (mpfr_ptr) 0);
}

// apply matrix
// ( 1, 0 ) -- row i
// ( g, 1 ) -- row j
//   |   |
//   |   col j
//   col i
// to the cols i and j
void PreciseMatrix::apply_gaussian_elimination_right (size_t i, size_t j, mpfr_t g) {
	mpfr_t tmp, a1, a2;
	mpfr_inits2(prec_, tmp, a1, a2, (mpfr_ptr) 0);
	for (size_t k=0;k<rows_;k++) {
		mpfr_mul(tmp, coeff(k, j), g, rnd_);
		mpfr_add(coeff(k, i), coeff(k, i), tmp, rnd_);
	}
	mpfr_clears(tmp, a1, a2, (mpfr_ptr) 0);
}


void PreciseMatrix::solve_eigenproblem (PreciseMatrix& wr, PreciseMatrix& wi, PreciseMatrix& U, PreciseMatrix& V) {
	PreciseMatrix Q;
	U = Eigen::MatrixXd::Identity(rows_, cols_);
	V = Eigen::MatrixXd::Identity(rows_, cols_);
	reduce_to_hessenberg();
	extract_hessenberg_UV(U, V);
	for (size_t i=0;i<cols_;i++)
		for (size_t j=i+2;j<rows_;j++) {
			mpfr_set_zero(coeff(j, i), +1);
		}
	reduce_to_ev(wr, wi);
	applyOnTheLeft(Q);
	Q.transpose_in_place();
	applyOnTheRight(Q);
	//U.applyOnTheRight(Q);
	//Q.transpose_in_place();
	//V.applyOnTheLeft(Q);
	//split_real_eigenvalues(U, V);
}


