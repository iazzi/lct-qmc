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

PreciseMatrix operator* (const Eigen::MatrixXd& A, const PreciseMatrix& B) {
	PreciseMatrix C(B.precision());
	C = Eigen::MatrixXd::Zero(A.rows(), B.cols());
	mpfr_t c;
	mpfr_init2(c, B.precision());
	for (int i=0;i<A.rows();i++){
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

ostream& operator<< (ostream& out, const PreciseMatrix& A) {
	for (size_t i=0;i<A.rows();i++) {
		for (size_t j=0;j<A.cols();j++) {
			mpfr_exp_t e;
			char *s = mpfr_get_str(NULL, &e, 10, 0, A.coeff(i, j), A.rnd());
			if (mpfr_sgn(A.coeff(i, j))<0) {
				out << s[0] << s[1] << '.' << &s[2];
			} else {
				out << s[0] << '.' << &s[1];
			}
			out << 'e' << e << (j==A.cols()-1?'\n':' ');
			mpfr_free_str(s);
		}
	}
	return out;
}

void PreciseMatrix::swap (PreciseMatrix& other) {
	std::swap(prec_, other.prec_);
	std::swap(rows_, other.rows_);
	std::swap(cols_, other.cols_);
	std::swap(rnd_, other.rnd_);
	std::swap(data_, other.data_);
}

int PreciseMatrix::inPlaceLU () {
	const int N = rows();
	int d = 1;
	int i,imax,j,k;
	mpfr_t big, dum, sum, temp;
	mpfr_inits2(prec_, big, dum, sum, temp, (mpfr_ptr) 0);
	PreciseMatrix vv; //   vv stores the implicit scaling of each row.
	vv = Eigen::VectorXd::Zero(N);
	int indx[N];
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
		mpfr_set_zero(big, rnd_);
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
		indx[j]=imax;
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
	return d;
}



//int main () {
	//Eigen::MatrixXd R = Eigen::MatrixXd::Random(4, 4);
	//Eigen::PartialPivLU<Eigen::MatrixXd> LU(R);
	//PreciseMatrix A, B;
	//A = R;
	//B = Eigen::MatrixXd::Identity(3, 4);
	//cout << A << endl;
	//cout << B << endl;
	////A.applyOnTheLeft(Eigen::MatrixXd::Identity(3, 4));
	//A.inPlaceLU();
	//cout << A << endl;
	//cout << LU.matrixLU() << endl;
//}
