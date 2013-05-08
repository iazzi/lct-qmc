#ifndef REAL_TYPES_HPP
#define REAL_TYPES_HPP

//#define USE_LONG_DOUBLE

#ifdef USE_LONG_DOUBLE
typedef long double Real;
#elif USE_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

typedef std::complex<Real> Complex;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Matrix_d;
typedef Eigen::Array<Real, Eigen::Dynamic, 1> Array_d;
typedef Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic> Array_dd;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector_d;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> Vector_cd;


#endif // REAL_TYPES_HPP

