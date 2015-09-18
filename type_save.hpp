#ifndef TYPE_SAVE_HPP
#define TYPE_SAVE_HPP

#include <Eigen/Dense>

#include "slice.hpp"
#include "hubbard.hpp"
#include "configuration.hpp"

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/inf.hpp>

//#include <iostream>

namespace alps {
	namespace numeric {
		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> sin (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.sin();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> cos (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.cos();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> tan (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.tan();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> sinh (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.sinh();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> cosh (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.cosh();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> tanh (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.tanh();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> asin (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.asin();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> acos (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.acos();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> atan (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.atan();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> abs (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.abs();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> exp (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.exp();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> sqrt (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.sqrt();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> log (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.log();
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> sq (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>(x*x);
		}

		template <typename Derived>
		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> sq (Eigen::EigenBase<Derived> const & x) {
			return Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>(x*x);
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> pow (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x, double y) {
			return x.pow(y);
		}

		inline Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> cbrt (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) {
			return x.pow(1.0/3.0);
		}

		template <> struct invert<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> {
			Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> operator() (Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> x) { return x.inverse(); }
		};

		template <> struct inf<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> {
			operator Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> () const { return Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>(); }
		};
	}
}


inline void save (alps::hdf5::archive & ar, std::string const& p, HubbardVertex const & v) {
    ar[p+"/x"] << v.x;
    ar[p+"/t"] << v.tau;
    ar[p+"/s"] << v.sigma;
}
inline void load (alps::hdf5::archive & ar, std::string const& p, HubbardVertex & v) {
    ar[p+"/x"] >> v.x;
    ar[p+"/t"] >> v.tau;
    ar[p+"/s"] >> v.sigma;
}

template <typename M>
inline void save (alps::hdf5::archive & ar, std::string const& p, Slice<M> const & s) {
	ar[p+"/size"] << s.size();
	for (size_t i=0;i<s.size();i++) {
		ar[p+"/"+std::to_string(i)] << s.get_vertex(i);
	}
}
template <typename M>
inline void load (alps::hdf5::archive & ar, std::string const& p, Slice<M> & s) {
	s.clear();
	size_t size;
	ar[p+"/size"] >> size;
	for (size_t i=0;i<size;i++) {
		typename Slice<M>::Vertex v;
		ar[p+"/"+std::to_string(i)] >> v;
		s.prepare(v);
		s.insert(v);
	}
}

template <typename M>
inline void save (alps::hdf5::archive & ar, std::string const& p, Configuration<M> const & c) {
	for (size_t i=0;i<c.slice_number();i++) {
		//std::cerr << i << ' ' << c.slice(i).size() << std::endl;
		ar[p+"/"+std::to_string(i)] << c.slice(i);
	}
}
template <typename M>
inline void load (alps::hdf5::archive & ar, std::string const& p, Configuration<M> & c) {
	for (size_t i=0;i<c.slice_number();i++) {
		ar[p+"/"+std::to_string(i)] >> c.slice(i);
		//std::cerr << i << ' ' << c.slice(i).size() << std::endl;
	}
}


namespace alps {
	namespace hdf5 {
		template <typename T>
		inline void save (archive & ar, std::string const & p, Eigen::ArrayBase<T> const & a) {
		}

		template <typename T>
		inline void load (archive & ar, std::string const & p, Eigen::ArrayBase<T> & a) {
		}

		template <> inline void save<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> (archive & ar, std::string const & p, Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> const & a, std::vector<std::size_t> size, std::vector<std::size_t> chunk, std::vector<std::size_t> offset) {
		}

		template <> inline void load<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> (archive & ar, std::string const & p, Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> & , std::vector<std::size_t> chunk, std::vector<std::size_t> offset) {
		}

		template<> struct scalar_type<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > {
			typedef double type;
		};

	}
}

#endif // TYPE_SAVE_HPP

