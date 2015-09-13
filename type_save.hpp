#ifndef TYPE_SAVE_HPP
#define TYPE_SAVE_HPP

#include "slice.hpp"
#include "hubbard.hpp"

namespace alps {
	namespace hdf5 {
		void save (archive & ar, std::string const & p, HubbardVertex const & v) {
			save(ar, p+"/x", v.x);
			save(ar, p+"/t", v.tau);
			save(ar, p+"/s", v.sigma);
		}
		void load (archive & ar, std::string const & p, HubbardVertex & v) {
			load(ar, p+"/x", v.x);
			load(ar, p+"/t", v.tau);
			load(ar, p+"/s", v.sigma);
		}

		template <typename M>
		void save (archive & ar, std::string const & p, Slice<M> const & s) {
		}
		template <typename M>
		void load (archive & ar, std::string const & p, Slice<M> & s) {
		}

		template <typename M>
		void save (archive & ar, std::string const & p, Configuration<M> const & c) {
			for (size_t i=0;i<c.slice_number();i++) {
				save(ar, p, c.slice(i));
			}
		}
		template <typename M>
		void load (archive & ar, std::string const & p, Configuration<M> & c) {
			for (size_t i=0;i<c.slice_number();i++) {
				load(ar, p, c.slice(i));
			}
		}

		template <typename T>
		void save (archive & ar, std::string const & p, Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> const & a) {
		}

		template <typename T>
		void load (archive & ar, std::string const & p, Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> & a) {
		}
	}
}

#endif // TYPE_SAVE_HPP

