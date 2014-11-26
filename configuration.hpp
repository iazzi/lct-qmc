#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "svd.hpp"
#include "slice.hpp"
#include <vector>
#include <cmath>


template <typename Model>
class Configuration {
	public:
		typedef typename Model::Lattice Lattice;
		typedef typename Model::Interaction Interaction;
		typedef typename Interaction::Vertex Vertex;

	private:
		std::vector<Slice<Model>> slices;

		std::mt19937_64 &generator;
		Model &model;

		double beta, mu;
		double dtau;
		size_t M;

		SVDHelper svd; // this holds the deomposition of the matrix B for the up species
		SVDHelper G_up, G_dn;

		size_t index; // This is the index of the LAST SLICE IN B

	public:
		Configuration (std::mt19937_64 &g, Model &m) : generator(g), model(m), index(0) {}

		void setup (double b, double m, size_t n) {
			beta = b;
			mu = m;
			M = n;
			dtau = beta/M;
			slices.resize(M, Slice<Model>(model));
			for (size_t i=0;i<M;i++) {
				slices[i].setup(dtau);
			}
		}

		void set_index (size_t i) { index = i%M; }

		double log_abs_det () {
			return svd.S.array().abs().log().sum();
		}

		double slice_log_abs_det () {
			double ret = 0.0;
			for (size_t i=0;i<M;i++) {
				ret += slices[i].log_abs_det();
			}
			return ret;
		}

		void insert (Vertex v) {
			if (v.tau<beta) {
				size_t i = v.tau/dtau;
				v.tau -= i*dtau;
				slices[i].insert(v);
			}
		}

		void compute_B () {
			svd.setIdentity(model.lattice().volume()); // FIXME: maybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				svd.U.applyOnTheLeft(slices[(i+index+1)%M].matrix()); // FIXME: apply directly from the slice rather than multiplying the temporary
				svd.absorbU(); // FIXME: have a random matrix applied here possibly only when no vertices have been applied
			}
		}

		void compute_G () {
			G_up = svd; // B
			G_up.invertInPlace(); // B^-1
			G_up.add_identity(std::exp(-beta*mu)); // 1+exp(-beta*mu)*B^-1
			G_up.invertInPlace(); // 1/(1+exp(-beta*mu)*B^-1) = B/(1+B)
			G_dn = svd;
			//G_dn.invertInPlace(); // the down part is automatically the inverse of the up part
			G_dn.add_identity(std::exp(beta*mu));
			G_dn.invertInPlace();
		}

		std::pair<double, double> probability () {
			SVDHelper A_up, A_dn;
			A_up = svd;
			A_dn = svd;
			A_up.add_identity(exp(beta*mu));
			A_dn.add_identity(exp(beta*mu));
			std::pair<double, double> ret;
			ret.first = A_up.S.array().log().sum() + A_dn.S.array().log().sum();
			ret.second = (A_up.U*A_up.Vt*A_dn.U*A_dn.Vt).determinant()>0.0?1.0:-1.0;
			return ret;
		}
};

#endif // CONFIGURATION_HPP

