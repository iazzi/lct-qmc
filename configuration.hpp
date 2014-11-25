#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "svd.hpp"
#include <vector>
#include "slice.hpp"


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

		double beta;
		double dtau;
		size_t M;

		SVDHelper svd;
		size_t index;

		void compute () {
			svd.setIdentity(model.lattice().volume()); // FIXME: amybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				svd.U.applyOnTheLeft(slices[(i+index)%M].matrix()); // FIXME: apply directly from the slice rather than multiplying the temporary
				svd.absorbU(); // FIXME: have a random matrix applied here possibly only when no vertices have been applied
			}
		}
	public:
		Configuration (std::mt19937_64 &g, Model &m) : generator(g), model(m), index(0) {}
		void setup (double b, size_t m) {
			beta = b;
			M = m;
			dtau = beta/M;
			slices.resize(M);
			for (size_t i=0;i<M;i++) {
				slices[i] = Slice<Model>(m);
				slices[i].setup(dtau);
			}
		}
};

#endif // CONFIGURATION_HPP

