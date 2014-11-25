#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

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
		double M;
	public:
		Configuration (std::mt19937_64 &g, Model &m) : generator(g), model(m) {}
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

