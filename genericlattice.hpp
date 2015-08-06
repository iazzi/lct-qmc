#ifndef GENERICLATTICE_HPP
#define GENERICLATTICE_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "parameters.hpp"
#include <fstream>

class GenericLattice {
	size_t V;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;

	Eigen::VectorXd eigenvalues_;
	Eigen::MatrixXd eigenvectors_;

	Eigen::MatrixXd H;

	bool computed;

	Eigen::VectorXd cached_exp;

	public:

	void setup (const Parameters &p) {
		// get dimensions
		if (p.contains("V")) {
			V = p.getInteger("V");
		} else {
			V = 1;
		}
		if (p.contains("H")) {
			H.resize(V, V);
			std::string fn = p.getString("H");
			std::ifstream in(fn);
			for (size_t x=0;x<V;x++) {
				for (size_t y=0;y<V;y++) {
					double z;
					in >> z;
					H(x, y) = z;
				}
			}
		}
		computed = false;
	}

	void compute () {
		if (computed) return;
		solver.compute(H);
		eigenvectors_ = solver.eigenvectors();
		eigenvalues_ = solver.eigenvalues();
		computed = true;
	}

	const Eigen::VectorXd & eigenvalues () const { return eigenvalues_; }
	const Eigen::MatrixXd & eigenvectors () const { return eigenvectors_; }

	size_t volume () const { return V; }
	size_t states () const { return V; }
	size_t dimension () const { return V; }

	template <typename T>
		void propagate (double t, T& M) {
			cached_exp = eigenvalues_;
			cached_exp *= -t;
			cached_exp = cached_exp.array().exp();
			M.applyOnTheLeft(cached_exp.matrix().asDiagonal());
			//M.array().colwise() *= (-t*eigenvalues_.array()).exp(); // this causes allocation!
		}

	template <typename T>
		double kinetic_energy (const T &M) {
			return (eigenvalues_.array() * M.diagonal().array()).sum();
		}

	GenericLattice (): V(1), computed(false) { compute(); }
	GenericLattice (const GenericLattice &l): V(l.V), H(l.H), computed(false) { compute(); }
	GenericLattice (const Parameters &p): V(1), computed(false) { setup(p); compute(); }
};



#endif // GENERICLATTICE_HPP

