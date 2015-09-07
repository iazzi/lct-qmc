#ifndef GENERICLATTICE_HPP
#define GENERICLATTICE_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

#include "parameters.hpp"
#include <fstream>

class GenericLattice {
	size_t V;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;

	Eigen::VectorXd eigenvalues_;
	Eigen::MatrixXd eigenvectors_;

	Eigen::MatrixXd H;

	bool computed;

	public:

	void setup (const Parameters &p) {
		if (p.contains("H")) {
			std::string fn = p.getString("H");
			std::ifstream in(fn);
			in >> V;
			H.resize(V, V);
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

	GenericLattice (): V(1), computed(false) { compute(); }
	GenericLattice (const GenericLattice &l): V(l.V), H(l.H), computed(false) { compute(); }
	GenericLattice (const Parameters &p): V(1), computed(false) { setup(p); compute(); }
};



#endif // GENERICLATTICE_HPP

