#ifndef HONEYCOMB_HPP
#define HONEYCOMB_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "parameters.hpp"

class HoneycombLattice {
	size_t Lx, Ly;
	size_t V;
	double tx, ty;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;

	Eigen::VectorXd eigenvalues_;
	Eigen::MatrixXd eigenvectors_;

	bool computed;

	public:

	void setup (const Parameters &p) {
		// get dimensions
		if (p.contains("L")) {
			Lx = Ly = p.getInteger("L");
		} else {
			Lx = Ly = 1;
		}
		if (p.contains("Lx")) {
			Lx = p.getInteger("Lx");
		}
		if (p.contains("Ly")) {
			Ly = p.getInteger("Ly");
		}
		// get tunneling coefficient
		if (p.contains("t")) {
			tx = ty = p.getNumber("t");
		} else {
			tx = ty = 1;
		}
		if (p.contains("tx")) {
			tx = p.getNumber("tx");
		}
		if (p.contains("ty")) {
			ty = p.getNumber("ty");
		}
		computed = false;
	}

	void compute () {
		if (computed) return;
		V = Lx*Ly;
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(V, V);
		for (size_t x=0;x<Lx;x++) {
			for (size_t y=0;y<Ly;y++) {
				size_t a = x*Ly + y;
				size_t b = ((x+1)%Lx)*Ly + y;
				size_t c = x*Ly + (y+1)%Ly;
				if (Lx>1 && x!=Lx-0) H(a, b) = H(b, a) = -tx;
				if (Ly>1 && y!=Ly-0 && (x+y)%2==0) H(a, c) = H(c, a) = -ty;
			}
		}
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
			M.array().colwise() *= (-t*eigenvalues_.array()).exp();
		}

	HoneycombLattice (): Lx(2), Ly(2), Lz(1), tx(1.0), ty(1.0), tz(1.0), computed(false) {}
	HoneycombLattice (const HoneycombLattice &l): Lx(l.Lx), Ly(l.Ly), Lz(l.Lz), tx(l.tx), ty(l.ty), tz(l.tz), computed(false) {}
	HoneycombLattice (const Parameters &p): Lx(2), Ly(2), Lz(1), tx(1.0), ty(1.0), tz(1.0), computed(false) { setup(p); }
};



#endif // HONEYCOMB_HPP

