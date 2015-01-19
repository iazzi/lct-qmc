#ifndef CUBICLATTICE_HPP
#define CUBICLATTICE_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "parameters.hpp"

class CubicLattice {
	size_t Lx, Ly, Lz;
	size_t V;
	double tx, ty, tz;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;

	Eigen::VectorXd eigenvalues_;
	Eigen::MatrixXd eigenvectors_;

	bool computed;

	Eigen::MatrixXd H;

	Eigen::VectorXd cached_exp;

	public:

	void setup (const Parameters &p) {
		// get dimensions
		if (p.contains("L")) {
			Lx = Ly = Lz = p.getInteger("L");
		} else {
			Lx = Ly = Lz = 1;
		}
		if (p.contains("Lx")) {
			Lx = p.getInteger("Lx");
		}
		if (p.contains("Ly")) {
			Ly = p.getInteger("Ly");
		}
		if (p.contains("Lz")) {
			Lz = p.getInteger("Lz");
		}
		V = Lx*Ly*Lz;
		// get tunneling coefficient
		if (p.contains("t")) {
			tx = ty = tz = p.getNumber("t");
		} else {
			tx = ty = tz = 1;
		}
		if (p.contains("tx")) {
			tx = p.getNumber("tx");
		}
		if (p.contains("ty")) {
			ty = p.getNumber("ty");
		}
		if (p.contains("tz")) {
			tz = p.getNumber("tz");
		}
		computed = false;
	}

	void set_size (size_t a, size_t b, size_t c) {
		Lx = a;
		Ly = b;
		Lz = c;
		V = a*b*c;
		computed = false;
	}

	void set_tunnelling (double a, double b, double c) {
		tx = a;
		ty = b;
		tz = c;
		computed = false;
	}

	void compute () {
		if (computed) return;
		V = Lx*Ly*Lz;
		H = Eigen::MatrixXd::Zero(V, V);
		for (size_t x=0;x<Lx;x++) {
			for (size_t y=0;y<Ly;y++) {
				for (size_t z=0;z<Lz;z++) {
					size_t a = x*Ly*Lz + y*Lz + z;
					size_t b = ((x+1)%Lx)*Ly*Lz + y*Lz + z;
					size_t c = x*Ly*Lz + ((y+1)%Ly)*Lz + z;
					size_t d = x*Ly*Lz + y*Lz + (z+1)%Lz;
					if (Lx>1 && x!=Lx-0) H(a, b) = H(b, a) = -tx;
					if (Ly>1 && y!=Ly-0) H(a, c) = H(c, a) = -ty;
					if (Lz>1 && z!=Lz-0) H(a, d) = H(d, a) = -tz;
				}
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
			cached_exp = eigenvalues_;
			cached_exp *= -t;
			cached_exp = cached_exp.exp();
			M.applyOnTheLeft(cached_exp.matrix().asDiagonal());
			//M.array().colwise() *= (-t*eigenvalues_.array()).exp(); // this causes allocation!
		}

	template <typename T>
		double kinetic_energy (const T &M) {
			return (eigenvalues_.array() * M.diagonal().array()).sum();
		}

	CubicLattice (): Lx(2), Ly(2), Lz(1), tx(1.0), ty(1.0), tz(1.0), computed(false) {}
	CubicLattice (const CubicLattice &l): Lx(l.Lx), Ly(l.Ly), Lz(l.Lz), tx(l.tx), ty(l.ty), tz(l.tz), computed(false) {}
	CubicLattice (const Parameters &p): Lx(2), Ly(2), Lz(1), tx(1.0), ty(1.0), tz(1.0), computed(false) { setup(p); compute(); }
};



#endif // CUBICLATTICE_HPP

