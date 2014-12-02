#ifndef CUBICLATTICE_HPP
#define CUBICLATTICE_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class CubicLattice {
	size_t Lx, Ly, Lz;
	size_t V;
	double tx, ty, tz;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;

	bool computed;

	public:

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
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(V, V);
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
		Eigen::MatrixXd H2 = Eigen::MatrixXd::Zero(2*V, 2*V);
		H2.block(0, 0, V, V) = H;
		H2.block(V, V, V, V) = H;
		solver.compute(H2);
		if (solver.info()==Eigen::Success) computed = true;
	}

	const typename Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>::RealVectorType & eigenvalues () const { return solver.eigenvalues(); }
	const typename Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>::MatrixType & eigenvectors () const { return solver.eigenvectors(); }

	size_t volume () const { return V; }
	size_t states () const { return 2*V; }
	size_t dimension () const { return 2*V; }

	template <typename T>
		void propagate (double t, T& M) {
			M.array().colwise() *= (-t*solver.eigenvalues().array()).exp();
		}

	CubicLattice (): Lx(2), Ly(2), Lz(1), V(4), tx(1.0), ty(1.0), tz(1.0), computed(false) {}
};



#endif // CUBICLATTICE_HPP

