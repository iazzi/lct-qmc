#ifndef SLICE_HPP
#define SLICE_HPP

#include <set>
#include <Eigen/Dense>

template <typename Model>
class Slice {
	public:
	typedef typename Model::Lattice Lattice;
	typedef typename Model::Interaction Interaction;
	typedef typename Interaction::Vertex Vertex;
	private:
	Lattice &L;
	Interaction &I;

	std::set<Vertex, typename Vertex::Compare> verts;

	size_t N;
	double beta;

	Eigen::MatrixXd matrix_;
	Eigen::MatrixXd matrix_inv_;

	public:
	Slice (Model &m) : L(m.lattice()), I(m.interaction()), N(m.interaction().volume()) {}

	void setup (double b) {
		beta = b;
	}

	void insert (const Vertex &v) { verts.insert(v); }
	void clear () { verts.clear(); }

	Eigen::MatrixXd matrix () {
		matrix_.setIdentity(N, N);
		double t0 = 0.0;
		for (auto v : verts) {
			if (v.tau>t0) L.propagate(v.tau-t0, matrix_);
			t0 = v.tau;
			I.apply_vertex_on_the_left(v, matrix_);
		}
		return matrix_;
	}

	Eigen::MatrixXd inverse () {
		matrix_inv_.setIdentity(N, N);
		double t0 = beta;
		for (auto v=verts.rbegin();v!=verts.rend();v++) {
			if (v->tau<t0) L.propagate(v->tau-t0, matrix_inv_);
			t0 = v->tau;
			I.apply_inverse_on_the_left(v, matrix_inv_);
		}
		if (0.0<t0) L.propagate(-t0, matrix_inv_);
		return matrix_inv_;
	}
};


#endif // SLICE_HPP

