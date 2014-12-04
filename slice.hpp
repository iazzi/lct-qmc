#ifndef SLICE_HPP
#define SLICE_HPP

#include <set>
#include <Eigen/Dense>
#include <cmath>

template <typename Model>
class Slice {
	public:
		typedef typename Model::Lattice Lattice;
		typedef typename Model::Interaction Interaction;
		typedef typename Interaction::Vertex Vertex;
		typedef typename Interaction::UpdateType UpdateType;
	private:
		Lattice *L;
		Interaction *I;

		std::set<Vertex, typename Vertex::Compare> verts;

		size_t N;
		double beta;

		Eigen::MatrixXd matrix_;
		Eigen::MatrixXd matrix_inv_;

	public:
		Slice (Model &m) : L(&m.lattice()), I(&m.interaction()), N(m.interaction().dimension()), beta(1.0) {}
		Slice (const Slice &s) : L(s.L), I(s.I), N(s.N), beta(s.beta) {}

		void setup (double b) {
			beta = b;
		}

		size_t size () const { return verts.size(); }
		void insert (const Vertex &v) { verts.insert(v); }
		void clear () { verts.clear(); }

		Eigen::MatrixXd matrix () {
			matrix_.setIdentity(N, N);
			double t0 = 0.0;
			for (auto v : verts) {
				if (v.tau>t0) L->propagate(v.tau-t0, matrix_);
				t0 = v.tau;
				I->apply_vertex_on_the_left(v, matrix_);
			}
			if (beta>t0) L->propagate(beta-t0, matrix_);
			return matrix_;
		}

		Eigen::MatrixXd inverse () {
			matrix_inv_.setIdentity(N, N);
			double t0 = beta;
			for (auto v=verts.rbegin();v!=verts.rend();v++) {
				if (v->tau<t0) L->propagate(v->tau-t0, matrix_inv_);
				t0 = v->tau;
				I->apply_inverse_on_the_left(*v, matrix_inv_);
			}
			if (0.0<t0) L->propagate(-t0, matrix_inv_);
			return matrix_inv_;
		}

		// apply the slice with forward propagators and direct vertices
		template <typename T>
		void apply_matrix (T &A) {
			double t0 = 0.0;
			for (auto v : verts) {
				if (v.tau>t0) L->propagate(v.tau-t0, A);
				t0 = v.tau;
				I->apply_vertex_on_the_left(v, A);
			}
			if (beta>t0) L->propagate(beta-t0, A);
		}

		// apply the inverse slice (with backward propagators and inverse vertices)
		template <typename T>
		void apply_inverse (T &A) {
			double t0 = beta;
			for (auto v=verts.rbegin();v!=verts.rend();v++) {
				if (v->tau<t0) L->propagate(v->tau-t0, A);
				t0 = v->tau;
				I->apply_inverse_on_the_left(*v, A);
			}
			if (0.0<t0) L->propagate(-t0, A);
		}

		// apply the slice with forward propagators and direct vertices
		template <typename T>
		void apply_matrix_11 (T &A) {
			double t0 = 0.0;
			for (auto v : verts) {
				if (v.tau>t0) L->propagate(v.tau-t0, A);
				t0 = v.tau;
				I->apply_vertex_on_the_left(v, A);
			}
			if (beta>t0) L->propagate(beta-t0, A);
		}

		// apply the slice with forward propagators and inverse vertices
		template <typename T>
		void apply_matrix_12 (T &A) {
			double t0 = 0.0;
			for (auto v : verts) {
				if (v.tau>t0) L->propagate(v.tau-t0, A);
				t0 = v.tau;
				I->apply_inverse_on_the_left(v, A);
			}
			if (beta>t0) L->propagate(beta-t0, A);
		}

		// apply the slice with inverse propagators and forward vertices
		template <typename T>
		void apply_matrix_21 (T &A) {
			double t0 = beta;
			for (auto v=verts.rbegin();v!=verts.rend();v++) {
				if (v->tau<t0) L->propagate(v->tau-t0, A);
				t0 = v->tau;
				I->apply_vertex_on_the_left(*v, A);
			}
			if (0.0<t0) L->propagate(-t0, A);
		}

		// apply the slice with inverse propagators and inverse vertices
		template <typename T>
		void apply_matrix_22 (T &A) {
			double t0 = beta;
			for (auto v=verts.rbegin();v!=verts.rend();v++) {
				if (v->tau<t0) L->propagate(v->tau-t0, A);
				t0 = v->tau;
				I->apply_inverse_on_the_left(*v, A);
			}
			if (0.0<t0) L->propagate(-t0, A);
		}

		double log_abs_det () {
			double ret = 0.0;
			for (auto v : verts) {
				ret += I->log_abs_det(v);
			}
			return ret;
		}

		UpdateType matrixU (const Vertex v) {
			UpdateType u = I->matrixU(v);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(w->tau-t0, u);
				t0 = w->tau;
				I->apply_vertex_on_the_left(*w, u);
			}
			if (beta>t0) L->propagate(beta-t0, u);
			return u;
		}

		UpdateType matrixVt (const Vertex v) {
			UpdateType vt = I->matrixVt(v);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(t0-w->tau, vt);
				t0 = w->tau;
				I->apply_inverse_on_the_left(*w, vt);
			}
			if (beta>t0) L->propagate(t0-beta, vt);
			return vt;
		}

		UpdateType matrixU2 (const Vertex v) {
			UpdateType u = I->matrixU(v);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(w->tau-t0, u);
				t0 = w->tau;
				I->apply_inverse_on_the_left(*w, u);
			}
			if (beta>t0) L->propagate(beta-t0, u);
			return u;
		}

		UpdateType matrixVt2 (const Vertex v) {
			UpdateType vt = I->matrixVt(v);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(t0-w->tau, vt);
				t0 = w->tau;
				I->apply_vertex_on_the_left(*w, vt);
			}
			if (beta>t0) L->propagate(t0-beta, vt);
			return vt;
		}
};


#endif // SLICE_HPP

