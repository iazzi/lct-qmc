#ifndef SLICE_HPP
#define SLICE_HPP

#include <set>
#include <Eigen/Dense>
#include <cmath>
#include <iterator>

//FIXME
#include <iostream>

template <typename Model>
class Slice {
	public:
		typedef typename Model::Lattice Lattice;
		typedef typename Model::Interaction Interaction;
		typedef typename Interaction::Vertex Vertex;
		typedef typename Interaction::MatrixType MatrixType;
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
		size_t remove (const Vertex &v) { return verts.erase(v); }
		void clear () { verts.clear(); }

		Vertex get_vertex (size_t i) const {
			auto iter = verts.begin();
			std::advance(iter, i);
			return *iter;
		}

		Eigen::MatrixXd matrix () {
			matrix_.setIdentity(N, N);
			apply_matrix(matrix_);
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

		// TODO: apply_matrix_on_the_right(T &A)

		// apply the slice with forward propagators and direct vertices
		//template <typename T>
		//void apply_matrix (T &A, double tau) {
			//double t0 = 0.0;
			//for (auto v : verts) {
				//if (v.tau>=tau) break;
				//if (v.tau>t0) L->propagate(v.tau-t0, A);
				//t0 = v.tau;
				//I->apply_vertex_on_the_left(v, A);
			//}
			//if (tau>t0) L->propagate(tau-t0, A);
		//}

		// apply the slice with forward propagators and direct vertices
		template <typename T>
		void apply_on_the_right (T &A) {
			double t0 = beta;
			for (auto v=verts.rbegin();v!=verts.rend();v++) {
				//if (v.tau>=tau) break;
				if (v->tau<t0) L->propagate_on_the_right(t0-v->tau, A);
				t0 = v->tau;
				I->apply_vertex_on_the_right(*v, A);
			}
			if (t0>0.0) L->propagate_on_the_right(t0, A);
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

		// TODO: apply_inverse_on_the_right(T &A)

		double log_abs_det () {
			double ret = 0.0;
			for (auto v : verts) {
				ret += I->log_abs_det(v);
			}
			return ret;
		}

		double log_abs_det_block (size_t i) {
			double ret = 0.0;
			for (auto v : verts) {
				ret += I->log_abs_det_block(v, i);
			}
			return ret;
		}

		void matrixU (const Vertex v, MatrixType &u) {
			I->matrixU(v, u);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(w->tau-t0, u);
				t0 = w->tau;
				I->apply_vertex_on_the_left(*w, u);
			}
			if (beta>t0) L->propagate(beta-t0, u);
		}

		void matrixVt (const Vertex v, MatrixType &vt) {
			I->matrixV(v, vt);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(t0-w->tau, vt);
				t0 = w->tau;
				I->apply_inverse_on_the_left(*w, vt);
			}
			if (beta>t0) L->propagate(t0-beta, vt);
		}

		void inverseU (const Vertex v, MatrixType &u) {
			I->matrixU(v, u);
			u = -u;
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(w->tau-t0, u);
				t0 = w->tau;
				I->apply_vertex_on_the_left(*w, u);
			}
			if (beta>t0) L->propagate(beta-t0, u);
		}

		void inverseVt (const Vertex v, MatrixType &vt) {
			I->matrixV(v, vt);
			I->apply_inverse_on_the_left(v, vt);
			double t0 = v.tau;
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				if (w->tau>t0) L->propagate(t0-w->tau, vt);
				t0 = w->tau;
				I->apply_inverse_on_the_left(*w, vt);
			}
			if (beta>t0) L->propagate(t0-beta, vt);
		}

		void matrixU_right (const Vertex v, MatrixType &u) {
			I->matrixU(v, u);
			double t0 = v.tau;
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				if (w->tau<t0) L->propagate(w->tau-t0, u);
				t0 = w->tau;
				I->apply_inverse_on_the_left(*w, u);
			}
			if (t0>0.0) L->propagate(-t0, u);
			//std::cerr << (u-inverse()*matrixU(v)).norm() << std::endl;
		}

		void matrixVt_right (const Vertex v, MatrixType &vt) {
			I->matrixV(v, vt);
			double t0 = v.tau;
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				if (w->tau<t0) L->propagate(t0-w->tau, vt);
				t0 = w->tau;
				I->apply_vertex_on_the_left(*w, vt);
			}
			if (t0>0.0) L->propagate(t0, vt);
			//std::cerr << (vt-matrix().transpose()*matrixVt(v)).norm() << std::endl;
		}

		void inverseU_right (const Vertex v, MatrixType &u) {
			I->matrixU(v, u);
			u = -u;
			I->apply_inverse_on_the_left(v, u);
			double t0 = v.tau;
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				if (w->tau<t0) L->propagate(w->tau-t0, u);
				t0 = w->tau;
				I->apply_inverse_on_the_left(*w, u);
			}
			if (t0>0.0) L->propagate(-t0, u);
			//std::cerr << (u-inverse()*inverseU(v)).norm() << std::endl;
		}

		void inverseVt_right (const Vertex v, MatrixType &vt) {
			I->matrixV(v, vt);
			//I->apply_inverse_on_the_left(v, vt);
			double t0 = v.tau;
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				if (w->tau<t0) L->propagate(t0-w->tau, vt);
				t0 = w->tau;
				I->apply_vertex_on_the_left(*w, vt);
			}
			if (t0>0.0) L->propagate(t0, vt);
			//std::cerr << (vt-matrix().transpose()*inverseVt(v)).norm() << std::endl;
		}

		MatrixType matrixU (const Vertex v) {
			MatrixType u;
			matrixU(v, u);
			return u;
		}

		MatrixType matrixVt (const Vertex v) {
			MatrixType vt;
			matrixVt(v, vt);
			return vt;
		}

		MatrixType inverseU (const Vertex v) {
			MatrixType u;
			inverseU(v, u);
			return u;
		}

		MatrixType inverseVt (const Vertex v) {
			MatrixType vt;
			inverseVt(v, vt);
			return vt;
		}
};


#endif // SLICE_HPP

