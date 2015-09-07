#ifndef SLICE_HPP
#define SLICE_HPP

#include <set>
#include <Eigen/Dense>
#include <cmath>
#include <iterator>

//FIXME
#include <iostream>

template <typename Interaction>
class Slice {
	public:
		typedef typename Interaction::Vertex Vertex;
		typedef typename Interaction::MatrixType MatrixType;
	private:
		Interaction *I;

		std::set<Vertex, typename Vertex::Compare> verts;

		size_t N;
		double beta;

	public:
		Slice (Interaction *i, double b = 1.0) : I(i), N(i->dimension()), beta(b) {}
		Slice (const Slice &s) : I(s.I), N(s.N), beta(s.beta) {}

		size_t size () const { return verts.size(); }
		void insert (const Vertex &v) { verts.insert(v); }
		size_t remove (const Vertex &v) { return verts.erase(v); }
		void clear () { verts.clear(); }

		Vertex get_vertex (size_t i) const {
			auto iter = verts.begin();
			std::advance(iter, i);
			return *iter;
		}

		// apply the slice with forward propagators and direct vertices
		template <typename T>
		void apply_matrix (T &A) {
			for (auto v=verts.begin();v!=verts.end();v++) {
				I->apply_displaced_vertex_on_the_left(*v, A);
			}
			I->propagate(beta, A);
		}

		// apply the slice with forward propagators and direct vertices
		template <typename T>
		void apply_on_the_right (T &A) {
			I->propagate_on_the_right(beta, A);
			for (auto v=verts.rbegin();v!=verts.rend();v++) {
				I->apply_displaced_vertex_on_the_right(*v, A);
			}
		}

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

		void matrixU (const Vertex &v, MatrixType &u) {
			u = v.data.U * v.data.mat.asDiagonal();
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				I->apply_displaced_vertex_on_the_left(*w, u);
			}
			I->propagate(beta, u);
		}

		Eigen::Matrix<double, 2, Eigen::Dynamic> vtt;
		void matrixVt (const Vertex &v, MatrixType &vt) {
			vtt = v.data.V.transpose();
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				I->apply_displaced_inverse_on_the_right(*w, vtt);
			}
			vt = vtt.transpose();
			I->propagate(-beta, vt);
		}

		void inverseU (const Vertex &v, MatrixType &u) {
			u = -v.data.U * v.data.mat.asDiagonal();
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				I->apply_displaced_vertex_on_the_left(*w, u);
			}
			I->propagate(beta, u);
		}

		void inverseVt (const Vertex &v, MatrixType &vt) {
			vtt = v.data.V.transpose();
			I->apply_displaced_inverse_on_the_right(v, vtt);
			for (auto w = verts.upper_bound(v);w!=verts.end();w++) {
				I->apply_displaced_inverse_on_the_right(*w, vtt);
			}
			vt = vtt.transpose();
			I->propagate(-beta, vt);
		}

		void matrixU_right (const Vertex &v, MatrixType &u) {
			u = v.data.U * v.data.mat.asDiagonal();
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				I->apply_displaced_inverse_on_the_left(*w, u);
			}
		}

		void matrixVt_right (const Vertex &v, MatrixType &vt) {
			vtt = v.data.V.transpose();
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				I->apply_displaced_vertex_on_the_right(*w, vtt);
			}
			vt = vtt.transpose();
		}

		void inverseU_right (const Vertex &v, MatrixType &u) {
			u = -v.data.U * v.data.mat.asDiagonal();
			I->apply_displaced_inverse_on_the_left(v, u);
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				I->apply_displaced_inverse_on_the_left(*w, u);
			}
		}

		void inverseVt_right (const Vertex &v, MatrixType &vt) {
			vtt = v.data.V.transpose();
			auto w = verts.lower_bound(v);
			for (;w!=verts.begin();) {
				w--;
				I->apply_displaced_vertex_on_the_right(*w, vtt);
			}
			vt = vtt.transpose();
		}

		MatrixType matrixU (const Vertex &v) {
			MatrixType u;
			matrixU(v, u);
			return u;
		}

		MatrixType matrixVt (const Vertex &v) {
			MatrixType vt;
			matrixVt(v, vt);
			return vt;
		}

		MatrixType inverseU (const Vertex &v) {
			MatrixType u;
			inverseU(v, u);
			return u;
		}

		MatrixType inverseVt (const Vertex &v) {
			MatrixType vt;
			inverseVt(v, vt);
			return vt;
		}
};


#endif // SLICE_HPP

