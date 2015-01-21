#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "svd.hpp"
#include "slice.hpp"
#include <vector>
#include <cmath>

//FIXME
#include <iostream>


template <typename Model>
class Configuration {
	public:
		typedef typename Model::Lattice Lattice;
		typedef typename Model::Interaction Interaction;
		typedef typename Model::Interaction::MatrixType MatrixType;
		typedef typename Interaction::Vertex Vertex;

	private:
		std::vector<Slice<Model>> slices;

		std::mt19937_64 &generator;
		Model &model;

		double beta; // inverse temperature
		double mu; // chemical potential
		double dtau; // imaginary time size of a slice
		size_t M; // number of slices

		SVDHelper B; // this holds the deomposition of the matrix B
		std::vector<SVDHelper> blocks; // helper classes for the blocks
		SVDHelper G; // SVD decomposition of the Green function

		Eigen::MatrixXd G_matrix;

		size_t index; // This is the index of the LAST SLICE IN B

		Eigen::MatrixXd R; // a random matrix to solve degeneracies
		Eigen::MatrixXd R2; // inverse of R

		struct {
			Vertex v;
			MatrixType u, vt;
			double probability;
			Eigen::Matrix2d matrix;
		} cache;
	public:
		Configuration (std::mt19937_64 &g, Model &m) : generator(g), model(m), index(0) {}

		void setup (double b, double m, size_t n) {
			beta = b;
			mu = m;
			M = n;
			dtau = beta/M;
			slices.resize(M, Slice<Model>(model));
			for (size_t i=0;i<M;i++) {
				slices[i].setup(dtau);
			}
			// use block information
			blocks.resize(model.interaction().blocks());
		}

		void set_index (size_t i) { index = i%M; }

		double log_abs_det () {
			return B.S.array().abs().log().sum();
		}

		double slice_log_abs_det () {
			double ret = 0.0;
			for (size_t i=0;i<M;i++) {
				ret += slices[i].log_abs_det();
			}
			return ret;
		}

		double log_abs_det_block (size_t j) {
			double ret = 0.0;
			for (size_t i=0;i<M;i++) {
				ret += slices[i].log_abs_det_block(j);
			}
			return ret;
		}

		double log_abs_max () const {
			return B.S.array().abs().log().abs().maxCoeff();
		}

		void insert (Vertex v) {
			slices[index].insert(v);
		}

		size_t remove (Vertex v) {
			return slices[index].remove(v);
		}

		void insert_and_update (Vertex v, const MatrixType& matrixU, const MatrixType& matrixVt) {
			G_matrix -= (G_matrix * matrixU) * (Eigen::Matrix2d::Identity() + matrixVt.transpose() * G_matrix * matrixU).inverse() * (matrixVt.transpose() * G_matrix);
			G_matrix += matrixU * (matrixVt.transpose() * G_matrix);
			B.U += matrixU * (matrixVt.transpose() * B.U);
			insert(v);
		}

		void insert_and_update (Vertex v) {
			if (v==cache.v) {
			} else {
				insert_probability(v);
			}
			insert_and_update(v, cache.u, cache.vt);
		}

		size_t remove_and_update (Vertex v, const MatrixType& inverseU, const MatrixType& inverseVt) {
			G_matrix -= (G_matrix * inverseU) * (Eigen::Matrix2d::Identity() + inverseVt.transpose() * G_matrix * inverseU).inverse() * (inverseVt.transpose() * G_matrix);
			G_matrix += inverseU * (inverseVt.transpose() * G_matrix);
			B.U += inverseU * (inverseVt.transpose() * B.U);
			return remove(v);
		}

		size_t remove_and_update (Vertex v) {
			if (v==cache.v) {
			} else {
				remove_probability(v);
			}
			return remove_and_update(v, cache.u, cache.vt);
		}

		void commit_changes () {
			decompose_U();
			fix_sign_B();
		}

		void fix_sign_B () {
			for (int i=0;i<B.U.cols();i++) {
				double s = B.U.col(i).sum();
				if (s<0.0) {
					B.U.col(i) *= -1.0;
					B.Vt.row(i) *= -1.0;
				}
			}
		}

		void decompose_U () {
			for (size_t i=0;i<model.interaction().blocks();i++) {
				size_t a = model.interaction().block_start(i);
				size_t b = model.interaction().block_size(i);
				blocks[i].Vt = B.Vt.block(a, a, b, b);
				blocks[i].S = B.S.segment(a, b);
				blocks[i].U = B.U.block(a, a, b, b);
				blocks[i].absorbU();
				//blocks[i].fullSVD(B.U.block(a, a, b, b) * B.S.segment(a, b).asDiagonal());
				B.U.block(a, a, b, b) = blocks[i].U;
				B.S.segment(a, b) = blocks[i].S;
				B.Vt.block(a, a, b, b) = blocks[i].Vt;
			}
		}

		void compute_B () {
			if (R.size()<model.lattice().dimension()*model.lattice().dimension()) {
				R = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension()) + 0.000*Eigen::MatrixXd::Random(model.lattice().dimension(), model.lattice().dimension());
				R2 = R.inverse();
			}
			B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				slices[(i+index+1)%M].apply_matrix(B.U);
				decompose_U();
				// FIXME: have a random matrix applied here possibly only when no vertices have been applied
			}
			fix_sign_B();
		}

                // Wraps B(i) with B_{i+1} and B_{i+1}^{-1}, resulting in B(i+1)
		void wrap_B () {
			set_index(index+1);
                        //std::cerr << "Applying matrix on the left" << std::endl;
			slices[index].apply_matrix(B.U);
			decompose_U();
                        //std::cerr << "Applying inverse on the right" << std::endl;
			B.transposeInPlace();
			B.U.applyOnTheLeft(slices[index].inverse().transpose());
			decompose_U();
			B.transposeInPlace();
			fix_sign_B();
		}

                // Wraps B(i) with B_{i+1} and B_{i+1}^{-1}, resulting in B(i+1)
		void check_wrap_B () {
			set_index(index+1);
                        std::cerr << "Applying inverse on the right" << std::endl;
			B.transposeInPlace();
			B.U.applyOnTheLeft(slices[index].inverse().transpose());
			decompose_U();
			B.transposeInPlace();
                        fix_sign_B();
			for (size_t i=0;i<model.interaction().blocks();i++) {
				size_t a = model.interaction().block_start(i);
				size_t b = model.interaction().block_size(i);
				std::cerr << "block " << i << " -> " << B.S.segment(a, b).array().abs().log().sum() << ' ' << -slices[index].log_abs_det_block(i)+log_abs_det_block(i) << std::endl;
			}
			Eigen::MatrixXd t_U = B.U;
			Eigen::MatrixXd t_Vt = B.Vt;
			Eigen::VectorXd t_S = B.S;
                        B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
                        for (size_t i=0;i<M-1;i++) {
                                slices[(i+index+1)%M].apply_matrix(B.U);
                                decompose_U();
                        }
                        fix_sign_B();
			std::cerr << B.S.transpose() << std::endl;
			std::cerr << (t_S-B.S).transpose() << std::endl << std::endl;
			std::cerr << B.U-t_U << std::endl << std::endl;
			t_S.swap(B.S);
			t_U.swap(B.U);
			t_Vt.swap(B.Vt);

                        std::cerr << "Applying matrix on the left" << std::endl;
			slices[index].apply_matrix(B.U);
			decompose_U();
			fix_sign_B();
			for (size_t i=0;i<model.interaction().blocks();i++) {
				size_t a = model.interaction().block_start(i);
				size_t b = model.interaction().block_size(i);
				std::cerr << "block " << i << " -> " << B.S.segment(a, b).array().abs().log().sum() << ' ' << log_abs_det_block(i) << std::endl;
			}
		}

		void compute_G () {
			G = B; // B
			//G.invertInPlace(); // B^-1
			//G.add_identity(std::exp(-beta*mu)); // 1+exp(-beta*mu)*B^-1
			//G.invertInPlace(); // 1/(1+exp(-beta*mu)*B^-1) = B/(1+B)
			for (size_t i=0;i<model.interaction().blocks();i++) {
				size_t a = model.interaction().block_start(i);
				size_t b = model.interaction().block_size(i);
				blocks[i].Vt = B.Vt.block(a, a, b, b);
				blocks[i].S = B.S.segment(a, b);
				blocks[i].U = B.U.block(a, a, b, b);
				blocks[i].invertInPlace();
				blocks[i].add_identity(std::exp(-beta*mu));
				blocks[i].invertInPlace();
				G.U.block(a, a, b, b) = blocks[i].U;
				G.S.segment(a, b) = blocks[i].S;
				G.Vt.block(a, a, b, b) = blocks[i].Vt;
			}
		}

		void compute_G_alt () {
			G = B; // B
			for (size_t i=0;i<model.interaction().blocks();i++) {
				size_t a = model.interaction().block_start(i);
				size_t b = model.interaction().block_size(i);
				blocks[i].Vt = B.Vt.block(a, a, b, b);
				blocks[i].S = B.S.segment(a, b);
				blocks[i].U = B.U.block(a, a, b, b);
				G_matrix.block(a, a, b, b) = blocks[i].get_propagator(std::exp(-beta*mu));
			}
		}

		std::pair<double, double> probability () {
			SVDHelper A;
			A = B;
			A.add_identity(exp(beta*mu));
			std::pair<double, double> ret;
			ret.first = A.S.array().log().sum();
			ret.second = (A.U*A.Vt).determinant()>0.0?1.0:-1.0;
			return ret;
		}

		double insert_probability (Vertex v) {
			cache.v = v;
			slices[index].matrixU(v, cache.u);
			slices[index].matrixVt(v, cache.vt);
			cache.probability = (Eigen::Matrix2d::Identity() + cache.vt.transpose() * G_matrix * cache.u).determinant();
			return cache.probability;
		}

		double remove_probability (Vertex v) {
			cache.v = v;
			slices[index].inverseU(v, cache.u);
			slices[index].inverseVt(v, cache.vt);
			cache.probability = (Eigen::Matrix2d::Identity() + cache.vt.transpose() * G_matrix * cache.u).determinant();
			return cache.probability;
		}

		void save_G () {
			G_matrix = G.matrix();
		}

		double check_and_save_G () {
			double ret = 0.0;
			Eigen::MatrixXd A;
			A = G.matrix();
			//std::cerr << B.U << std::endl << std::endl;
			//std::cerr << B.Vt << std::endl << std::endl;
			//std::cerr << G_matrix.array() << std::endl << std::endl;
			//std::cerr << ((A-G_matrix).array().abs()>1e-9) << std::endl << std::endl;
			//std::cerr << cache.col(0).normalized().transpose() << std::endl << std::endl;
			//std::cerr << (G_matrix_up-A).col(0).normalized().transpose() << std::endl << std::endl;
			//Eigen::VectorXd B = (G_matrix_up-A).col(0).normalized();
			//Eigen::JacobiSVD<Eigen::MatrixXd> svd(G_matrix_up, Eigen::ComputeThinU | Eigen::ComputeThinV);
			//std::cerr << (svd.solve(B)).transpose().normalized() << std::endl << std::endl;
			ret += (G_matrix-A).norm();
			G_matrix.swap(A);
			return ret;
		}

		double check_B () {
			double ret = 0.0;
			Eigen::MatrixXd t_U, t_Vt;
			t_U = B.U;
			t_Vt = B.Vt;
			//tmp = B.matrix();
			//std::cerr << B.S.array().abs().log().sum() << std::endl;
                        compute_B();
			fix_sign_B();
			//std::cerr << B.S.array().abs().log().sum() << std::endl;
                        //std::cerr << t_U << std::endl << std::endl;
                        //std::cerr << std::endl;
                        std::cerr << t_U-B.U << std::endl;
			ret += (t_U-B.U).norm();
			ret += (t_Vt-B.Vt).norm();
			//if (ret>1.0e-6) throw -1;
			return ret;
		}

		const Eigen::MatrixXd & green_function () {
			return G_matrix;
		}

		double slice_start () const { return dtau*index; }
		double slice_end () const { return dtau*(index+1); }
		size_t slice_size () const { return slices[index].size(); }
		size_t slice_number () const { return M; } // MUST be same as slices.size()

		Vertex get_vertex (size_t i) const { return slices[index].get_vertex(i); }

		double insert_factor () { return +log(beta/slice_number()) -log(slice_size()+1) +model.interaction().combinatorial_factor(); }
		double remove_factor () { return -log(beta/slice_number()) +log(slice_size()+0) -model.interaction().combinatorial_factor(); }

		size_t size () const { size_t ret = 0; for (const auto &s : slices) ret += s.size(); return ret; }
		void show_verts () const { for (const auto &s : slices) std::cerr << s.size() << std::endl; }
		void advance (int n) { set_index(2*M+index+n); }
};

#endif // CONFIGURATION_HPP

