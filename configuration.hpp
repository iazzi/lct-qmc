#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "parameters.hpp"
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

		typedef enum {
			left_to_right,
			right_to_left
		} sweep_direction_type;

	private:
		std::vector<Slice<Model>> slices;
		std::vector<SVDHelper> right_side;
		std::vector<SVDHelper> left_side;

		Model model;

		double beta; // inverse temperature
		double mu; // chemical potential
		double dtau; // imaginary time size of a slice
		size_t M; // number of slices

		SVDHelper B; // this holds the deomposition of the matrix B
		std::vector<SVDHelper> blocks; // helper classes for the blocks
		SVDHelper G; // SVD decomposition of the Green function

		Eigen::MatrixXd G_matrix;
		Eigen::MatrixXd big_matrix;

		size_t index; // This is the index of the LAST SLICE IN B

		struct {
			Vertex v;
			MatrixType u, vt;
			double probability;
			Eigen::Matrix2d matrix;
		} cache;

		sweep_direction_type sweep_direction_;
	public:
		Configuration (const Model &m) : model(m), index(0) {}
		Configuration (const Parameters &p) : model(p), index(0) {}

		void setup (const Parameters &p) {
			beta = p.getNumber("beta", 1.0);
			mu = p.getNumber("mu", 0.0);
			M = p.getInteger("slices", 4*beta);
			dtau = beta/M;
			slices.resize(M, Slice<Model>(model));
			right_side.resize(M+1);
			left_side.resize(M+1);
			for (size_t i=0;i<M;i++) {
				slices[i].setup(dtau);
			}
			// use block information
			blocks.resize(model.interaction().blocks());
			size_t I = p.getInteger("sites", 0);
			if (I>0) model.interaction().set_interactive_sites(I);
		}

		void set_index (size_t i) { index = i%M; }
		void set_slice (size_t i) { index = i%M; }

		void compute_right_side (size_t j) {
			if (j==0) {
				B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
			} else {
				B = right_side[j-1];
				slices[j-1].apply_matrix(B.U);
				decompose_U();
				fix_sign_B();
			}
			right_side[j] = B;
		}

		void compute_left_side (size_t j) {
			if (j==M) {
				B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
			} else {
				B = left_side[j+1];
				slices[j].apply_on_the_right(B.Vt);
				//B.Vt.applyOnTheRight(slices[j+1].matrix());
				decompose_Vt();
				fix_sign_B();
			}
			left_side[j] = B;
		}

		void check_propagation_from_right () {
			Eigen::MatrixXd M, M1, M2;
			M1 = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension());
			model.lattice().propagate(0.023, M1);
			M2 = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension());
			model.lattice().propagate_on_the_right(0.023, M2);
			std::cerr << "propagate " << (M1-M2).norm() << std::endl;
			Vertex v = model.interaction().generate(0.0, 0.2);
			M1 = M2 = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension());
			model.interaction().apply_vertex_on_the_left(v, M1);
			model.interaction().apply_vertex_on_the_right(v, M2);
			std::cerr << "vertex " << (M1-M2).norm() << std::endl;
			M = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension());
			slices[index+1].apply_matrix(M);
			std::cerr << "from right " << (M-slices[index+1].matrix()).norm() << std::endl;
			M = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension());
			slices[index+1].apply_on_the_right(M);
			std::cerr << "from right " << (M-slices[index+1].matrix()).norm() << std::endl;
		}

		void start () {
			for (size_t j=0;j<=M;j++) {
				compute_right_side(j);
				compute_left_side(M-j);
			}
		}

		double check_B_vs_last_right_side () {
			double ret = 0.0;
			size_t old_index = index;
			set_index(M-1);
			B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				slices[(i+index+1)%M].apply_matrix(B.U);
				decompose_U();
				fix_sign_B();
			}
			set_index(old_index);
			ret += (right_side[M].U-B.U).norm();
			ret += (right_side[M].Vt-B.Vt).norm();
			//std::cerr << (right_side[M-1].U-B.U) << std::endl << std::endl;
			//std::cerr << (right_side[M-1].Vt-B.Vt) << std::endl << std::endl;
			//if (ret>1.0e-6) throw -1;
			return ret;
		}

		//Eigen::FullPivLU<Eigen::MatrixXd> lu;
		Eigen::PartialPivLU<Eigen::MatrixXd> lu;

		void compute_all_propagators (const SVDHelper &left, const SVDHelper &right, Eigen::MatrixXd &ret, double zl = 1.0, double zr = 1.0) {
			size_t N = left.S.size();
			size_t M = (right.S.array().abs()*zr>1.0).count() + (left.S.array().abs()*zl>1.0).count();
			//Eigen::MatrixXd big_matrix = Eigen::MatrixXd::Zero(2*N, 2*N);
			big_matrix.resize(2*N, 2*N);
			big_matrix.topLeftCorner(N, N) = left.U.transpose() * right.Vt.transpose();
			big_matrix.bottomRightCorner(N, N) = -right.U.transpose() * left.Vt.transpose();
			big_matrix.bottomLeftCorner(N, N).setZero();
			big_matrix.bottomLeftCorner(N, N).diagonal() = zr*right.S;
			big_matrix.topRightCorner(N, N).setZero();
			big_matrix.topRightCorner(N, N).diagonal() = zl*left.S;
			const bool separate_scales = false; // use the Woodbury Matrix Identity for separating scales
			Eigen::MatrixXd U, V, C;
			if (separate_scales) {
				U = Eigen::MatrixXd::Zero(2*N, M);
				V = Eigen::MatrixXd::Zero(M, 2*N);
				C = Eigen::MatrixXd::Zero(M, M);
				int j = 0;
				for (size_t i=0;i<N;i++) {
					if (fabs(big_matrix.topRightCorner(N, N).diagonal()[i])>1.0) {
						C(j, j) = big_matrix.topRightCorner(N, N).diagonal()[i];
						big_matrix.topRightCorner(N, N).diagonal()[i] = 0.0;
						U(i, j) = 1.0;
						V(j, N+i) = 1.0;
						j++;
					}
					if (fabs(big_matrix.bottomLeftCorner(N, N).diagonal()[i])>1.0) {
						C(j, j) = big_matrix.bottomLeftCorner(N, N).diagonal()[i];
						big_matrix.bottomLeftCorner(N, N).diagonal()[i] = 0.0;
						U(N+i, j) = 1.0;
						V(j, i) = 1.0;
						j++;
					}
				}
			}
			lu.compute(big_matrix);
			big_matrix = lu.inverse();
			if (separate_scales) {
				C.diagonal() = C.diagonal().cwiseInverse();
				C += V*big_matrix*U;
				C = C.fullPivLu().inverse();
				big_matrix -= big_matrix*U*C*V*big_matrix;
			}
			big_matrix.topRows(N).applyOnTheLeft(right.Vt.transpose());
			big_matrix.bottomRows(N).applyOnTheLeft(-left.Vt.transpose());
			big_matrix.leftCols(N).applyOnTheRight(left.U.transpose());
			big_matrix.rightCols(N).applyOnTheRight(right.U.transpose());
			ret = Eigen::MatrixXd::Identity(2*N, 2*N) - big_matrix;
		}

		Eigen::MatrixXd full_propagator;

		Eigen::MatrixXd compute_propagators_2 () {
			size_t N = B.S.size();
			double zl = std::exp((M-index-1)*dtau*mu);
			double zr = std::exp((index+1)*dtau*mu);
			compute_all_propagators(left_side[index+1], right_side[index+1], full_propagator, zl, zr);
			//std::cerr << "--> " << (G_matrix-full_propagator.bottomRightCorner(N, N)).norm() << std::endl;
			G_matrix = full_propagator.bottomRightCorner(N, N);
			return full_propagator;
		}

		Eigen::MatrixXd compute_propagators_2_right () {
			size_t N = B.S.size();
			double zl = std::exp((M-index)*dtau*mu);
			double zr = std::exp((index)*dtau*mu);
			compute_all_propagators(left_side[index], right_side[index], full_propagator, zl, zr);
			//std::cerr << "--> " << (G_matrix-full_propagator.bottomRightCorner(N, N)).norm() << std::endl;
			G_matrix = full_propagator.bottomRightCorner(N, N);
			return full_propagator;
		}

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

		void insert_and_update (Vertex v, const MatrixType& matrixU, const MatrixType& matrixVt, const Eigen::Matrix2d &mat) {
			G_matrix -= (G_matrix * matrixU) * mat.inverse() * (matrixVt.transpose() * G_matrix);
			G_matrix += matrixU * (matrixVt.transpose() * G_matrix);
			B.U += matrixU * (matrixVt.transpose() * B.U);
			insert(v);
		}

		void insert_and_update (Vertex v) {
			if (v==cache.v) {
			} else {
				insert_probability(v);
			}
			insert_and_update(v, cache.u, cache.vt, cache.matrix);
		}

		size_t remove_and_update (Vertex v, const MatrixType& inverseU, const MatrixType& inverseVt, const Eigen::Matrix2d &mat) {
			G_matrix -= (G_matrix * inverseU) * mat.inverse() * (inverseVt.transpose() * G_matrix);
			G_matrix += inverseU * (inverseVt.transpose() * G_matrix);
			B.U += inverseU * (inverseVt.transpose() * B.U);
			return remove(v);
		}

		size_t remove_and_update (Vertex v) {
			if (v==cache.v) {
			} else {
				remove_probability(v);
			}
			return remove_and_update(v, cache.u, cache.vt, cache.matrix);
		}

		void insert_and_update_right (Vertex v, const MatrixType& matrixU, const MatrixType& matrixVt, const Eigen::Matrix2d &mat) {
			G_matrix -= (G_matrix * matrixU) * mat.inverse() * (matrixVt.transpose() * G_matrix);
			G_matrix += (G_matrix * matrixU) * matrixVt.transpose();
			B.U += matrixU * (matrixVt.transpose() * B.U);
			insert(v);
		}

		void insert_and_update_right (Vertex v) {
			if (v==cache.v) {
			} else {
				insert_probability_right(v);
			}
			insert_and_update_right(v, cache.u, cache.vt, cache.matrix);
		}

		size_t remove_and_update_right (Vertex v, const MatrixType& inverseU, const MatrixType& inverseVt, const Eigen::Matrix2d &mat) {
			G_matrix -= (G_matrix * inverseU) * mat.inverse() * (inverseVt.transpose() * G_matrix);
			G_matrix += (G_matrix * inverseU) * inverseVt.transpose();
			B.U += inverseU * (inverseVt.transpose() * B.U);
			return remove(v);
		}

		size_t remove_and_update_right (Vertex v) {
			if (v==cache.v) {
			} else {
				remove_probability_right(v);
			}
			return remove_and_update_right(v, cache.u, cache.vt, cache.matrix);
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

		void decompose_Vt () {
			for (size_t i=0;i<model.interaction().blocks();i++) {
				size_t a = model.interaction().block_start(i);
				size_t b = model.interaction().block_size(i);
				blocks[i].Vt = B.Vt.block(a, a, b, b);
				blocks[i].S = B.S.segment(a, b);
				blocks[i].U = B.U.block(a, a, b, b);
				blocks[i].absorbVt();
				//blocks[i].fullSVD(B.U.block(a, a, b, b) * B.S.segment(a, b).asDiagonal());
				B.U.block(a, a, b, b) = blocks[i].U;
				B.S.segment(a, b) = blocks[i].S;
				B.Vt.block(a, a, b, b) = blocks[i].Vt;
			}
		}

		void compute_B () {
			B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				slices[(i+index+1)%M].apply_matrix(B.U);
				decompose_U();
				fix_sign_B();
				// FIXME: have a random matrix applied here possibly only when no vertices have been applied
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
			G_matrix.resize(B.S.size(), B.S.size());
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
			//slices[index].matrixU_right(v, cache.u);
			//slices[index].matrixVt_right(v, cache.vt);
			cache.v = v;
			slices[index].matrixU(v, cache.u);
			slices[index].matrixVt(v, cache.vt);
			cache.matrix = Eigen::Matrix2d::Identity();
			cache.matrix.noalias() += cache.vt.transpose() * G_matrix * cache.u;
			cache.probability = cache.matrix.determinant();
			return cache.probability;
		}

		double remove_probability (Vertex v) {
			//slices[index].inverseU_right(v, cache.u);
			//slices[index].inverseVt_right(v, cache.vt);
			cache.v = v;
			slices[index].inverseU(v, cache.u);
			slices[index].inverseVt(v, cache.vt);
			cache.matrix = Eigen::Matrix2d::Identity();
			cache.matrix.noalias() += cache.vt.transpose() * G_matrix * cache.u;
			cache.probability = cache.matrix.determinant();
			return cache.probability;
		}

		double insert_probability_right (Vertex v) {
			//slices[index].matrixU_right(v, cache.u);
			//slices[index].matrixVt_right(v, cache.vt);
			cache.v = v;
			slices[index].matrixU_right(v, cache.u);
			slices[index].matrixVt_right(v, cache.vt);
			cache.matrix = Eigen::Matrix2d::Identity();
			cache.matrix.noalias() += cache.vt.transpose() * G_matrix * cache.u;
			cache.probability = cache.matrix.determinant();
			return cache.probability;
		}

		double remove_probability_right (Vertex v) {
			//slices[index].inverseU_right(v, cache.u);
			//slices[index].inverseVt_right(v, cache.vt);
			cache.v = v;
			slices[index].inverseU_right(v, cache.u);
			slices[index].inverseVt_right(v, cache.vt);
			cache.matrix = Eigen::Matrix2d::Identity();
			cache.matrix.noalias() += cache.vt.transpose() * G_matrix * cache.u;
			cache.probability = cache.matrix.determinant();
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
			//fix_sign_B();
			//std::cerr << B.S.array().abs().log().sum() << std::endl;
                        //std::cerr << t_U << std::endl << std::endl;
                        //std::cerr << std::endl;
                        //std::cerr << t_U-B.U << std::endl;
			ret += (t_U-B.U).norm();
			ret += (t_Vt-B.Vt).norm();
			//if (ret>1.0e-6) throw -1;
			return ret;
		}

		const Eigen::MatrixXd & green_function () const {
			return G_matrix;
		}

		void gf_tau (Eigen::MatrixXd &gf, double t) {
			size_t N = B.S.size();
			gf = full_propagator.bottomLeftCorner(N, N);
			slices[(index+1)%M].apply_matrix(gf, t);
		}

		double slice_start () const { return dtau*index; }
		double slice_end () const { return dtau*(index+1); }
		size_t slice_size () const { return slices[index].size(); }
		size_t slice_number () const { return M; } // MUST be same as slices.size()

		size_t current_slice () const { return index; }

		sweep_direction_type sweep_direction () const { return sweep_direction_; }
		void set_sweep_direction (sweep_direction_type d) { sweep_direction_ = d; }

		Vertex get_vertex (size_t i) const { return slices[index].get_vertex(i); }
		Vertex generate_vertex (std::mt19937_64 &generator) { return model.interaction().generate(0.0, slice_end()-slice_start(), generator); }

		double insert_factor () { return +log(beta/slice_number()) -log(slice_size()+1) +model.interaction().combinatorial_factor(); }
		double remove_factor () { return -log(beta/slice_number()) +log(slice_size()+0) -model.interaction().combinatorial_factor(); }

		size_t size () const { size_t ret = 0; for (const auto &s : slices) ret += s.size(); return ret; }
		size_t vertices () const { return size(); }
		void show_verts () const { for (const auto &s : slices) std::cerr << s.size() << std::endl; }
		void advance (int n) { set_index(2*M+index+n); }

		size_t volume () const { return model.lattice().volume(); }
		double inverse_temperature () const { return beta; }

		double kinetic_energy (const Eigen::MatrixXd& cache) const {
			return model.lattice().kinetic_energy(cache);
		}

		double interaction_energy (const Eigen::MatrixXd& cache) const {
			return model.interaction().interaction_energy(cache);
		}

		void check_all_det (int block) {
			size_t a = model.interaction().block_start(block);
			size_t b = model.interaction().block_size(block);
			for (size_t i=0;i<M;i++) {
				double A = 0.0, B = 0.0;
				for (size_t j=0;j<M;j++) {
					if (j<=i) A += slices[j].log_abs_det_block(block);
					else B += slices[j].log_abs_det_block(block);
				}
				std::cerr << i << ") "<< A << ' ' << B << " <=> " << left_side[i+1].S.segment(a, b).array().abs().log().sum() << ' ' << right_side[i+1].S.segment(a, b).array().abs().log().sum() << std::endl;
			}
		}

		void check_first_slice () {
			Eigen::MatrixXd A = slices[0].matrix();
			std::cerr << " * " << (A-right_side[1].matrix()).norm() << std::endl;
		}

		void check_all_prop () {
			Eigen::MatrixXd G;
			for (size_t i=0;i<slice_number();i++) {
				set_index(i);
				compute_propagators_2();
				G = green_function();
				//std::cerr << G << std::endl << std::endl;
				compute_B();
				compute_G();
				save_G();  
				std::cerr << "check propagators: " << (double(i)/slice_number()) << ' '
					<< (green_function()-G).norm() << ' '
					<< (green_function()-G).cwiseAbs().maxCoeff() << std::endl;
			}                                       
		}
};

#endif // CONFIGURATION_HPP

