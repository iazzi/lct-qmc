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
		SVDHelper G; // SVD decomposition of the Green function

		Eigen::MatrixXd G_matrix;

		size_t index; // This is the index of the LAST SLICE IN B

		Eigen::MatrixXd R; // a random matrix to solve degeneracies
		Eigen::MatrixXd R2; // inverse of R
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

		double log_abs_max () const {
			return B.S.array().abs().log().abs().maxCoeff();
		}

		void insert (Vertex v) {
			slices[index].insert(v);
		}

		size_t remove (Vertex v) {
			slices[index].matrix() += slices[index].inverseU(v)*slices[index].inverseVt(v).transpose()*slices[index].matrix();
			return slices[index].remove(v);
		}

		void insert_and_update (Vertex v) {
			size_t i = index;
			G_matrix -= (G_matrix * slices[index].matrixU(v)) * (Eigen::Matrix2d::Identity() + slices[index].matrixVt(v).transpose() * G_matrix * slices[index].matrixU(v)).inverse() * (slices[index].matrixVt(v).transpose() * G_matrix);
			G_matrix += slices[i].matrixU(v) * (slices[i].matrixVt(v).transpose() * G_matrix);
			insert(v);
		}

		size_t remove_and_update (Vertex v) {
				size_t i = index;
				G_matrix -= (G_matrix * slices[index].inverseU(v)) * (Eigen::Matrix2d::Identity() + slices[index].inverseVt(v).transpose() * G_matrix * slices[index].inverseU(v)).inverse() * (slices[index].inverseVt(v).transpose() * G_matrix);
				G_matrix += slices[i].inverseU(v) * (slices[i].inverseVt(v).transpose() * G_matrix);
				return slices[index].remove(v);
		}

		void compute_B () {
			B.setIdentity(model.lattice().dimension()); // FIXME: maybe have a direct reference to the lattice here too
			R = Eigen::MatrixXd::Identity(model.lattice().dimension(), model.lattice().dimension()) + 0.002*Eigen::MatrixXd::Random(model.lattice().dimension(), model.lattice().dimension());
			R2 = R.inverse();
			for (size_t i=0;i<M;i++) {
				slices[(i+index+1)%M].apply_matrix(B.U);
				B.U.applyOnTheLeft(R);
				B.absorbU(); // FIXME: have a random matrix applied here possibly only when no vertices have been applied
				B.U.applyOnTheLeft(R2);
			}
			B.absorbU(); // FIXME: only apply this if the random matrix is used in the last step
		}
		void compute_G () {
			G = B; // B
			G.invertInPlace(); // B^-1
			G.add_identity(std::exp(-beta*mu)); // 1+exp(-beta*mu)*B^-1
			G.invertInPlace(); // 1/(1+exp(-beta*mu)*B^-1) = B/(1+B)
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
			return (Eigen::Matrix2d::Identity() + slices[index].matrixVt(v).transpose() * G_matrix * slices[index].matrixU(v)).determinant();
		}

		double remove_probability (Vertex v) {
			return (Eigen::Matrix2d::Identity() + slices[index].inverseVt(v).transpose() * G_matrix * slices[index].inverseU(v)).determinant();
		}

		void save_G () {
			G_matrix = G.matrix();
		}

		double check_and_save_G () {
			double ret = 0.0;
			Eigen::MatrixXd A;
			A = G.matrix();
			//std::cerr << A.array()-cache.array() << std::endl << std::endl;
			//std::cerr << cache.col(0).normalized().transpose() << std::endl << std::endl;
			//std::cerr << (G_matrix_up-A).col(0).normalized().transpose() << std::endl << std::endl;
			//Eigen::VectorXd B = (G_matrix_up-A).col(0).normalized();
			//Eigen::JacobiSVD<Eigen::MatrixXd> svd(G_matrix_up, Eigen::ComputeThinU | Eigen::ComputeThinV);
			//std::cerr << (svd.solve(B)).transpose().normalized() << std::endl << std::endl;
			ret += (G_matrix-A).norm();
			G_matrix.swap(A);
			return ret;
		}

		double slice_start () const { return dtau*index; }
		double slice_end () const { return dtau*(index+1); }
		size_t slice_size () const { return slices[index].size(); }
		size_t slice_number () const { return M; } // MUST be same as slices.size()

		Vertex get_vertex (size_t i) const { return slices[index].get_vertex(i); }
};

#endif // CONFIGURATION_HPP

