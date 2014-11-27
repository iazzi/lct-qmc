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

		double beta, mu;
		double dtau;
		size_t M;

		SVDHelper B_up, B_dn; // this holds the deomposition of the matrix B for the up species
		SVDHelper G_up, G_dn;

		Eigen::MatrixXd G_matrix_up;
		Eigen::MatrixXd G_matrix_dn;

		size_t index; // This is the index of the LAST SLICE IN B

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
			return B_up.S.array().abs().log().sum() + B_dn.S.array().abs().log().sum();
		}

		double log_abs_det_up () {
			return B_up.S.array().abs().log().sum();
		}

		double slice_log_abs_det () {
			double ret = 0.0;
			for (size_t i=0;i<M;i++) {
				ret += slices[i].log_abs_det();
			}
			return ret;
		}

		size_t slice_size () const { return slices[index].size(); }

		void insert (Vertex v) {
			if (v.tau<beta) {
				size_t i = v.tau/dtau;
				v.tau -= i*dtau;
				slices[i].insert(v);
			}
		}

		void insert_and_update (Vertex v) {
			if (v.tau<beta) {
				size_t i = v.tau/dtau;
				v.tau -= i*dtau;
				G_matrix_up -= v.sigma * (G_matrix_up * slices[index].matrixU(v)) * (slices[index].matrixVt(v).transpose() * G_matrix_up) / (1.0 + v.sigma * slices[index].matrixVt(v).transpose() * G_matrix_up * slices[index].matrixU(v));
				G_matrix_up += v.sigma * slices[i].matrixU(v) * (slices[i].matrixVt(v).transpose() * G_matrix_up);
				G_matrix_dn -= -v.sigma/(1.0+v.sigma) * (G_matrix_dn * slices[index].matrixU2(v)) * (slices[index].matrixVt2(v).transpose() * G_matrix_dn) / (1.0 + -v.sigma/(1.0+v.sigma) * slices[index].matrixVt2(v).transpose() * G_dn.matrix() * slices[index].matrixU2(v));
				G_matrix_dn += -v.sigma/(1.0+v.sigma) * slices[i].matrixU2(v) * (slices[i].matrixVt2(v).transpose() * G_matrix_dn);
				slices[i].insert(v);
			}
		}

		void compute_B () {
			B_up.setIdentity(model.lattice().volume()); // FIXME: maybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				slices[(i+index+1)%M].apply_matrix_11(B_up.U);
				B_up.absorbU(); // FIXME: have a random matrix applied here possibly only when no vertices have been applied
			}
			B_dn.setIdentity(model.lattice().volume()); // FIXME: maybe have a direct reference to the lattice here too
			for (size_t i=0;i<M;i++) {
				slices[(i+index+1)%M].apply_matrix_12(B_dn.U);
				B_dn.absorbU(); // FIXME: have a random matrix applied here possibly only when no vertices have been applied
			}
		}

		void compute_G () {
			G_up = B_up; // B
			G_up.invertInPlace(); // B^-1
			G_up.add_identity(std::exp(-beta*mu)); // 1+exp(-beta*mu)*B^-1
			G_up.invertInPlace(); // 1/(1+exp(-beta*mu)*B^-1) = B/(1+B)
			G_dn = B_dn;
			G_dn.invertInPlace();
			G_dn.add_identity(std::exp(-beta*mu));
			G_dn.invertInPlace();
		}

		std::pair<double, double> probability () {
			SVDHelper A_up, A_dn;
			A_up = B_up;
			A_dn = B_dn;
			A_up.add_identity(exp(beta*mu));
			A_dn.add_identity(exp(beta*mu));
			std::pair<double, double> ret;
			ret.first = A_up.S.array().log().sum() + A_dn.S.array().log().sum();
			ret.second = (A_up.U*A_up.Vt*A_dn.U*A_dn.Vt).determinant()>0.0?1.0:-1.0;
			return ret;
		}

		double probability_ratio (Vertex v) { //FIXME it will only work for rank-1 vertices
			size_t i = v.tau/dtau;
			v.tau -= i*dtau; // FIXME we should not have to modify the vertex time here;
			double ret = 1.0 + v.sigma * slices[index].matrixVt(v).transpose() * G_matrix_up * slices[index].matrixU(v);
			      ret *= 1.0 - v.sigma/(1.0+v.sigma) * slices[index].matrixVt2(v).transpose() * G_matrix_dn * slices[index].matrixU2(v);
			return ret;
		}

		void save_G () {
			G_matrix_up = G_up.matrix();
			G_matrix_dn = G_dn.matrix();
		}

		double check_and_save_G () {
			double ret = 0.0;
			Eigen::MatrixXd A;
			A = G_up.matrix();
			//std::cerr << A.array()-cache.array() << std::endl << std::endl;
			//std::cerr << cache.col(0).normalized().transpose() << std::endl << std::endl;
			//std::cerr << (G_matrix_up-A).col(0).normalized().transpose() << std::endl << std::endl;
			//Eigen::VectorXd B = (G_matrix_up-A).col(0).normalized();
			//Eigen::JacobiSVD<Eigen::MatrixXd> svd(G_matrix_up, Eigen::ComputeThinU | Eigen::ComputeThinV);
			//std::cerr << (svd.solve(B)).transpose().normalized() << std::endl << std::endl;
			ret += (G_matrix_up-A).norm();
			G_matrix_up.swap(A);
			A = G_dn.matrix();
			ret += (G_matrix_dn-A).norm();
			G_matrix_dn.swap(A);
			return ret;
		}

		double slice_start () const { return dtau*index; }
		double slice_end () const { return dtau*(index+1); }
};

#endif // CONFIGURATION_HPP

