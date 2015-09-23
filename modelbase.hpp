#ifndef MODELBASE_HPP
#define MODELBASE_HPP

class ModelBase {
	protected:
	Eigen::MatrixXd H;
	Eigen::VectorXd eigenvalues_;
	Eigen::MatrixXd eigenvectors_;

	public:
	Eigen::VectorXd eigenvalues () const { return eigenvalues_; }
	Eigen::MatrixXd eigenvectors () const { return eigenvectors_; }
	Eigen::MatrixXd hamiltonian () const { return H; }
};

#endif // MODELBASE_HPP

