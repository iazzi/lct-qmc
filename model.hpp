#ifndef MODEL_HPP
#define MODEL_HPP

#include "parameters.hpp"

template <class L, class I> 
class Model {
	I i;
	public:
	Model (const Model &m) : i(m.i) { i.set_lattice_eigenvectors(i.eigenvectors()); i.set_lattice_eigenvalues(i.eigenvalues()); }
	Model (const L&a, const I &b) : i(b) { i.set_lattice_eigenvectors(b.eigenvectors()); i.set_lattice_eigenvalues(b.eigenvalues()); }
	Model (const I &b) : i(b) { i.set_lattice_eigenvectors(i.eigenvectors()); i.set_lattice_eigenvalues(i.eigenvalues()); }
	Model (const Parameters &params) : i(params) { L l(params); i.set_lattice_eigenvectors(l.eigenvectors()); i.set_lattice_eigenvalues(l.eigenvalues()); }
	I& interaction () { return i; }
	const I& interaction () const { return i; }
	typedef I Interaction;
	template <typename T> void propagate (double t, const T &M) { i.propagate(t, M); }
	template <typename T> void propagate_right (double t, const T &M) { i.propagate_right(t, M); }
};

template <class L, class I> 
Model<L, I> make_model (L &a, I &b) {
	return Model<L, I>(a, b);
}

#endif // MODEL_HPP

