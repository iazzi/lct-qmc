#ifndef MODEL_HPP
#define MODEL_HPP

#include "parameters.hpp"

template <class L, class I> 
class Model {
	L l;
	I i;
	public:
	Model (const Model &m) : l(m.l), i(m.i) { i.set_lattice_eigenvectors(l.eigenvectors()); i.set_lattice_eigenvalues(l.eigenvalues()); }
	Model (const L &a, const I &b) : l(a), i(b) { i.set_lattice_eigenvectors(l.eigenvectors()); i.set_lattice_eigenvalues(l.eigenvalues()); }
	Model (const Parameters &params) : l(params), i(params) { i.set_lattice_eigenvectors(l.eigenvectors()); i.set_lattice_eigenvalues(l.eigenvalues()); }
	I& interaction () { return i; }
	const I& interaction () const { return i; }
	typedef I Interaction;
	template <typename T> void propagate (double t, const T &M) { l.propagate(t, M); }
	template <typename T> void propagate_right (double t, const T &M) { l.propagate_right(t, M); }
};

template <class L, class I> 
Model<L, I> make_model (L &a, I &b) {
	return Model<L, I>(a, b);
}

#endif // MODEL_HPP

