#ifndef MODEL_HPP
#define MODEL_HPP

#include "parameters.hpp"

template <class L, class I> 
class Model {
	I i;
	public:
	//Model (const Model &m) : l(m.l), i(m.i) { i.set_lattice_eigenvectors(l.eigenvectors()); i.set_lattice_eigenvalues(l.eigenvalues()); }
	//Model (const L &a, const I &b) : i(b) { i.set_lattice_eigenvectors(a.eigenvectors()); i.set_lattice_eigenvalues(a.eigenvalues()); }
	Model (const Parameters &params) : i(params) {}
	I& interaction () { return i; }
	const I& interaction () const { return i; }
	typedef I Interaction;
};

#endif // MODEL_HPP

