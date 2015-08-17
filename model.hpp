#ifndef MODEL_HPP
#define MODEL_HPP

template <class L, class I> 
class Model {
	L &l;
	I &i;
	public:
	Model (L &a, I &b) : l(a), i(b) { i.set_lattice_eigenvectors(l.eigenvectors()); }
	L& lattice () { return l; }
	const L& lattice () const { return l; }
	I& interaction () { return i; }
	const I& interaction () const { return i; }
	typedef L Lattice;
	typedef I Interaction;
};

template <class L, class I> 
Model<L, I> make_model (L &a, I &b) {
	return Model<L, I>(a, b);
}

#endif // MODEL_HPP

