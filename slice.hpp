#ifndef SLICE_HPP
#define SLICE_HPP

#include <set>

template <typename Interaction>
class Slice {
	public:
	typedef typename Interaction::Vertex Vertex;
	private:
	Interaction &I;
	std::set<Vertex, typename Vertex::Compare> verts;
	public:
	Slice (Interaction &i) : I(i) {}
};


#endif // SLICE_HPP

