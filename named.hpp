#ifndef NAMED_HPP
#define NAMED_HPP

#include <iostream>

#define named_type(t,s) \
		  struct s##_t { \
		    t s; \
		    t value () const { return s; } \
		    typename std::remove_reference<t>::type& ref () { return s; } \
		    const char *name () const { return #s; } \
		    static t value (const s##_t& x) { return x.s; } \
		    static typename std::remove_reference<t>::type& ref (s##_t& x) { return x.s; } \
		    static const char *name (const s##_t& x) { return #s; } \
		  }

#define named_value(x,s,n) ([](){ \
		  struct T { \
		    decltype(x) s; \
		    decltype(x) value () const { return s; } \
		    typename std::remove_reference<decltype(x)>::type& ref () { return s; } \
		    const char *name () const { return n; } \
		    static decltype(x) value (const T& t) { return t.s; } \
		    static typename std::remove_reference<decltype(x)>::type& ref (T& t) { return t.s; } \
		    static const char *name (const T& t) { return n; } \
		    T () : s(x) {} \
		  }; \
		  T ret; \
		  return ret; \
		}())

template <class ... Args>
struct named_tuple : public Args... {
	named_tuple () {}
	named_tuple (const Args&... args) : Args(args)... {}
	void stream (std::ostream& out) const {
		out << "{ ";
		int dummy[sizeof...(Args)] = { (out << Args::name() << " = " << Args::value() << "; ", 0)... };
		out << "}";
	}
};

template <>
struct named_tuple<> {
	named_tuple () {}
	void stream (std::ostream& out) const {
		out << "{ ";
		out << "}";
	}
};


template <class ... Args>
named_tuple<Args...> make_named_tuple (const Args&... args) {
	return named_tuple<Args...>(args...);
}

template <class ... Args>
std::ostream& operator<< (std::ostream& out, const named_tuple<Args...>& t) {
	t.stream(out);
	return out;
}

#endif // NAMED_HPP

