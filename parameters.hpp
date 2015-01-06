#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>
#include <map>
#include <cstdlib>

#include <iostream>

class Parameters {
	std::map<std::string, std::string> params;

	public:

	Parameters (int argc, char **argv) {
		bool argument = false;
		std::string k;
		for (int i=1;i<argc;i++) {
			if (argv[i][0]=='-' && argv[i][1]=='-') {
				if (argument) {
					params[k] = "1";
				} else {
				}
				k = (std::string(&argv[i][2]));
				argument = true;
			} else if (argument) {
				params[k] = argv[i];
				argument = false;
			}
		}
		if (argument) {
			params[k] = "1";
		}
	}

	std::string getString (const std::string &k) const { if (params.find(k)!=params.end()) return params.at(k); else return std::string(); }
	bool contains (const std::string &k) const { return params.find(k)!=params.end(); }
	double getNumber (const std::string &k) const { return atof(getString(k).c_str()); }
	int getInteger (const std::string &k) const { return atoi(getString(k).c_str()); }
	void list () const { for (auto x : params) std::cerr << x.first << " --> " << x.second << std::endl; }
};

#endif // PARAMETERS_HPP
