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

	bool contains (const std::string &k) const { return params.find(k)!=params.end(); }
	std::string getString (const std::string &k, const std::string &def = std::string()) const { if (contains(k)) return params.at(k); else return def; }
	double getNumber (const std::string &k, double def = 0.0) const { if (contains(k)) return atof(params.at(k).c_str()); else return def; }
	int getInteger (const std::string &k, int def = 0) const { if (contains(k)) return atoi(params.at(k).c_str()); else return def; }
	void list () const { for (auto x : params) std::cerr << x.first << " --> " << x.second << std::endl; }
};

#endif // PARAMETERS_HPP
