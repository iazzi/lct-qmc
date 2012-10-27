CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/physics/Eigen -I $(HOME)/local/include -I /usr/include/python2.7 -I /usr/lib/python2.7/site-packages/numpy/core/include
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib -lboost_system -lboost_python -lpython2.7 -lalps -lm -lstdc++

all: main perturbative full1d full2d full2x2 exact2x2

main: main.o

perturbative: perturbative.o

full1d: full1d.o

full2d: full2d.o

full2x2: full2x2.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -msse4 -msse4.1 -DEIGEN_NO_DEBUG -fopenmp" MYLDFLAGS="-fopenmp -lfftw3 -lfftw3_threads"

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -msse4 -msse4.1 -DEIGEN_NO_DEBUG" MYLDFLAGS="-lfftw3"

debug:
	$(MAKE) all MYCXXFLAGS="-ggdb" MYLDFLAGS="-lfftw3 -lfftw3_threads"


