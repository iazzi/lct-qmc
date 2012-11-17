CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/local/include `pkg-config --cflags python-2.7 eigen3 ompi-cxx` -I /usr/lib/python2.7/site-packages/numpy/core/include
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib -lboost_system -lboost_python `pkg-config --libs python-2.7 eigen3 ompi-cxx` -lalps -lm -lstdc++ -llapack

all: main perturbative full1d full2d full2x2 exact2x2 continuoustime full ctsingle

main: main.o

perturbative: perturbative.o

full: full.o

full1d: full1d.o

full2d: full2d.o

full2x2: full2x2.o

continuoustime: continuoustime.o

ctsingle: ctsingle.o

continuoustime.o: continuoustime.cpp ct_aux.hpp

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DEIGEN_NO_DEBUG -fopenmp" MYLDFLAGS="-fopenmp -lfftw3 -lfftw3_threads"

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DEIGEN_NO_DEBUG" MYLDFLAGS="-lfftw3"

debug:
	$(MAKE) all MYCXXFLAGS="-ggdb" MYLDFLAGS="-lfftw3 -lfftw3_threads"


