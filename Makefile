CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/local/include `pkg-config --cflags python-2.7 eigen3 ompi-cxx` -I /usr/lib/python2.7/site-packages/numpy/core/include -Wall
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib -lboost_system -lboost_python `pkg-config --libs python-2.7 eigen3 ompi-cxx` -lalps -lm -lstdc++ -llapack -llua

all: full

main: main.o

perturbative: perturbative.o

test.o: measurements.hpp

test: test.o

full: full.o helpers.o

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
	$(MAKE) all MYCXXFLAGS="-g" MYLDFLAGS="-lfftw3"


