CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/local/include `pkg-config --cflags eigen3 ` -Wall
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib `pkg-config --libs eigen3`
LDLIBS=$(MYLDLIBS) -lgmp -lmpfr `pkg-config --libs eigen3` -lm -lstdc++ -lmkl_gf_lp64 -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_core -llua -pthread -lfftw3_threads -lfftw3

all: full unstable sign stable stablefast test_params ct anneal_wf

main: main.o

ct: ct.o

perturbative: perturbative.o

test.o: measurements.hpp

test: test.o

full: full.o helpers.o

unstable: unstable.o

full1d: full1d.o

full2d: full2d.o

full2x2: full2x2.o

continuoustime: continuoustime.o

ctsingle: ctsingle.o

continuoustime.o: continuoustime.cpp ct_aux.hpp

stablefast: stablefast.o simulation.o mpfr.o

simulation.o: simulation.cpp simulation.hpp

test_params: test_params.o simulation.o mpfr.o

mpfr_test: mpfr_test.o mpfr.o

anneal_wf: anneal_wf.o simulation.o mpfr.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp" MYLDFLAGS="-fopenmp -lfftw3_threads"

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG" MYLDFLAGS=""

debug:
	$(MAKE) all MYCXXFLAGS="-g" MYLDFLAGS=""


