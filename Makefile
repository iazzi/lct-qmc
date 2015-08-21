include Makefile.conf
CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -Wall
LDFLAGS=$(MYLDFLAGS) 
LDLIBS=$(MYLDLIBS)  -llua

all: generic

process_gf: process_gf.o

main: main.o simulation.o mpfr.o

v3ct: v3ct.o

pqmc: pqmc.o

lct: lct.o

generic: generic.o

lct.o: lct.cpp svd.hpp accumulator.hpp measurements.hpp hubbard.hpp slice.hpp cubiclattice.hpp model.hpp configuration.hpp

simulation.o: simulation.cpp simulation.hpp

main.o: main.cpp simulation.hpp

test_params: test_params.o simulation.o mpfr.o

setup_batch.o: setup_batch.cpp simulation.hpp

setup_batch: setup_batch.o simulation.o mpfr.o

zerotemp: zerotemp.o zerotemperature.hpp

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp $(MYCXXFLAGS)" MYLDFLAGS="-fopenmp -lfftw3_threads"

mkl:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL $(MYCXXFLAGS)" MYLDFLAGS="$(MYLDFLAGS)" MYLDLIBS="$(MYLDLIBS)"

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL $(MYCXXFLAGS)" MYLDFLAGS="$(MYLDFLAGS)" MYLDLIBS="$(MYLDLIBS)"

debug:
	$(MAKE) all MYCXXFLAGS="-g -ggdb -O0 $(MYCXXFLAGS)" MYLDFLAGS="-g -ggdb -O0 $(MYLDFLAGS)" MYLDLIBS="$(MYLDLIBS)"


