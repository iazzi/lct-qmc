CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/local/include `pkg-config --cflags eigen3 ` -Wall
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib `pkg-config --libs eigen3`
LDLIBS=$(MYLDLIBS) -lgmp -lmpfr `pkg-config --libs eigen3` -lm -lstdc++ -lmkl_gf_lp64 -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_core -llua -pthread -lfftw3_threads -lfftw3 -lmpi

all: full unstable main test_params setup_batch process_gf

process_gf: process_gf.o

main: main.o simulation.o mpfr.o

simulation.o: simulation.cpp simulation.hpp

main.o: main.cpp simulation.hpp

test_params: test_params.o simulation.o mpfr.o

setup_batch.o: setup_batch.cpp simulation.hpp

setup_batch: setup_batch.o simulation.o mpfr.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp $(MYCXXFLAGS)" MYLDFLAGS="-fopenmp -lfftw3_threads"

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG $(MYCXXFLAGS)" MYLDFLAGS=""

debug:
	$(MAKE) all MYCXXFLAGS="-g" MYLDFLAGS=""


