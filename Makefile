CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/local/include `pkg-config --cflags eigen3 ` -Wall
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib `pkg-config --libs eigen3`
LDLIBS=$(MYLDLIBS) -lgmp -lmpfr `pkg-config --libs eigen3` -lm -lstdc++ -lmkl_gf_lp64 -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_core -llua -pthread -lfftw3_threads -lfftw3 -lmpi

all: main test_params setup_batch process_gf ct_main v3ct pqmc lct

process_gf: process_gf.o

main: main.o simulation.o mpfr.o

ct_main: ct_main.o ct_simulation.o

v3ct: v3ct.o

pqmc: pqmc.o

lct: lct.o hubbard.o

lct.o: lct.cpp svd.hpp accumulator.hpp measurements.hpp hubbard.hpp slice.hpp cubiclattice.hpp model.hpp configuration.hpp

simulation.o: simulation.cpp simulation.hpp

ct_simulation.o: ct_simulation.cpp ct_simulation.hpp

main.o: main.cpp simulation.hpp

ct_main.o: ct_main.cpp ct_simulation.hpp

test_params: test_params.o simulation.o mpfr.o

setup_batch.o: setup_batch.cpp simulation.hpp

setup_batch: setup_batch.o simulation.o mpfr.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp $(MYCXXFLAGS)" MYLDFLAGS="-fopenmp -lfftw3_threads"

mkl:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL $(MYCXXFLAGS)" MYLDFLAGS=""

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG $(MYCXXFLAGS)" MYLDFLAGS="$(MYLDFLAGS)" MYLDLIBS="$(MYLDLIBS)"

debug:
	$(MAKE) all MYCXXFLAGS="-g -ggdb -O0" MYLDFLAGS="-g -ggdb -O0"


