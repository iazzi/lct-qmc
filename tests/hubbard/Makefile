CXXFLAGS=$(MYCXXFLAGS) -std=c++11 -I $(HOME)/local/include `pkg-config --cflags eigen3 ` -Wall -I ../../
LDFLAGS=$(MYLDFLAGS) -L $(HOME)/local/lib `pkg-config --libs eigen3`
LDLIBS=$(MYLDLIBS) -lgmp -lmpfr `pkg-config --libs eigen3` -lm -lstdc++ -lmkl_gf_lp64 -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_core -llua -pthread -lfftw3_threads -lfftw3 -lmpi

all: hubbard1_test hubbard2_test

hubbard1_test: hubbard1
	./hubbard1

hubbard2_test: hubbard2
	./hubbard2

hubbard1: hubbard1.o ../../hubbard.o

hubbard2: hubbard2.o ../../hubbard.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp $(MYCXXFLAGS)" MYLDFLAGS="-fopenmp -lfftw3_threads"

mkl:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL $(MYCXXFLAGS)" MYLDFLAGS=""

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG $(MYCXXFLAGS)" MYLDFLAGS=""

debug:
	$(MAKE) all MYCXXFLAGS="-g -ggdb -O0" MYLDFLAGS="-g -ggdb -O0"


