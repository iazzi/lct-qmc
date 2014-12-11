include ../../Makefile.conf
CXXFLAGS=$(MYCXXFLAGS) -std=c++11 `pkg-config --cflags eigen3 ` -Wall -I ../../
LDFLAGS=$(MYLDFLAGS) `pkg-config --libs eigen3`
LDLIBS=$(MYLDLIBS) `pkg-config --libs eigen3`

all: hubbard1.test hubbard2.test slice.test probability.test greenfunction.test

%.test: %
	./$<

hubbard1: hubbard1.o ../../hubbard.o

hubbard2: hubbard2.o ../../hubbard.o

slice: slice.o ../../hubbard.o

probability: probability.o ../../hubbard.o

greenfunction: greenfunction.o ../../hubbard.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp $(MYCXXFLAGS)" MYLDFLAGS="-fopenmp -lfftw3_threads"

mkl:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL $(MYCXXFLAGS)" MYLDFLAGS=""

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG $(MYCXXFLAGS)" MYLDFLAGS=""

debug:
	$(MAKE) all MYCXXFLAGS="-g -ggdb -O0" MYLDFLAGS="-g -ggdb -O0"


