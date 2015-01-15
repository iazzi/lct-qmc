include ../../Makefile.conf
CXXFLAGS=$(MYCXXFLAGS) -std=c++11 `pkg-config --cflags eigen3 ` -Wall -I ../../
LDFLAGS=$(MYLDFLAGS) `pkg-config --libs eigen3`
LDLIBS=$(MYLDLIBS) `pkg-config --libs eigen3`

BIN=hubbard1 hubbard2 slice determinant insert remove insert_and_remove wrap metropolis sweep kinetic

all: ${BIN}

hubbard1: hubbard1.o

hubbard2: hubbard2.o

slice: slice.o

determinant: determinant.o

insert: insert.o

remove: remove.o

insert_and_remove: insert_and_remove.o

wrap: wrap.o

metropolis: metropolis.o

sweep: sweep.o

kinetic: kinetic.o

parallel:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -fopenmp $(MYCXXFLAGS)" MYLDFLAGS="-fopenmp -lfftw3_threads $(MYLDFLAGS=)" MYLDLIBS="$(MYLDLIBS)"

mkl:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_USE_MKL_ALL $(MYCXXFLAGS)" MYLDFLAGS="$(MYLDFLAGS=)" MYLDLIBS="$(MYLDLIBS)"

optimized:
	$(MAKE) all MYCXXFLAGS="-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG $(MYCXXFLAGS)" MYLDFLAGS="$(MYLDFLAGS)" MYLDLIBS="$(MYLDLIBS)"

debug:
	$(MAKE) all MYCXXFLAGS="-g -ggdb -O0 $(MYCXXFLAGS)" MYLDFLAGS="-g -ggdb -O0 $(MYLDFLAGS)" MYLDLIBS="$(MYLDLIBS)"

clean:
	rm -f ${BIN} *.o
