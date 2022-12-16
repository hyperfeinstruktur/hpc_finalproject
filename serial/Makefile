CXX=g++
LD=${CXX}
#CXXFLAGS+=-Wall -Wextra -Werror -pedantic -std=c++11 -I${OPENBLAS_ROOT}/include -pg -g -O1
CXXFLAGS+=-Wall -Wextra -Werror -pedantic -std=c++11 -pg -g -O0
#LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lcblas

OBJS=cg.o matrix.o matrix_coo.o mmio.o

all: cgsolver_dense cgsolver_sparse

cgsolver_dense: cg_main_dense.o $(OBJS)
	$(LD) -o $@ cg_main_dense.o $(OBJS) $(LDFLAGS)
cgsolver_sparse: cg_main_sparse.o $(OBJS)
	$(LD) -o $@ cg_main_sparse.o $(OBJS) $(LDFLAGS)

clean:
	rm -f cgsolver_sparse cgsolver_dense *.o *~
