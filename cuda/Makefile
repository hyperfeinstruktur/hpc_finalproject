CXX=nvcc
LD=${CXX}
#CXXFLAGS+=-Wall -Wextra -Werror -pedantic -std=c++11 -I${OPENBLAS_ROOT}/include -pg -g -O1
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std=c++11 -O1
#LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lcblas -lcublas

OBJS=cg_main_sparse.o cg.o matrix_coo.o mmio.o

all: cgsolver_sparse

cgsolver_sparse: $(OBJS) matrix_coo_gpu.o
	$(LD) -o $@ $(OBJS) matrix_coo_gpu.o $(LDFLAGS)

matrix_coo_gpu.o: matrix_coo_gpu.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@
clean:
	rm -f cgsolver_sparse *.o *~