CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std=c++11 -O0 -I${OPENBLAS_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas

OBJS=cg_main_sparse.o cg.o matrix_coo.o mmio.o

all: cgsolver_sparse

cgsolver_sparse: $(OBJS) matrix_coo_gpu.o
	$(LD) -o $@ $(OBJS) matrix_coo_gpu.o $(LDFLAGS)

matrix_coo_gpu.o: matrix_coo_gpu.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@
clean:
	rm -f cgsolver_sparse *.o *~
