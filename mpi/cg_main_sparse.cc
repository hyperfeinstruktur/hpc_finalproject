#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char ** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
              << std::endl;
    return 1;
  }

  MPI_Init(&argc, &argv);
  int prank, psize;

  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  
  CGSolverSparse sparse_solver(prank,psize);
  sparse_solver.read_matrix(argv[1]);
  
  int n = sparse_solver.n();
  int m = sparse_solver.m();
  double h = 1. / n;

  
  sparse_solver.init_source_term(h);

  std::vector<double> x_s(n);
  std::fill(x_s.begin(), x_s.end(), 0.);

  if (prank == 0){
    std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")"
              << std::endl;
  }
  /*
  // p-1 blocks of size blocksize & 1 block of size lastblocksize
  int blocksize = n/ psize;
  int lastblocksize = n - psize*blocksize + blocksize;
  int N_block;

  if (prank == 0){
    std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")"
              << std::endl;

    std::cout << "Using " << psize - 1 << " block(s) of size " << blocksize
              << " and 1 block of size " << lastblocksize << std::endl;
    if (prank < psize -1) N_block = blocksize;
    else N_block = lastblocksize;
  }
  */
  
  auto t1 = clk::now();
  sparse_solver.solve(x_s);
  second elapsed = clk::now() - t1;
  if (prank == 0)
  {
    std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";
  }

  MPI_Finalize();
  return 0;
}
