#include "cg.hh"
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char ** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
              << " [CUDA block size]" << std::endl;
    return 1;
  }
  
  // ------------ I copy the error checking section from the assignemtnt 2 code ------------
  // By default, we use device 0,
  int dev_id = 0;

  cudaDeviceProp device_prop;
  cudaGetDevice(&dev_id);
  cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                 "threads can use ::cudaSetDevice()"
              << std::endl;
    return -1;
  }

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "cudaGetDeviceProperties returned error code " << error
              << ", line(" << __LINE__ << ")" << std::endl;
    return error;
  } else {
    std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
              << "\" with compute capability " << device_prop.major << "."
              << device_prop.minor << std::endl;
  }
  
  // -----------------------------------------------------------------------------------
  
  CGSolverSparse sparse_solver;
  sparse_solver.read_matrix(argv[1]);
  
  dim3 block_size;
  block_size.x = std::stoi(argv[2]);
  dim3 grid_size;
  grid_size.x = sparse_solver.nz() / block_size.x + (sparse_solver.nz() % block_size.x != 0); // floor of nz/blocksize

  int n = sparse_solver.n();
  int m = sparse_solver.m();
  double h = 1. / n;

  sparse_solver.init_source_term(h);

  std::vector<double> x_s(n);
  std::fill(x_s.begin(), x_s.end(), 0.);

  std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")"
            << std::endl;
  auto t1 = clk::now();
  sparse_solver.solve(x_s,grid_size,block_size);
  second elapsed = clk::now() - t1;
  std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";

  return 0;
}
