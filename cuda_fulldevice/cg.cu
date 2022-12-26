#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

__global__ void daxpy_cuda(const int len,const double alpha,const double* X,double* Y)
  {
    // Store X + alpha*Y in Y
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < len)
    {
        Y[i] += alpha*X[i];
    }
    
  }
  
__global__ void fillzero_cuda(double* X, const int len)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < len)
    {
        X[i] = 0.0;
    }
}
/*
    cgsolver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

*/


/*
Sparse version of the cg solver
*/
void CGSolverSparse::solve(std::vector<double> & xvect,const dim3 & grid_size,const dim3 & block_size) {

  static bool first{true};
    if (first) {
        std::cout << "Block size:    " << block_size.x << ":" << block_size.y << "\n"
                  << "Grid_size:     " << grid_size.x << ":" << grid_size.y << std::endl;
        first = false;
    }

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  cublasCreate(&cublasH);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(cublasH, stream);

  
  // Initialize algorithm variables
  std::vector<double> r(m_n); // == vector r(k) = b-Ax(k)
  std::vector<double> p(m_n); // == vector p(k) = r(k) + beta*p(k-1)   (beta is a scalar)
  std::vector<double> Ap(m_n); // == matrix A times vector pk
  std::vector<double> tmp(m_n);

  // Device Pointers
  double* dev_p;
  double* dev_Ap;
  double* dev_r;
  double* dev_tmp;
  double* dev_x;

  auto bitsize = m_n*sizeof(double);
  // Allocate device memory
  cudaMalloc(&dev_p,bitsize);
  cudaMalloc(&dev_Ap,bitsize);
  cudaMalloc(&dev_r,bitsize);
  cudaMalloc(&dev_tmp,bitsize);
  cudaMalloc(&dev_x,bitsize);

  // #### The part before the loop is all on host to avoid having to allocate more memory #### //

  cudaMemcpy(dev_x,xvect.data(),bitsize,cudaMemcpyHostToDevice);
  // r = b - A * x;
  m_A.mat_vec(xvect, Ap); // <--- stores Ax in vector Ap.
  r = m_b;

  //r = m_b; // <--- r0 = b - Ax0 in the alg. Ax0 is subtracted below:
  // cblas_daxpy: "Computes a constant times a vector plus a vector (double-precision)."
  // usage= daxpy(N: vector size, alpha: mult. constant, X: vector, stride, Y: vector, stride)
  // result is alpha*X + Y and is stored in Y
  // NOTE: vector.data() returns a pointer to the memory array used internally
  
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1); // <--- stores r - Ap in r

  p = r;


  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  double ddot_temp = 0;
  double rsnew = 0;

  cudaMemcpy(dev_p,p.data(),bitsize,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Ap,Ap.data(),bitsize,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_r,r.data(),bitsize,cudaMemcpyHostToDevice);


  for (; k < m_n; ++k) {
    // Ap = A * p;
    m_A.mat_vec_cuda(dev_p, dev_Ap,grid_size,block_size); // <-- This is where 99% of time is spent according to gprof

    // alpha = rsold / (p' * Ap);
    //auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
    //                              rsold * NEARZERO);

    cublasDdot(cublasH,m_n,dev_p,1,dev_Ap,1,&ddot_temp);
    cudaStreamSynchronize(stream);
    double alpha = rsold/std::max(ddot_temp,rsold * NEARZERO);

    // x = x + alpha * p;
    //cblas_daxpy(m_n, alpha, p.data(), 1, xvect.data(), 1);
    daxpy_cuda<<<200,50>>>(m_n,alpha,dev_p,dev_x);
    // r = r - alpha * Ap;
    //cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);
    daxpy_cuda<<<200,50>>>(m_n,-alpha,dev_Ap,dev_r);

    cudaDeviceSynchronize();
    // rsnew = r' * r;
    cublasDdot(cublasH,m_n,dev_r,1,dev_r,1,&rsnew);
    cudaStreamSynchronize(stream);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    cudaMemcpy(dev_tmp,dev_r,bitsize,cudaMemcpyDeviceToDevice);

    //cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    daxpy_cuda<<<200,50>>>(m_n,beta,dev_p,dev_tmp);

    cudaMemcpy(dev_p,dev_tmp,bitsize,cudaMemcpyDeviceToDevice);

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }
  }
  if (DEBUG) {
    m_A.mat_vec(xvect, r);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, xvect.data(), 1, xvect.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
  
  cudaFree(dev_p);
  cudaFree(dev_Ap);
  cudaFree(dev_r);
  cudaFree(dev_tmp);
  cudaFree(dev_x);
  cublasDestroy(cublasH);
  cudaStreamDestroy(stream);
  cudaDeviceReset();
}

void CGSolverSparse::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}