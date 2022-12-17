#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

__global__ void daxpy_cuda(const double* X,const double alpha,double* Y,const int len)
  {
    // Store X + alpha*Y in Y
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < len)
    {
        Y[i] = X[i] + alpha*Y[i];
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

  // Initialize algorithm variables
  //std::vector<double> r(m_n); // == vector r(k) = b-Ax(k)
  //std::vector<double> p(m_n); // == vector p(k) = r(k) + beta*p(k-1)   (beta is a scalar)
  //std::vector<double> Ap(m_n); // == matrix A times vector pk
  //std::vector<double> tmp(m_n);

  //cublasHandle_t handle;
  //cublasCreate(&handle);

  double r[m_n];
  double p[m_n];
  double Ap[m_n];
  double tmp[m_n];
  double x[m_n];

  // Device Pointers
  //double* xptr = x;
  double* pptr = p;
  double* Apptr = Ap;
  double* rptr = r;
  // Allocate device memory for arrays that are used in kernels
  //cudaMallocManaged(&xptr,m_n*sizeof(double));
  cudaMalloc(&pptr,m_n*sizeof(double));
  //cudaMallocManaged(&pptr,m_n*sizeof(double));
  cudaMalloc(&Apptr,m_n*sizeof(double));
  cudaMalloc(&rptr,m_n*sizeof(double));

  std::copy(xvect.begin(),xvect.end(),x);
  //cudaMemcpy(xptr,x,m_n*sizeof(double),cudaMemcpyHostToDevice);

  // TODO
  // Replace all arrays by device code, use kernels and cublas, copy back to host only at exit
  // r = b - A * x;
  m_A.mat_vec(x, Ap,m_n); // <--- stores Ax in vector Ap.
  std::copy(m_b.begin(),m_b.end(),r); 
  cudaMemcpy(rptr,r,m_n*sizeof(double),cudaMemcpyHostToDevice);

  //r = m_b; // <--- r0 = b - Ax0 in the alg. Ax0 is subtracted below:
  // cblas_daxpy: "Computes a constant times a vector plus a vector (double-precision)."
  // usage= daxpy(N: vector size, alpha: mult. constant, X: vector, stride, Y: vector, stride)
  // result is alpha*X + Y and is stored in Y
  // NOTE: vector.data() returns a pointer to the memory array used internally
  //cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1); // <--- stores r - Ap in r
  //cblas_daxpy(m_n, -1., Ap, 1, r, 1); // <--- stores r - Ap in r
  daxpy_cuda<<<200,50>>>(Apptr,-1,rptr,m_n);
  cudaMemcpy(r,rptr,m_n*sizeof(double),cudaMemcpyDeviceToHost);

  // p0 = r0;
  //p = r;
  std::copy(r,r+m_n,p); 

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r, 1, r, 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {
    // Ap = A * p;
    std::fill(Ap,Ap+m_n,0.);
    cudaMemcpy(pptr,p,m_n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Apptr,Ap,m_n*sizeof(double),cudaMemcpyHostToDevice);
    m_A.mat_vec_cuda(pptr, Apptr,grid_size,block_size); // <-- This is where 99% of time is spent according to gprof
    cudaMemcpy(p,pptr,m_n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Ap,Apptr,m_n*sizeof(double),cudaMemcpyDeviceToHost);
    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p, 1, Ap, 1),
                                  rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p, 1, x, 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap, 1, r, 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r, 1, r, 1);
    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    //tmp = r;
    // TODO: copy on GPU?
    std::copy(r,r+m_n,tmp); 

    cblas_daxpy(m_n, beta, p, 1, tmp, 1);
    //p = tmp;
    std::copy(tmp,tmp+m_n,p); 

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }
  }
  if (DEBUG) {
    m_A.mat_vec(x, r,m_n);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r, 1);
    auto res = std::sqrt(cblas_ddot(m_n, r, 1, r, 1)) /
               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x, 1, x, 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
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