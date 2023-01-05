#include "cg.hh"
#include <mpi.h>

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

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

CGSolverSparse::CGSolverSparse(const int prank,const int psize)
: prank(prank), psize(psize)
{}

/*
Sparse version of the cg solver

-- MPI Version --
According to gprof, 99% of the time is spent in the mat_vec function. Thus, this is the step of
the algorithm that is parallelized/accelerated here.
Each of the threads considers only a range of the nonzero elements of A, so that effectively
the loop over [0,nnz] is parallelized. At the end, the vectors y=Ax of all threads are
summed (reduced) on the root thread, that does the rest of the computations sequentially.
In order to do Ax, all threads need to know the same up-to-date x. Thus, before calling
mat_vec, x is broadcasted from the root threads to all others.
These operations (bcast and sum-reduce) introduce a significant overhead and constitute the
main drawback of my implementation.
*/
void CGSolverSparse::solve(std::vector<double> & x) {

  // Initialize algorithm variables
  std::vector<double> r(m_n); // == vector r(k) = b-Ax(k)
  std::vector<double> p(m_n); // == vector p(k) = r(k) + beta*p(k-1)   (beta is a scalar)
  std::vector<double> Ap(m_n); // == matrix A times vector pk
  std::vector<double> Aptmp(m_n); // == matrix A times vector pk
  std::vector<double> tmp(m_n);
  double rsold = 1.0;
  double rsnew = 1.0;

  // r = b - A * x;
  m_A.mat_vec(x, Ap,z_start,z_end); // <--- stores Ax in vector Ap.
  MPI_Reduce(Ap.data(), Aptmp.data(), m_n, MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);

  if (prank == 0)
  {
    Ap = Aptmp;
    r = m_b; // <--- r0 = b - Ax0 in the alg. Ax0 is subtracted below:
    // cblas_daxpy: "Computes a constant times a vector plus a vector (double-precision)."
    // usage= daxpy(N: vector size, alpha: mult. constant, X: vector, stride, Y: vector, stride)
    // result is alpha*X + Y and is stored in Y
    // NOTE: vector.data() returns a pointer to the memory array used internally
    cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1); // <--- stores r - Ap in r

    // p0 = r0;
    p = r;

    // rsold = r' * r;
    rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);
  }
  // for i = 1:length(b)
  int k = 0;

  for (; k < m_n; ++k) {
    // Ap = A * p;
    MPI_Bcast(p.data(),m_n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    m_A.mat_vec(p, Ap,z_start,z_end);
    MPI_Reduce(Ap.data(), Aptmp.data(), m_n, MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);

    if (prank == 0)
    {
      Ap = Aptmp;
      // alpha = rsold / (p' * Ap);
      auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
                                    rsold * NEARZERO);

      // x = x + alpha * p;
      cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

      // r = r - alpha * Ap;
      cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

      // rsnew = r' * r;
      rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);
    }

    MPI_Bcast(&rsnew,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    if (prank == 0)
    {
      auto beta = rsnew / rsold;
      // p = r + (rsnew / rsold) * p;
      tmp = r;
      cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
      p = tmp;

      // rsold = rsnew;
      rsold = rsnew;
      if (DEBUG) {
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(rsold) << "\r" << std::flush;
      }
    }
  }

  if (DEBUG and prank==0) {
    m_A.mat_vec(x, r,z_start,z_end);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}

void CGSolverSparse::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();

  // Compute the range of z-indices that each thread does when calling mat_vec
  int N_block = m_A.nz() / psize + (m_A.nz() % psize != 0); // number of elements of A computed by a block
  int lastblocksize = m_A.nz() - (psize-1)*N_block; // number of elements of A for the last block (if nz is not a multiple of psize)
  z_start = prank*N_block;    // Start of index range
  z_end = (prank+1)*N_block;  // End of index range

  // Last thread has different range / blocksize
  if (prank == psize - 1) {
    N_block = lastblocksize;
    z_end = m_A.nz();
    }
  
  // Root thread prints some info
  if (prank == 0){
    std::cout << "Using " << psize - 1 << " block(s) of size " << N_block
        << " and 1 block of size " << lastblocksize << std::endl;

  }

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