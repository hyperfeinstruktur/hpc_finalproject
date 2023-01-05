#include <iostream>
#include <exception>
#include "matrix_coo.hh"

#include <cuda.h>

/*
The major issue I encountered was that the array Ap is shared between all threads in
the kernel and so, due to the nature of the COO format, multiple threads may acces a
given element Ap[i] at the same time. In the dense variant, one could imagine that there
is one thread per row (per i) and then this problem never happens. I’ve looked into
solutions on the internet, some of which involved converting the COO matrix to other
sparse formats that are better suited to avoid this problem, like the CSR format.

Eventually, however, an easy solution turned out to be using a CUDA atomic function
(https://docs.nvidia.com/cuda/cuda-c-programming-guide/indexhtml#atomic-functions)
that prevents multiple threads from accessing the same element of Ap at the same time.
As explained in the documentation linked in the footnote, atomicAdd is not defined for
double precision numbers depending on the compute capability and CUDA version, and so
I followed the documentation’s explanation to add it manually (see below).
Using this function solves the race condition problem, albeit not in the cleanest way.
*/

// This is the custom atomicAdd function for doubles (from CUDA documentation)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// Kernel to compute the sparse matrix-vector product
__global__ void compute_element(const double* x,
                                double* const y,
                                const int* irn,
                                const int* jcn,
                                const double* a,
                                const bool sym,
                                const int nz)
  {
    int z = blockDim.x*blockIdx.x + threadIdx.x;
    if (z < nz) {
      auto i = irn[z];
      auto j = jcn[z];
      auto a_ = a[z];
    
      double tmp = a_ * x[j];
      atomicAdd(&y[i],tmp);
      if (sym and (i != j)) {
        tmp = a_ * x[i];
        atomicAdd(&y[j],tmp);
      }
    }
  }

// GPU version of mat_vec. Same idea as the initial code but calls a kernel to do the actual computation.
void MatrixCOO::mat_vec_cuda(const double* x, double* y ,const dim3 & grid_size,const dim3 & block_size)
 {  
    cudaMemset(y,0,m_n*sizeof(double));
    compute_element<<<grid_size,block_size>>>(x,y,irn_storage,jcn_storage,a_storage,m_is_sym,m_nz);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if(error != cudaSuccess) {
        throw std::runtime_error("Error Launching Kernel: "
                                 + std::string(cudaGetErrorName(error)) + " - "
                                 + std::string(cudaGetErrorString(error)));
    }
  }

// CPU version of mat_vec (used as reference and for the single call in debug block)
void MatrixCOO::mat_vec(const std::vector<double> & x, std::vector<double> & y)
 {
    std::fill_n(y.begin(), y.size(), 0.);

    for (int z = 0; z < m_nz; ++z) {
      auto i = irn[z];
      auto j = jcn[z];
      auto a_ = a[z];

      y[i] += a_ * x[j];
      if (m_is_sym and (i != j)) {
        y[j] += a_ * x[i];
      }
    }
  }