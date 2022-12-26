#include <iostream>
#include <exception>
#include "matrix_coo.hh"

#include <cuda.h>

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

