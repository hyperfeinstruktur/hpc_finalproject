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
                                const bool* sym,
                                const size_t* nz)
  {
    int z = blockDim.x*blockIdx.x + threadIdx.x;
    //int z = blockIdx.x + stride*threadIdx.x;
    if (z < *nz) {
      auto i = irn[z];
      auto j = jcn[z];
      auto a_ = a[z];
    
      //y[i] += a_ * x[j];
      atomicAdd(&y[i],a_ * x[j]);
      if (*sym and (i != j)) {
        atomicAdd(&y[j],a_ * x[i]);
        //y[j] += a_ * x[i];
      }
    }
  }



void MatrixCOO::mat_vec(const std::vector<double> & x, std::vector<double> & y)
 {
    std::fill_n(y.begin(), y.size(), 0.);

    for (size_t z = 0; z < *nz_storage; ++z) {
      auto i = irn_storage[z];
      auto j = jcn_storage[z];
      auto a_ = a_storage[z];

      y[i] += a_ * x[j];
      if (*m_is_sym_storage and (i != j)) {
        y[j] += a_ * x[i];
      }
    }
  }

void MatrixCOO::mat_vec(const double* x, double* y, const size_t & len)
 {
    std::fill_n(y, len, 0.);
    for (size_t z = 0; z < *nz_storage; ++z) {
      auto i = irn_storage[z];
      auto j = jcn_storage[z];
      auto a_ = a_storage[z];

      y[i] += a_ * x[j];
      if (*m_is_sym_storage and (i != j)) {
        y[j] += a_ * x[i];
      }
    }
  }

void MatrixCOO::mat_vec_cuda(const double* x, double* y, const size_t & len ,const dim3 & grid_size,const dim3 & block_size)
 {
    //std::fill_n(y, len, 0.);

    if (false) {std::cout << len << std::endl;}
    //dim3 grid_size;
    //dim3 block_size;
    //block_size.x = 128;
    //grid_size.x = *nz_storage / block_size.x + (*nz_storage % block_size.x != 0);
    static bool first{true};
    if (first) {
        std::cout << "Block size:    " << block_size.x << ":" << block_size.y << "\n"
                  << "Grid_size:     " << grid_size.x << ":" << grid_size.y << std::endl;
        first = false;
    }

    //compute_element<<<grid_size,block_size>>>(x,y,irn_storage,jcn_storage,a_storage,m_is_sym_storage,nz_storage,grid_size.x);
    compute_element<<<596,50>>>(x,y,irn_storage,jcn_storage,a_storage,m_is_sym_storage,nz_storage);
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if(error != cudaSuccess) {
        throw std::runtime_error("Error Launching Kernel: "
                                 + std::string(cudaGetErrorName(error)) + " - "
                                 + std::string(cudaGetErrorString(error)));
    }
  }

