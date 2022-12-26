#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  inline int nz() const { return m_nz; }
  inline int is_sym() const { return m_is_sym; }

  void read(const std::string & filename);

  // Compute y = Ax. Stores the result of Ax in the second argument
  // in standard (dense) einstein notation, y_i = A_ij x_j

  // This is the default serial CPU version, for reference and for the first call to mat_vec
  void mat_vec(const std::vector<double> & x, std::vector<double> & y);

  // This is the GPU version
  void mat_vec_cuda(const double* x, double* y,const dim3 & grid_size,const dim3 & block_size);
  
private:
  // Host Data
  std::vector<int> irn;
  std::vector<int> jcn;
  std::vector<double> a;

  int m_m{0};
  int m_n{0};
  bool m_is_sym{false};
  int m_nz{0};
  

  // Pointers for Device memory
  int* irn_storage;
  int* jcn_storage;
  double* a_storage;
};

#endif // __MATRIX_COO_H_
