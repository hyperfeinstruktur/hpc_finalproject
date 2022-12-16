#include <algorithm>
#include <string>
#include <vector>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  inline int nz() const { return *nz_storage; }
  inline int is_sym() const { return *m_is_sym_storage; }

  void read(const std::string & filename);

  // Compute y = Ax. Stores the result of Ax in the second argument
  // in standard (dense) einstein notation, y_i = A_ij x_j
  void mat_vec(const double* x, double* y, const size_t & len);
  void mat_vec_cuda(const double* x, double* y, const size_t & len);
  void mat_vec(const std::vector<double> & x, std::vector<double> & y);

private:
  int m_m{0};
  int m_n{0};

  size_t* nz_storage;
  int* irn_storage;
  int* jcn_storage;
  double* a_storage;
  bool* m_is_sym_storage;
};

#endif // __MATRIX_COO_H_
