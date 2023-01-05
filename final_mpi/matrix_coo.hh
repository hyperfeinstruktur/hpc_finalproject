#include <algorithm>
#include <string>
#include <exception>
#include <iostream>
#include <vector>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  inline int nz() const { return irn.size(); }
  inline int is_sym() const { return m_is_sym; }

  void read(const std::string & filename);

  // Compute y = Ax. Stores the result of Ax in the second argument
  void mat_vec(const std::vector<double> & x, std::vector<double> & y, const int & z_start,const int & z_end);

  std::vector<int> irn;
  std::vector<int> jcn;
  std::vector<double> a;

private:
  int m_m{0};
  int m_n{0};
  bool m_is_sym{false};
};

#endif // __MATRIX_COO_H_
