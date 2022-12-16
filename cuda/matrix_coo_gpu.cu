#include <iostream>
#include <exception>
#include "matrix_coo.hh"


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