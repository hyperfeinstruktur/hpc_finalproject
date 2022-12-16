#include "matrix.hh"
#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>

#ifndef __CG_HH__
#define __CG_HH__

class Solver {
public:
  virtual void read_matrix(const std::string & filename) = 0;

  // Initialize the vector b (from a sin())
  void init_source_term(double h);

  // Solve Ax=b
  virtual void solve(std::vector<double> & x) = 0;

  // Getter functions for size
  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  // Setter for size
  void tolerance(double tolerance) { m_tolerance = tolerance; }

protected:
  // Dimensions of the matrix A
  int m_m{0};
  int m_n{0};

  // b vector in Ax = b
  std::vector<double> m_b;

  // Epsilon: ||r||2 < eps, with r = b-Ax
  double m_tolerance{1e-10};
};

class CGSolverSparse : public Solver {
public:
  CGSolverSparse() = default;

  // Set the matrix A from input file
  virtual void read_matrix(const std::string & filename);

  // Solve Ax = b using CG <-- 99% spent here from gprof
  virtual void solve(std::vector<double> & x);

private:

  // The matrix A in Ax = b
  MatrixCOO m_A;
};



// =========== UNUSED =============

class CGSolver : public Solver {
public:
  CGSolver() = default;
  virtual void read_matrix(const std::string & filename);
  virtual void solve(std::vector<double> & x);

private:
  Matrix m_A;
};

#endif /* __CG_HH__ */
