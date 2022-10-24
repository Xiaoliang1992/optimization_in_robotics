#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "problem.h"
#include <deque>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

namespace optimization_solver {
enum SolverType {
  MGradientDescent = 0,
};

enum LineSearchMethod {
  ArmijoCondition,
  WolfeWeakCondition,
  WolfeStrongCondition,
};

struct SolverParameters {
  int max_iter = 80;                              // max iteration time
  uint8_t linesearch_method = WolfeWeakCondition; // line search method
  double c1 = 0.0001;                             // c1
  double c2 = 0.9;                                // c2
  double t0 = 1.0;                                // init step size
  double terminate_threshold = 1e-6; // iteration terminate threshold
  uint16_t m = 30;                    // LBFGS memory size
};

// solver base

struct SolverDebugInfo {
  std::vector<Eigen::VectorXd> x_vec;
  std::vector<Eigen::VectorXd> dx_vec;
  std::vector<Eigen::VectorXd> g_vec;
  std::vector<double> tau_vec;
  std::vector<double> obj_val;
  std::vector<int> iter_vec;
};

class Solver {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void SetProblem(const ProblemType &type) final;
  virtual void SetParam(const SolverParameters &param) final;
  virtual double LineSearch(const Eigen::VectorXd &d, const Eigen::VectorXd &x,
                            const Eigen::VectorXd &g,
                            const SolverParameters &param) final;
  virtual double LOLineSearch(const Eigen::VectorXd &d,
                              const Eigen::VectorXd &x,
                              const Eigen::VectorXd &g,
                              const SolverParameters &param) final;
  virtual Eigen::VectorXd BFGS(const Eigen::VectorXd &dx,
                               const Eigen::VectorXd &g,
                               const Eigen::VectorXd &dg) final;

  virtual Eigen::VectorXd LBFGS(const Eigen::VectorXd &dx,
                                const Eigen::VectorXd &g,
                                const Eigen::VectorXd &dg) final;

  virtual Eigen::VectorXd Solve(const Eigen::VectorXd &x0) = 0;

  virtual Eigen::VectorXd Getx() final { return x_; }
  virtual Eigen::VectorXd Getg() final { return g_; }
  virtual SolverDebugInfo *GetInfoPtr() final { return &info_; }
  virtual int GetIter() final { return iter_; }
  void DebugInfo();

protected:
  std::shared_ptr<Problem> problem_ptr_; // problem ptr
  SolverParameters param_;               // parameters
  Eigen::VectorXd x_;                    // iterative optimization variables
  Eigen::VectorXd dx_; // iterative optimization variables increments
  Eigen::MatrixXd H_;  // hessian matrix
  Eigen::MatrixXd B_;  // dx = B * dg
  Eigen::MatrixXd M_;  // PSD matrix M = H + alpha * I
  Eigen::VectorXd g_;  // gradient
  Eigen::VectorXd d_;  // iterate direction
  Eigen::VectorXd dg_; // iterative gradient increments
  Eigen::VectorXd lb_; // lower bound
  Eigen::VectorXd ub_; // upper bound

  std::deque<Eigen::VectorXd> dx_vec_;
  std::deque<Eigen::VectorXd> dg_vec_;
  std::deque<double> rho_vec_;

  double f_; // cost
  double alpha_ = 0.0;
  double t_ = 0.0; // step size
  int iter_ = 0;   // iter time

  SolverDebugInfo info_; // debug info
};

// line-search steepest gradient descent with Armijo condition
class GradientDescent : public Solver {
public:
  GradientDescent() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
};

// NewtonsMethod with linesearch
class NewtonsMethod : public Solver {
public:
  NewtonsMethod() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
};

class QuasiNewtonsMethod : public Solver {
public:
  QuasiNewtonsMethod() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
};

} // namespace optimization_solver

#endif