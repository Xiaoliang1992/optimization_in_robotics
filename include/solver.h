#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "problem.h"
#include <eigen3/Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

namespace optimization_solver {
enum SolverType {
  MGradientDescent = 0,
};

struct SolverParameters {
  int max_iter = 5000;               // max iteration time
  double c = 0.4;                    // c
  double tau_init = 1.0;             // init step size
  double terminate_threshold = 1e-6; // iteration terminate threshold
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
  virtual void SetProblem(const ProblemType &type) final;
  virtual void SetParam(const SolverParameters &param) final;
  virtual double LineSearch(const Eigen::VectorXd &d, const double c,
                            const double tau_init, const Eigen::VectorXd &x,
                            const Eigen::VectorXd &g) final;
  virtual Eigen::VectorXd Solve(const Eigen::VectorXd &x0) = 0;

  virtual Eigen::VectorXd Getx() final { return x_; }
  virtual Eigen::VectorXd Getg() final { return g_; }
  virtual SolverDebugInfo *GetInfoPtr() final { return &info_; }
  virtual int GetIter() final { return iter_; }

protected:
  std::shared_ptr<Problem> problem_ptr_; // problem ptr
  SolverParameters param_;               // parameters
  Eigen::VectorXd x_;                    // iterative optimization variables
  Eigen::VectorXd dx_; // iterative optimization variables increments
  Eigen::VectorXd g_;  // gradient
  Eigen::VectorXd lb_; // lower bound
  Eigen::VectorXd ub_; // upper bound
  double tau_ = 0.0;   // step size
  int iter_ = 0;       // iter time

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