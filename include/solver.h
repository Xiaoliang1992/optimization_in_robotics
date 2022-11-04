#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "problem.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

namespace optimization_solver {
enum SolverType {
  TGradientDescent,
  TNewtonsMethod,
  TQuasiNewtonsMethod,
  TNetonCGMethod,
};

enum LineSearchMethod {
  ArmijoCondition,
  WolfeWeakCondition,
  WolfeStrongCondition,
};

struct SolverParameters {
  size_t max_iter = 80;                           // max iteration time
  uint8_t linesearch_method = ArmijoCondition; // line search method
  double c1 = 0.0001;                             // c1
  double c2 = 0.9;                                // c2
  double t0 = 1.0;                                // init step size
  double terminate_threshold = 1e-6; // iteration terminate threshold
  uint16_t m_size = 30;              // LBFGS memory size
  bool debug_enable = false;         // debug flag
};
// solver base
struct SolverDebugInfo {
  size_t iter = 0;         // iter time
  size_t problem_size = 0; // problem size
  std::vector<Eigen::VectorXd> x_vec;
  std::vector<Eigen::VectorXd> dx_vec;
  std::vector<Eigen::VectorXd> g_vec;
  std::vector<double> tau_vec;
  std::vector<double> g_norm_vec;
  std::vector<double> cost_vec;
  std::vector<int> iter_vec;

  void Clear() {
    iter = 0;
    problem_size = 0;
    x_vec.clear();
    g_vec.clear();
    tau_vec.clear();
    g_norm_vec.clear();
    cost_vec.clear();
    iter_vec.clear();
  }
};

class SolverBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void SetProblem(const uint8_t &type) final;
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
  virtual std::shared_ptr<Problem> GetProblemPtr() final {
    return problem_ptr_;
  }

  // for debug

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
  double t_ = 0.0;  // step size
  size_t iter_ = 0; // iter

  SolverDebugInfo info_; // debug info
};

// line-search steepest gradient descent with Armijo condition
class GradientDescent : public SolverBase {
public:
  GradientDescent() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
};

// NewtonsMethod with linesearch
class NewtonsMethod : public SolverBase {
public:
  NewtonsMethod() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
};

class QuasiNewtonsMethod : public SolverBase {
public:
  QuasiNewtonsMethod() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
};

class NetonCGMethod : public SolverBase {
public:
  NetonCGMethod() {}
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0) override;
  Eigen::VectorXd Gamau(const Eigen::VectorXd &u, const Eigen::VectorXd &x);
};

// solver class
class Solver {
public:
  void SetSolver(const uint8_t &type);
  void SetProblem(const uint8_t &type);
  void SetParam(const SolverParameters &param);
  Eigen::VectorXd Solve(const Eigen::VectorXd &x0);
  SolverDebugInfo *GetInfoPtr() { return solver_ptr_->GetInfoPtr(); }
  SolverDebugInfo GetInfo() { return *solver_ptr_->GetInfoPtr(); }

private:
  std::shared_ptr<SolverBase> solver_ptr_;
  bool problem_setflag_ = false;
  bool solver_setflag_ = false;
};

} // namespace optimization_solver

#endif