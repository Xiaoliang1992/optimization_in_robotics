#include "solver.h"
#include "Eigen/src/Core/Matrix.h"
#include "problem.h"
#include <Eigen/Cholesky>
#include <iostream>
#include <memory>

#define SOLVER_DEBUG
namespace optimization_solver {

// solver base
void Solver::SetParam(const SolverParameters &param) { param_ = param; }

void Solver::SetProblem(const ProblemType &type) {
  switch (type) {
  case ProblemType::PRosenbrock:
    if (N % 2 > 0) {
      std::cout << "N should be even!\nprogram termintated!" << std::endl;
      exit(0);
    }
    problem_ptr_ = std::make_shared<RosenbrockFunction>();
    break;
  case ProblemType::Example1:
    problem_ptr_ = std::make_shared<Example1Func>();
    break;
  }
}

double Solver::LineSearch(const Eigen::VectorXd &d, const double c,
                          const double tau_init, const Eigen::VectorXd &x,
                          const Eigen::VectorXd &g) {
  double tau = tau_init;
  double f0 = problem_ptr_->GetObjective(x);
  double f1 = problem_ptr_->GetObjective(x + tau * d);

  // Armijo condition for backtracking line search
  while (f1 > f0 + c * tau * d.transpose() * g) {
    tau *= 0.5;
    f1 = problem_ptr_->GetObjective(x + tau * d);
  }

  return tau;
}

// line-search steepest gradient descent with Armijo condition
Eigen::VectorXd GradientDescent::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;

  for (int i = 0; i < param_.max_iter; ++i) {
    iter_++;
    tau_ = param_.tau_init;
    g_ = problem_ptr_->GetGradient(x_);

    Eigen::VectorXd d = -g_;
    tau_ = LineSearch(d, param_.c, tau_, x_, g_);

#ifdef SOLVER_DEBUG
    std::cout << "iter = " << iter_ << ", tau = " << tau_
              << ", d_norm = " << d.norm() << ":\n"
              << x_ << std::endl;
#endif

    dx_ = tau_ * d;
    x_ += dx_;
    if (dx_.norm() < param_.terminate_threshold) {
      break;
    }

    // info
    info_.obj_val.push_back(problem_ptr_->GetObjective(x_));
    info_.iter_vec.push_back(iter_);
  }

  return x_;
}

// line-search steepest gradient descent with Armijo condition
Eigen::VectorXd NewtonsMethod::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;

  for (int i = 0; i < param_.max_iter; ++i) {
    iter_++;
    g_ = problem_ptr_->GetGradient(x_);
    H_ = problem_ptr_->GetHessian(x0);
    alpha_ = std::min(1.0, g_.cwiseAbs().maxCoeff()) / 10.0;
    Eigen::MatrixXd eye;
    M_ = H_ + alpha_ * eye.setIdentity(H_.rows(), H_.cols());

    Eigen::VectorXd d = M_.llt().solve(-g_);
    tau_ = LineSearch(d, param_.c, param_.tau_init, x_, g_);

#ifdef SOLVER_DEBUG
    std::cout << "iter = " << iter_ << ", tau = " << tau_
              << ", g_norm = " << g_.norm() << ", alpha = " << alpha_ << ":\n"
              << "x = \n"
              << x_ << std::endl;
#endif

    dx_ = tau_ * d;
    x_ += dx_;
    if (g_.norm() < param_.terminate_threshold) {
      break;
    }

    // info
    info_.obj_val.push_back(problem_ptr_->GetObjective(x_));
    info_.iter_vec.push_back(iter_);
  }

  return x_;
}

} // namespace optimization_solver