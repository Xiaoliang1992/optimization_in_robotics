#include "solver.h"
#include "Eigen/src/Core/Matrix.h"
#include "problem.h"
#include <Eigen/Cholesky>
#include <iostream>
#include <memory>

#define SOLVER_DEBUG

using namespace std;
namespace optimization_solver {

// solver base
void Solver::SetParam(const SolverParameters &param) { param_ = param; }

void Solver::SetProblem(const ProblemType &type) {
  switch (type) {
  case ProblemType::PRosenbrock:
    if (kRosenbrockN % 2 > 0) {
      std::cout << "kRosenbrockN should be even!\nprogram termintated!"
                << std::endl;
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
                          const double t_init, const Eigen::VectorXd &x,
                          const Eigen::VectorXd &g) {
  double t = t_init;
  double f0 = problem_ptr_->GetObjective(x);
  double f1 = problem_ptr_->GetObjective(x + t * d);

  // Armijo condition for backtracking line search
  while (f1 > f0 + c * t * d.transpose() * g) {
    t *= 0.5;
    f1 = problem_ptr_->GetObjective(x + t * d);
  }

  return t;
}

void Solver::BFGS(Eigen::MatrixXd &B, const Eigen::VectorXd &dx,
                  const Eigen::VectorXd &dg) {

  Eigen::MatrixXd eye;
  eye.setIdentity(dx.size(), dx.size());

  Eigen::MatrixXd dx_dgT = dx * dg.transpose();

  double dxT_dg = dx.dot(dg);

  if (dxT_dg > 0.0) {
    B = (eye - dx_dgT / dxT_dg) * B * (eye - dx_dgT / dxT_dg) +
        dx * dx.transpose() / dxT_dg;
  }
}

// line-search steepest gradient descent with Armijo condition
Eigen::VectorXd GradientDescent::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;

  for (int i = 0; i < param_.max_iter; ++i) {
    iter_++;
    t_ = param_.t_init;
    g_ = problem_ptr_->GetGradient(x_);

    Eigen::VectorXd d = -g_;
    t_ = LineSearch(d, param_.c, t_, x_, g_);

#ifdef SOLVER_DEBUG
    std::cout << "iter = " << iter_ << ", t = " << t_
              << ", d_norm = " << d.norm() << ":\n"
              << x_ << std::endl;
#endif

    dx_ = t_ * d;
    x_ += dx_;
    if (dx_.norm() < param_.terminate_threshold) {
      break;
    }

    // info
    info_.obj_val.push_back(std::log10(g_.norm()));
    info_.iter_vec.push_back(iter_);
  }

  return x_;
}

// newton's method
Eigen::VectorXd NewtonsMethod::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;

  for (int i = 0; i < param_.max_iter; ++i) {
    iter_++;
    g_ = problem_ptr_->GetGradient(x_);
    H_ = problem_ptr_->GetHessian(x_);
    alpha_ = std::min(1.0, g_.cwiseAbs().maxCoeff()) / 10.0;
    Eigen::MatrixXd eye;
    M_ = H_ + alpha_ * eye.setIdentity(H_.rows(), H_.cols());

    Eigen::VectorXd d = M_.llt().solve(-g_);
    t_ = LineSearch(d, param_.c, param_.t_init, x_, g_);

#ifdef SOLVER_DEBUG
    std::cout << "iter = " << iter_ << ", t = " << t_ << ",\nd = " << d << endl
              << ", g_norm = " << g_.norm() << ", alpha = " << alpha_ << ":\n"
              << "x = \n"
              << x_ << std::endl;
#endif

    dx_ = t_ * d;
    x_ += dx_;

    // info
    info_.obj_val.push_back(std::log10(g_.norm()));
    info_.iter_vec.push_back(iter_);

    if (g_.norm() < param_.terminate_threshold) {
      break;
    }
  }

  return x_;
}

// quasi-newton's method
Eigen::VectorXd QuasiNewtonsMethod::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;
  B_.resize(x0.size(), x0.size());
  B_.setIdentity();

  g_ = problem_ptr_->GetDiffGradient(x_);
  for (int i = 0; i < param_.max_iter; ++i) {
    if (g_.norm() < param_.terminate_threshold) {
      break;
    }
    iter_++;
    Eigen::VectorXd d = -B_ * g_;
    t_ = LineSearch(d, param_.c, param_.t_init, x_, g_);
    dx_ = t_ * d;
    x_ += dx_;
    auto g = g_;
    g_ = problem_ptr_->GetDiffGradient(x_);
    dg_ = g_ - g;

    BFGS(B_, dx_, dg_);

#ifdef SOLVER_DEBUG
    std::cout << "iter = " << iter_ << ", t = " << t_
              << ", g_norm = " << g_.norm() << ", alpha = " << alpha_ << ":\n"
              << "B_ = " << B_ << std::endl
              << "x = \n"
              << x_ << std::endl;
#endif

    // info
    info_.obj_val.push_back(std::log10(g_.norm()));
    info_.iter_vec.push_back(iter_);
  }

  return x_;
}

} // namespace optimization_solver