#include "solver.h"
#include "Eigen/src/Core/Matrix.h"
#include "problem.h"
#include <Eigen/Cholesky>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>

#define SOLVER_DEBUG

static const double kMinimumStep = 1e-3;

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
  case ProblemType::Example2:
    problem_ptr_ = std::make_shared<Example2Func>();
    break;
  }
}

double Solver::LineSearch(const Eigen::VectorXd &d, const Eigen::VectorXd &x,
                          const Eigen::VectorXd &g,
                          const SolverParameters &param) {
  double t = param.t0;
  double f0 = problem_ptr_->GetCost(x);
  double f1 = problem_ptr_->GetCost(x + t * d);

  double c2_dT_g0 = param.c2 * d.transpose() * g;

  Eigen::VectorXd g1 = problem_ptr_->GetGradient(x + t * d);
  double dT_g1 = d.transpose() * g1;

  switch (param.linesearch_method) {
  case ArmijoCondition:
    while (f1 > f0 + param.c1 * t * d.transpose() * g) {
      t *= 0.5;
      f1 = problem_ptr_->GetCost(x + t * d);
      if (t < kMinimumStep) {
        break;
      }
    }
    break;
  case WolfeWeakCondition:
    while ((f1 > f0 + param.c1 * t * d.transpose() * g) || (c2_dT_g0 > dT_g1)) {
      t *= 0.5;
      f1 = problem_ptr_->GetCost(x + t * d);
      g1 = problem_ptr_->GetGradient(x + t * d);
      dT_g1 = d.transpose() * g1;
      if (t < kMinimumStep) {
        break;
      }
    }
    break;
  case WolfeStrongCondition:
    while ((f1 > f0 + param.c1 * t * d.transpose() * g) ||
           (std::abs(c2_dT_g0) < std::abs(dT_g1))) {
      t *= 0.5;
      f1 = problem_ptr_->GetCost(x + t * d);
      g1 = problem_ptr_->GetGradient(x + t * d);
      dT_g1 = d.transpose() * g1;
      if (t < kMinimumStep) {
        break;
      }
    }
    break;
  default:
    while (f1 > f0 + param.c1 * t * d.transpose() * g) {
      t *= 0.5;
      f1 = problem_ptr_->GetCost(x + t * d);
      if (t < kMinimumStep) {
        break;
      }
    }
  }

  return t;
}

void Solver::BFGS(Eigen::MatrixXd &B, const Eigen::VectorXd &dx,
                  const Eigen::VectorXd &g, const Eigen::VectorXd &dg) {
  Eigen::MatrixXd eye;
  eye.setIdentity(dx.size(), dx.size());

  Eigen::MatrixXd dx_dgT = dx * dg.transpose();
  Eigen::MatrixXd dg_dxT = dg * dx.transpose();

  double dxT_dg = dx.dot(dg);
  static const double kEps = 1e-6;

  if (dxT_dg > kEps * g.norm() * dx.norm()) {
    B = (eye - dx_dgT / dxT_dg) * B * (eye - dg_dxT / dxT_dg) +
        dx * dx.transpose() / dxT_dg;
  }
}

// line-search steepest gradient descent with Armijo condition
Eigen::VectorXd GradientDescent::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;

  for (int i = 0; i < param_.max_iter; ++i) {
    iter_++;
    t_ = param_.t0;
    g_ = problem_ptr_->GetGradient(x_);

    d_ = -g_;
    t_ = LineSearch(d_, x_, g_, param_);

    DebugInfo();

    dx_ = t_ * d_;
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

    d_ = M_.llt().solve(-g_);
    t_ = LineSearch(d_, x_, g_, param_);

    DebugInfo();

    dx_ = t_ * d_;
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

  f_ = problem_ptr_->GetCost(x_);
  g_ = problem_ptr_->GetGradient(x_);
  d_ = -B_ * g_;
  DebugInfo();

  while (g_.norm() > param_.terminate_threshold) {
    iter_++;
    d_ = -B_ * g_;

    t_ = LineSearch(d_, x_, g_, param_);

    dx_ = t_ * d_;
    x_ += dx_;
    auto g = g_;
    g_ = problem_ptr_->GetGradient(x_);
    dg_ = g_ - g;

    f_ = problem_ptr_->GetCost(x_);

    BFGS(B_, dx_, g_, dg_);

    DebugInfo();

    // info
    info_.obj_val.push_back(std::log10(g_.norm()));
    info_.iter_vec.push_back(iter_);

    if (iter_ >= param_.max_iter) {
      break;
    }
  }

  return x_;
}

void Solver::DebugInfo() {
#ifdef SOLVER_DEBUG
  std::cout << "----------------------- iter = " << iter_
            << "-----------------------" << endl
            << "t = " << t_ << ",\nd = " << d_ << endl
            << "f = " << f_ << endl
            << ", g_norm = " << g_.norm() << ", alpha = " << alpha_ << ":\n"
            << "g = \n"
            << g_ << "\nx = \n"
            << x_ << std::endl;
#endif
}

} // namespace optimization_solver