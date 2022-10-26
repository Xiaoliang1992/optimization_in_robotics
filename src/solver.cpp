#include "solver.h"
#include "Eigen/src/Core/Matrix.h"
#include "problem.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

#define SOLVER_DEBUG

static const double kMinimumStep = 1e-8;

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
  case ProblemType::Example3:
    problem_ptr_ = std::make_shared<Example3Func>();
    break;
  case ProblemType::Example4:
    problem_ptr_ = std::make_shared<Example4Func>();
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

double Solver::LOLineSearch(const Eigen::VectorXd &d, const Eigen::VectorXd &x,
                            const Eigen::VectorXd &g,
                            const SolverParameters &param) {

  double t = param.t0;
  double f0 = problem_ptr_->GetCost(x);
  double f1 = problem_ptr_->GetCost(x + t * d);

  double c2_dT_g0 = param.c2 * d.transpose() * g;

  Eigen::VectorXd g1 = problem_ptr_->GetGradient(x + t * d);
  double dT_g1 = d.transpose() * g1;

  bool armijo_condition = (f1 > f0 + param.c1 * t * d.transpose() * g);
  bool wolfe_condition = (c2_dT_g0 > dT_g1);

  double l = 0.0;
  double u_max = 100000000.0;
  double u = u_max;

  while (armijo_condition || wolfe_condition) {
    if (armijo_condition) {
      u = t;
    } else if (wolfe_condition) {
      l = t;
    } else {
      return t;
    }

    if (u < u_max) {
      t = (l + u) / 2.0;
    } else {
      t = 2.0 * l;
    }

    if (t < kMinimumStep) {
      break;
    }

    f1 = problem_ptr_->GetCost(x + t * d);
    g1 = problem_ptr_->GetGradient(x + t * d);
    dT_g1 = d.transpose() * g1;

    armijo_condition = (f1 > f0 + param.c1 * t * d.transpose() * g);
    wolfe_condition = (c2_dT_g0 > dT_g1);
  }

  return t;
}

Eigen::VectorXd Solver::BFGS(const Eigen::VectorXd &dx,
                             const Eigen::VectorXd &g,
                             const Eigen::VectorXd &dg) {
  Eigen::MatrixXd eye;
  eye.setIdentity(dx.size(), dx.size());

  Eigen::MatrixXd dx_dgT = dx * dg.transpose();
  Eigen::MatrixXd dg_dxT = dg * dx.transpose();

  double dxT_dg = dx.dot(dg);
  static const double kEps = 1e-6;

  if (dxT_dg > kEps * g.norm() * dx.norm()) {
    B_ = (eye - dx_dgT / dxT_dg) * B_ * (eye - dg_dxT / dxT_dg) +
         dx * dx.transpose() / dxT_dg;
  }

  auto d = -B_ * g_;

  return d;
}

Eigen::VectorXd Solver::LBFGS(const Eigen::VectorXd &dx,
                              const Eigen::VectorXd &g,
                              const Eigen::VectorXd &dg) {
  Eigen::VectorXd d = -g;
  while (dx_vec_.size() >= param_.m) {
    dx_vec_.pop_front();
    dg_vec_.pop_front();
    rho_vec_.pop_front();
  }

  dx_vec_.emplace_back(dx);
  dg_vec_.emplace_back(dg);
  rho_vec_.emplace_back(1.0 / std::max(dx.dot(dg), kMinimumStep));

  static std::vector<double> alpha_vec;
  alpha_vec.resize(param_.m);

  int k = dx_vec_.size();

  if (k > 2) {
    for (int i = k - 1; i >= 0; --i) {
      alpha_vec[i] = rho_vec_[i] * dx_vec_[i].dot(d);
      d = d - alpha_vec[i] * dg_vec_[i];
    }

    double gama = alpha_vec[k - 1] * dg_vec_[k - 1].norm();
    gama = std::max(gama, kMinimumStep);

    d = d / gama;

    for (int i = 0; i < k - 1; ++i) {
      double beta = rho_vec_[i] * dg_vec_[i].dot(d);
      d = d + dx_vec_[i] * (alpha_vec[i] - beta);
    }
  }
  return d;
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

    t_ = LOLineSearch(d_, x_, g_, param_);

    dx_ = t_ * d_;
    x_ += dx_;
    auto g = g_;
    g_ = problem_ptr_->GetGradient(x_);
    dg_ = g_ - g;

    f_ = problem_ptr_->GetCost(x_);

    // d_ = BFGS(dx_, g_, dg_);
    d_ = LBFGS(dx_, g_, dg_);

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

// newton's CG method
Eigen::VectorXd NetonCGMethod::Solve(const Eigen::VectorXd &x0) {
  x_ = x0;
  iter_ = 0;
  f_ = problem_ptr_->GetCost(x_);
  g_ = problem_ptr_->GetGradient(x_);

  DebugInfo();

  Eigen::VectorXd d(x_.size());
  Eigen::VectorXd v(x_.size());
  Eigen::VectorXd v1(x_.size());
  Eigen::VectorXd u(x_.size());
  int j = 0;

  while (g_.norm() > param_.terminate_threshold) {
    iter_++;
    double e = std::min(1.0, g_.norm()) / 10.0;
    d.setZero();
    v = -g_;
    u = v;
    j = 1;

    while (v.norm() > e * g_.norm()) {
      if (u.transpose() * Gamau(u, x_) <= 0.0) {
        if (j == 1) {
          d = -g_;
        }
        break;
      }

      // cout << "v = " << v << endl;
      // cout << "d = " << d << endl;

      t_ = v.norm() * v.norm() / (u.transpose() * Gamau(u, x_));

      d = d + t_ * u;
      v1 = v - t_ * Gamau(u, x_);

      // cout << "v1 = " << v1 << endl;
      // cout << "Gamau(u, x_) = " << Gamau(u, x_) << endl;
      double beta = v1.norm() * v1.norm() / (v.norm() * v.norm() + 1e-4);

      v = v1;
      u = v + beta * u;
      j++;

      // cout << "t_ = " << t_ << endl;
      // cout << "beta = " << beta << endl;
      // cout << "v = " << v << endl;
      // cout << "u = " << u << endl;
      // cout << "d = " << d << endl << endl;

      if (j > 100) {
        break;
      }
    }

    d_ = d;

    t_ = LOLineSearch(d_, x_, g_, param_);

    dx_ = t_ * d_;
    x_ += dx_;

    f_ = problem_ptr_->GetCost(x_);
    g_ = problem_ptr_->GetGradient(x_);
    DebugInfo();

    if (iter_ >= param_.max_iter) {
      break;
    }
  }

  return x_;
}

Eigen::VectorXd NetonCGMethod::Gamau(const Eigen::VectorXd &u,
                                     const Eigen::VectorXd &x) {
  static const double delta = 1e-4;

  return (problem_ptr_->GetGradient(x + delta * u / 2.0) -
          problem_ptr_->GetGradient(x - delta * u / 2.0)) /
         delta;
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