#include "problem.h"
#include <cmath>
#include <cstddef>

using namespace std;
namespace optimization_solver {
static const double eps_value = 1e-8;

// base problem
Eigen::VectorXd Problem::GetDiffGradient(const Eigen::VectorXd &x) {
  int size = x.size();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(size);
  Eigen::VectorXd dxi = Eigen::VectorXd::Zero(size);

  for (int i = 0; i < size; ++i) {
    dxi(i) = eps_value;
    g(i) = (GetCost(x + dxi) - GetCost(x - dxi)) / (2.0 * eps_value);
    dxi(i) = 0.0;
  }
  return g;
}

Eigen::MatrixXd Problem::GetDiffHessian(const Eigen::VectorXd &x) {
  int size = x.size();
  Eigen::MatrixXd h(size, size);
  Eigen::VectorXd dxi = Eigen::VectorXd::Zero(size);
  Eigen::VectorXd dxj = Eigen::VectorXd::Zero(size);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      dxi(i) = eps_value;
      dxj(j) = eps_value;
      h(i, j) = (GetCost(x + dxi + dxj) - GetCost(x - dxi + dxj) -
                 GetCost(x + dxi - dxj) + GetCost(x - dxi - dxj)) /
                (4.0 * eps_value * eps_value);
      dxi(i) = 0.0;
      dxj(j) = 0.0;
    }
  }

  return h;
}

Eigen::VectorXd Problem::GetGradient(const Eigen::VectorXd &x) {
  return GetDiffGradient(x);
}

Eigen::MatrixXd Problem::GetHessian(const Eigen::VectorXd &x) {
  return GetDiffHessian(x);
}

// RosenbrockFunction problem
double RosenbrockFunction::GetCost(const Eigen::VectorXd &x) {
  if (x.size() != kRosenbrockN) {
    std::cout
        << "x.size() is not equal to kRosenbrockN!\nPlease check input size!"
        << std::endl;
  }

  double s = 0.0;
  for (size_t i = 1; i <= kRosenbrockN / 2; ++i) {
    double x_2i1 = x(2 * i - 2);
    double x_2i = x(2 * i - 1);
    s += 100.0 * std::pow(x_2i1 * x_2i1 - x_2i, 2) + std::pow(x_2i1 - 1.0, 2);
  }

  return s;
}

Eigen::VectorXd RosenbrockFunction::GetGradient(const Eigen::VectorXd &x) {
  auto size = x.size();
  Eigen::VectorXd g(size);

  for (size_t i = 1; i <= kRosenbrockN / 2; ++i) {
    g(2 * i - 2) = 400.0 * std::pow(x(2 * i - 2), 3) -
                   400.0 * x(2 * i - 2) * x(2 * i - 1) + 2.0 * x(2 * i - 2) -
                   2.0;
    g(2 * i - 1) = -200.0 * std::pow(x(2 * i - 2), 2) + 200.0 * x(2 * i - 1);
  }

  return g;
}

Eigen::MatrixXd RosenbrockFunction::GetHessian(const Eigen::VectorXd &x) {
  auto size = x.size();
  Eigen::MatrixXd H(size, size);
  H.setZero();

  for (size_t i = 1; i <= kRosenbrockN / 2; ++i) {
    H(2 * i - 2, 2 * i - 2) =
        1200.0 * std::pow(x(2 * i - 2), 2) - 400.0 * x(2 * i - 1) + 2.0;
    H(2 * i - 2, 2 * i - 1) = -400.0 * x(2 * i - 2);

    H(2 * i - 1, 2 * i - 1) = 200.0;
    H(2 * i - 1, 2 * i - 2) = -400.0 * x(2 * i - 2);
  }

  return H;
}

// example1: f(x1, x2) = exp(x1 + 3x2 - 0.1) + exp(x1 - 3x2 - 0.1) + exp(-x1 -
// 0.1)
double Example1Func::GetCost(const Eigen::VectorXd &x) {
  double y = std::exp(x(0) + 3.0 * x(1) - 0.1) +
             std::exp(x(0) - 3.0 * x(1) - 0.1) + std::exp(-x(0) - 0.1);

  return y;
}

// example2: f(x1, x2) = (1 - x1)^2 + (x2 - x1^2)^2
double Example2Func::GetCost(const Eigen::VectorXd &x) {
  double y = std::pow(1.0 - x(0), 2) + std::pow(x(1) - x(0) * x(0), 2);
  return y;
}

// example3: nonsmooth
double Example3Func::GetCost(const Eigen::VectorXd &x) {
  double y = (1.0 - x(0)) * (1.0 - x(0)) + std::abs(x(1) - x(0) * x(0));
  return y;
}

Eigen::VectorXd Example3Func::GetGradient(const Eigen::VectorXd &x) {
  auto size = x.size();
  Eigen::VectorXd g(size);
  if (x(1) - x(0) * x(0) >= 0.0) {
    g << -2.0, 1.0;
  } else {
    g << -2.0 + 4.0 * x(0), -1.0;
  }
  return g;
}

} // namespace optimization_solver