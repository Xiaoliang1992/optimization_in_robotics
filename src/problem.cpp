#include "problem.h"
#include <cmath>
#include <cstddef>

namespace optimization_solver {
static const double eps_value = 1e-5;

// base problem
Eigen::VectorXd Problem::GetDiffGradient(const Eigen::VectorXd &x) {
  int size = x.size();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(size);
  Eigen::VectorXd dxi = Eigen::VectorXd::Zero(size);

  for (int i = 0; i < size; ++i) {
    dxi(i) = eps_value;
    g(i) = (GetObjective(x + dxi) - GetObjective(x - dxi)) / (2.0 * eps_value);
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
      h(i, j) = (GetObjective(x + dxi + dxj) - GetObjective(x - dxi + dxj) -
                 GetObjective(x + dxi - dxj) + GetObjective(x - dxi - dxj)) /
                (4.0 * eps_value * eps_value);
      dxi(i) = 0.0;
      dxj(j) = 0.0;
    }
  }

  return h;
}

// RosenbrockFunction problem
double RosenbrockFunction::GetObjective(const Eigen::VectorXd &x) {
  if (x.size() != N) {
    std::cout << "x.size() is not equal to N!\nPlease check input size!"
              << std::endl;
  }

  double s = 0.0;
  for (size_t i = 1; i <= N / 2; ++i) {
    double x_2i1 = x(2 * i - 2);
    double x_2i = x(2 * i - 1);
    s += 100.0 * std::pow(x_2i1 * x_2i1 - x_2i, 2) + std::pow(x_2i1 - 1.0, 2);
  }

  return s;
}

Eigen::VectorXd RosenbrockFunction::GetGradient(const Eigen::VectorXd &x) {
  int size = x.size();
  Eigen::VectorXd g(size);

  for (size_t i = 1; i <= N / 2; ++i) {
    g(2 * i - 2) = 400.0 * std::pow(x(2 * i - 2), 3) -
                   400.0 * x(2 * i - 2) * x(2 * i - 1) + 2.0 * x(2 * i - 2) -
                   2.0;
    g(2 * i - 1) = -200.0 * std::pow(x(2 * i - 2), 2) + 200.0 * x(2 * i - 1);
  }

  return g;
}

} // namespace optimization_solver