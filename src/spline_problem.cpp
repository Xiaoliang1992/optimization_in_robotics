#include "spline_problem.h"
#include <cstddef>
namespace optimization_solver {

void SplineProblem::Init(const Eigen::Vector2d &p0, const Eigen::Vector2d &pn,
                         const size_t n_point) {
  p0_ = p0;
  pn_ = pn;
  n_point_ = n_point;
  n_seg_ = n_point - 1;
}

void SplineProblem::GetPointsFromX(std::vector<Eigen::Vector2d> &point_vec,
                                   const Eigen::VectorXd &x) {
  point_vec.reserve(n_point_);
  point_vec.clear();

  for (size_t i = 0; i < n_point_; ++i) {
    if (i == 0) {
      point_vec.emplace_back(p0_);
    } else if (i == n_point_ - 1) {
      point_vec.emplace_back(pn_);
    } else {
      point_vec.emplace_back(x.block(2 * (i - 1), 0, 2, 1));
    }
  }
}

void SplineProblem::GetXFromPoints(
    Eigen::VectorXd &x, const std::vector<Eigen::Vector2d> &point_vec) {
  x.resize(2 * (n_point_ - 2));
  for (size_t i = 0; i < n_point_ - 2; ++i) {
    x.block(2 * i, 0, 2, 1) = point_vec[i + 1];
  }
}

// x means point from p(1) to p(n-1)
double SplineProblem::GetCost(const Eigen::VectorXd &x) {
  GetPointsFromX(point_vec_, x);

  // set point for spline2d
  spline_func_.SetPoints(point_vec_);

  double f_energy = 0.0;
  auto const &coef_vec = spline_func_.GetCoefVec();
  for (size_t i = 0; i < n_seg_; ++i) {
    f_energy += 12.0 * coef_vec[i].d.norm() +
                12.0 * coef_vec[i].c.dot(coef_vec[i].d) +
                4.0 * coef_vec[i].c.norm();
  }

  return f_energy;
}

} // namespace optimization_solver