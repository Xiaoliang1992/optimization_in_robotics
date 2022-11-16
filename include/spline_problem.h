#include "problem.h"
#include "spline_path2d.h"
#include <cstddef>

namespace optimization_solver {
class SplineProblem : public Problem {
public:
  void Init(const Eigen::Vector2d &p0, const Eigen::Vector2d &pn,
            const size_t n_point);
  double GetCost(const Eigen::VectorXd &x) override;
  // Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  // Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

  void GetPointsFromX(std::vector<Eigen::Vector2d> &point_vec,
                      const Eigen::VectorXd &x);

  void GetXFromPoints(Eigen::VectorXd &x,
                      const std::vector<Eigen::Vector2d> &point_vec);

private:
  size_t n_point_ = 0;
  size_t n_seg_ = 0;
  Eigen::Vector2d p0_;
  Eigen::Vector2d pn_;
  std::vector<Eigen::Vector2d> point_vec_;
  SplinePath2d spline_func_;
};
} // namespace optimization_solver