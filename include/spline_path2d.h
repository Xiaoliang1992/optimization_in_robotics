#ifndef __SPLINE_PATH2D_H__
#define __SPLINE_PATH2D_H__

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <vector>

class SplinePath2d {

  struct SplinePath2dCoef {
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    Eigen::Vector2d c;
    Eigen::Vector2d d;
  };

public:
  SplinePath2d() {}
  SplinePath2d(std::vector<Eigen::Vector2d> &point_vec) {
    SetPoints(point_vec);
  }
  ~SplinePath2d() {}
  void SetPoints(std::vector<Eigen::Vector2d> &point_vec);
  Eigen::Vector2d operator()(size_t i, double s) const;

  std::vector<SplinePath2dCoef> &GetCoefVec() { return seg_coef_vec_; }

private:
  void CalCoef();
  bool init_flag_ = false;
  size_t n_seg_ = 0;
  size_t n_point_ = 0;
  std::vector<Eigen::Vector2d> point_vec_;
  std::vector<SplinePath2dCoef> seg_coef_vec_;
};

#endif