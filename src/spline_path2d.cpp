#include "spline_path2d.h"
#include <Eigen/Cholesky>
#include <cstddef>
#include <iostream>

using namespace std;

void SplinePath2d::SetPoints(std::vector<Eigen::Vector2d> &point_vec) {
  n_point_ = point_vec.size();
  if (n_point_ > 2) {
    point_vec_ = point_vec;
    n_seg_ = n_point_ - 1;
    CalCoef();
    init_flag_ = true;
  } else {
    init_flag_ = false;
  }
}

void SplinePath2d::CalCoef() {
  // A will change only if init or point size change
  static Eigen::MatrixXd A;
  if (!init_flag_ || (static_cast<size_t>(A.rows()) != 2 * n_seg_)) {
    A.resize(2 * n_seg_, 2 * n_seg_);
    A.setZero();

    Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d I2_4 = 4.0 * I2;

    for (size_t i = 0; i < n_seg_; ++i) {
      if (i == 0) {
        A.block(0, 0, 2, 2) = I2_4;
        A.block(0, 2, 2, 2) = I2;
      } else if (i == n_seg_ - 1) {
        A.block(2 * (n_seg_ - 1), 2 * (n_seg_ - 2), 2, 2) = I2;
        A.block(2 * (n_seg_ - 1), 2 * (n_seg_ - 1), 2, 2) = I2_4;
      } else {
        A.block(2 * i, 2 * (i - 1), 2, 2) = I2;
        A.block(2 * i, 2 * i, 2, 2) = I2_4;
        A.block(2 * i, 2 * (i + 1), 2, 2) = I2;
      }
    }
  }

  // cout << "A = " << A << endl;

  Eigen::MatrixXd B;
  B.resize(2 * n_seg_, 1);
  B.setZero();

  for (size_t i = 0; i < n_seg_; ++i) {
    B.block(2 * i, 0, 2, 1) = 3.0 * (point_vec_[i + 2] - point_vec_[i]);
  }

  // cout << "B = " << B << endl;

  Eigen::MatrixXd D;
  D.resize(2 * n_seg_, 1);

  D = A.llt().solve(B);

  // cout << "D = " << D << endl;

  seg_coef_vec_.reserve(n_seg_);
  seg_coef_vec_.clear();
  SplinePath2dCoef coef;

  Eigen::Vector2d Di;
  Eigen::Vector2d Di1;
  for (size_t i = 0; i < n_seg_; ++i) {
    // cout << "-------- i = " << i << endl;
    if (i == 0) {
      Di.setZero();
      Di1 = D.block(0, 0, 2, 1);
    } else if (i == n_seg_ - 1) {
      Di = D.block(2 * (n_seg_ - 1), 0, 2, 1);
      Di1.setZero();
    } else {
      Di = D.block(2 * (i - 1), 0, 2, 1);
      Di1 = D.block(2 * i, 0, 2, 1);
    }

    coef.a = point_vec_[i];
    coef.b = Di;
    coef.c = 3.0 * (point_vec_[i + 1] - point_vec_[i]) - 2.0 * Di - Di1;
    coef.d = 2.0 * (point_vec_[i] - point_vec_[i + 1]) + Di + Di1;

    // cout << "Di = " << Di << endl;
    // cout << "Di1 = " << Di1 << endl;

    seg_coef_vec_.emplace_back(coef);
  }
}

Eigen::Vector2d SplinePath2d::operator()(size_t i, double s) const {
  if (i > n_seg_ - 1) {
    i = n_seg_ - 1;
  }

  Eigen::Vector2d point = seg_coef_vec_[i].a + seg_coef_vec_[i].b * s +
                          seg_coef_vec_[i].c * s * s +
                          seg_coef_vec_[i].d * s * s * s;

  return point;
}