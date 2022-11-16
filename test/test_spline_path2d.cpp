#include "spline_path2d.h"
#include <iostream>
using namespace std;

int main() {

  std::vector<Eigen::Vector2d> point_vec;
  point_vec.emplace_back(Eigen::Vector2d(0.0, 0.0));
  point_vec.emplace_back(Eigen::Vector2d(1.0, 1.0));
  point_vec.emplace_back(Eigen::Vector2d(2.0, 3.0));
  point_vec.emplace_back(Eigen::Vector2d(3.0, 5.0));
  SplinePath2d spline_test(point_vec);

  point_vec.emplace_back(Eigen::Vector2d(4.0, 9.0));

  spline_test.SetPoints(point_vec);

  cout << "seg0(0) = \n" << spline_test(0, 0.0) << endl;
  cout << "seg0(1) = \n" << spline_test(0, 1.0) << endl;

  cout << "seg1(0) = \n" << spline_test(1, 0.0) << endl;
  cout << "seg1(1) = \n" << spline_test(1, 1.0) << endl;

  cout << "seg2(0) = \n" << spline_test(2, 0.0) << endl;
  cout << "seg2(1) = \n" << spline_test(2, 1.0) << endl;

  cout << "seg3(0) = \n" << spline_test(3, 0.0) << endl;
  cout << "seg3(1) = \n" << spline_test(3, 1.0) << endl;

  return 0;
}