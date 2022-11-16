#include "spline_problem.h"
#include <iostream>
using namespace std;
using namespace optimization_solver;

int main() {

  std::vector<Eigen::Vector2d> point_vec;
  point_vec.emplace_back(Eigen::Vector2d(0.0, 0.0));
  point_vec.emplace_back(Eigen::Vector2d(1.0, 1.0));
  point_vec.emplace_back(Eigen::Vector2d(2.0, 3.0));
  point_vec.emplace_back(Eigen::Vector2d(3.0, 5.0));
  point_vec.emplace_back(Eigen::Vector2d(4.0, 9.0));
  SplinePath2d spline_test(point_vec);

  SplineProblem spline_problem;

  spline_problem.Init(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(4.0, 9.0),
                      point_vec.size());
  Eigen::VectorXd x;

  spline_problem.GetXFromPoints(x, point_vec);
  cout << "x = " << x << endl;

  std::vector<Eigen::Vector2d> point_vec2;
  spline_problem.GetPointsFromX(point_vec2, x);

  cout << "point_vec2 =" << endl;
  for (size_t i = 0; i < point_vec2.size(); ++i) {
    cout << point_vec2[i] << endl;
  }

  point_vec[0];
}