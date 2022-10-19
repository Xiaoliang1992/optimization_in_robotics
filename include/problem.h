#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <eigen3/Eigen/Core>
#include <iostream>

static const int N = 4;

namespace optimization_solver {
enum ProblemType {
  Example1 = 0,
  PRosenbrock = 1,
};

class Problem {
public:
  virtual double GetObjective(const Eigen::VectorXd &x) = 0;
  virtual Eigen::VectorXd GetGradient(const Eigen::VectorXd &x);
  virtual Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x);
  virtual Eigen::VectorXd GetDiffGradient(const Eigen::VectorXd &x) final;
  virtual Eigen::MatrixXd GetDiffHessian(const Eigen::VectorXd &x) final;

protected:
  int size_ = 0;
};

class RosenbrockFunction : public Problem {
public:
  RosenbrockFunction() { size_ = N; }
  double GetObjective(const Eigen::VectorXd &x) override;
  Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

class Example1Func : public Problem {
public:
  Example1Func() { size_ = 2; }
  double GetObjective(const Eigen::VectorXd &x) override;
  // Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  // Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

}; // namespace optimization_solver

#endif