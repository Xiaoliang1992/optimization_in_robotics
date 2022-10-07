#include "problem.h"
#include "solver.h"
#include <chrono>
#include <memory>

using namespace std;
using namespace optimization_solver;

int main() {

  shared_ptr<Solver> solver_ptr = make_shared<GradientDescent>();
  solver_ptr->SetProblem(ProblemType::PRosenbrock);
  Eigen::VectorXd x(N);
  x.setZero();

  auto t_start = std::chrono::system_clock::now();
  x = solver_ptr->Solve(x);
  auto t_end = std::chrono::system_clock::now();

  long int elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
          .count();
  double t_cost = elapsed / 1e6;

  cout << "iter = " << solver_ptr->GetIter() << ", time cost = " << t_cost
       << " ms\n"
       << "solution = \n"
       << x << endl;

  return 0;
}