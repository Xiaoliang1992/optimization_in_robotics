#include "matplotlibcpp.h"
#include "problem.h"
#include "solver.h"
#include <chrono>
#include <memory>

using namespace std;
using namespace optimization_solver;
using namespace matplotlibcpp;

int main() {

  shared_ptr<Solver> solver_ptr = make_shared<QuasiNewtonsMethod>();

  auto problem_type = ProblemType::PRosenbrock;
  solver_ptr->SetProblem(problem_type);

  Eigen::VectorXd x;
  switch (problem_type) {
  case ProblemType::PRosenbrock:
    x.resize(kRosenbrockN);
    break;
  case ProblemType::Example1:
    x.resize(2);
    break;
  case ProblemType::Example2:
    x.resize(2);
    break;
  }

  x << -0.5, -0.5;

  auto t_start = std::chrono::system_clock::now();
  x = solver_ptr->Solve(x);
  auto t_end = std::chrono::system_clock::now();

  long int elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
          .count();
  double t_cost = elapsed / 1e6;

  cout << "solution = \n"
       << x << "\niter = " << solver_ptr->GetIter()
       << ", \ntime cost = " << t_cost << " ms" << endl;

  figure();
  plot(solver_ptr->GetInfoPtr()->iter_vec, solver_ptr->GetInfoPtr()->obj_val);
  show();

  return 0;
}