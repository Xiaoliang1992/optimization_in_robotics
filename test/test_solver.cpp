#include "matplotlibcpp.h"
#include "problem.h"
#include "solver.h"
#include <chrono>
#include <memory>

using namespace std;
using namespace optimization_solver;
using namespace matplotlibcpp;

int main() {

  shared_ptr<Solver> solver_ptr = make_shared<NewtonsMethod>();
  solver_ptr->SetProblem(ProblemType::Example1);
  Eigen::VectorXd x(N);
  x.setZero();

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