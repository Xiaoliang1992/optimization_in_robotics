#include "matplotlibcpp.h"
#include "problem.h"
#include "solver.h"
#include <chrono>
#include <memory>

using namespace std;
using namespace optimization_solver;
using namespace matplotlibcpp;

int main() {
  auto solver_type = SolverType::TQuasiNewtonsMethod;
  auto problem_type = ProblemType::Example2;
  SolverParameters param;
  param.debug_enable = true;

  shared_ptr<Solver> solver_ptr = make_shared<Solver>();

  solver_ptr->SetSolver(solver_type);
  solver_ptr->SetProblem(problem_type);
  solver_ptr->SetParam(param);

  Eigen::VectorXd x0, x;
  x0.resize(solver_ptr->GetInfoPtr()->problem_size);
  x.resize(solver_ptr->GetInfoPtr()->problem_size);

  x0.setOnes();
  x0 = x0 * (-0.0);

  auto t_start = std::chrono::system_clock::now();
  x = solver_ptr->Solve(x0);
  auto t_end = std::chrono::system_clock::now();

  long int elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
          .count();
  double t_cost = elapsed / 1e6;

  cout << "solution = \n"
       << x << "\niter = " << solver_ptr->GetInfoPtr()->iter
       << ", \ntime cost = " << t_cost << " ms" << endl;

  //   solver_ptr->SetSolver(SolverType::TGradientDescent);
  //   solver_ptr->SetProblem(problem_type);
  //   x = solver_ptr->Solve(x0);

  //   cout << "solution = \n"
  //        << x << "\niter = " << solver_ptr->GetInfoPtr()->iter
  //        << ", \ntime cost = " << t_cost << " ms" << endl;

  figure();
  plot(solver_ptr->GetInfoPtr()->iter_vec, solver_ptr->GetInfoPtr()->g_norm_vec);
  show();

  return 0;
}