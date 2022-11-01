#include "matplotlibcpp.h"
#include "problem.h"
#include "solver.h"
#include <chrono>
#include <memory>

using namespace std;
using namespace optimization_solver;
using namespace matplotlibcpp;

int main() {
  auto solver_type = SolverType::TNetonCGMethod;
  auto problem_type = ProblemType::Example4;

  shared_ptr<Solver> solver_ptr = make_shared<Solver>();

  solver_ptr->SetSolver(solver_type);
  solver_ptr->SetProblem(problem_type);

  Eigen::VectorXd x;
  x.resize(solver_ptr->GetInfoPtr()->problem_size);

  x.setOnes();
  x = x * (-0.5);

  auto t_start = std::chrono::system_clock::now();
  x = solver_ptr->Solve(x);
  auto t_end = std::chrono::system_clock::now();

  long int elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
          .count();
  double t_cost = elapsed / 1e6;

  cout << "solution = \n"
       << x << "\niter = " << solver_ptr->GetInfoPtr()->iter
       << ", \ntime cost = " << t_cost << " ms" << endl;

  figure();
  plot(solver_ptr->GetInfoPtr()->iter_vec, solver_ptr->GetInfoPtr()->obj_val);
  show();

  return 0;
}