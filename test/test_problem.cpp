#include "problem.h"
#include "solver.h"
#include <memory>

using namespace std;
using namespace optimization_solver;

int main() {

  shared_ptr<Problem> problem_ptr = make_shared<RosenbrockFunction>();
  Eigen::VectorXd x(N);

  for (int i = 0; i < 30; ++i) {

    x.setRandom();

    problem_ptr->GetObjective(x);

    cout
        << "norm(GetNumGradient - GetGradient) = "
        << (problem_ptr->GetNumGradient(x) - problem_ptr->GetGradient(x)).norm()
        << endl;
  }

  return 0;
}