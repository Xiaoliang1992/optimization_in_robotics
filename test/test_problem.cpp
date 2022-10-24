#include "problem.h"
#include "solver.h"
#include <memory>

using namespace std;
using namespace optimization_solver;

int main() {

  shared_ptr<Problem> problem_ptr = make_shared<RosenbrockFunction>();
  Eigen::VectorXd x(kRosenbrockN);

  for (int i = 0; i < 30; ++i) {

    x.setRandom();

    problem_ptr->GetCost(x);

    cout << "norm(GetDiffGradient - GetGradient) = "
         << (problem_ptr->GetDiffGradient(x) - problem_ptr->GetGradient(x))
                .norm()
         << endl;

    cout << "norm(GetDiffHessian - GetHessian) = "
         << (problem_ptr->GetDiffHessian(x) - problem_ptr->GetHessian(x))
                .norm()
         << endl;
  }

  return 0;
}