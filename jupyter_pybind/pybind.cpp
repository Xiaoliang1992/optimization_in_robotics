#include "problem.h"
#include "solver.h"
#include <functional>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);

namespace py = pybind11;
using namespace optimization_solver;

class EigenClass {
public:
  Eigen::VectorXd GetVectorXd(int size) {
    Eigen::VectorXd x0{};
    x0.resize(size);
    return x0;
  }
};

PYBIND11_MODULE(solver_py, m) {
  m.doc() = "bind solver";
  py::bind_vector<std::vector<Eigen::VectorXd>>(m, "VectorXdvec");

  py::class_<EigenClass>(m, "EigenClass")
      .def(py::init<>())
      .def("GetVectorXd", &EigenClass::GetVectorXd);

  py::class_<SolverParameters>(m, "SolverParameters")
      .def(py::init<>())
      .def_readwrite("max_iter", &SolverParameters::max_iter)
      .def_readwrite("linesearch_method", &SolverParameters::linesearch_method)
      .def_readwrite("c1", &SolverParameters::c1)
      .def_readwrite("c2", &SolverParameters::c2)
      .def_readwrite("t0", &SolverParameters::t0)
      .def_readwrite("terminate_threshold",
                     &SolverParameters::terminate_threshold)
      .def_readwrite("m_size", &SolverParameters::m_size)
      .def_readwrite("debug_enable", &SolverParameters::debug_enable);

  py::class_<Solver>(m, "Solver")
      .def(py::init<>())
      .def("SetSolver", &Solver::SetSolver)
      .def("SetProblem", &Solver::SetProblem)
      .def("SetParam", &Solver::SetParam)
      .def("Solve", &Solver::Solve)
      .def("GetInfo", &Solver::GetInfo);

  py::class_<SolverDebugInfo>(m, "SolverDebugInfo")
      .def(py::init<>())
      .def_readwrite("iter", &SolverDebugInfo::iter)
      .def_readwrite("problem_size", &SolverDebugInfo::problem_size)
      .def_readwrite("x_vec", &SolverDebugInfo::x_vec)
      .def_readwrite("dx_vec", &SolverDebugInfo::dx_vec)
      .def_readwrite("g_vec", &SolverDebugInfo::g_vec)
      .def_readwrite("tau_vec", &SolverDebugInfo::tau_vec)
      .def_readwrite("g_norm_vec", &SolverDebugInfo::g_norm_vec)
      .def_readwrite("iter_vec", &SolverDebugInfo::iter_vec)
      .def_readwrite("cost_vec", &SolverDebugInfo::cost_vec)
      .def("Clear", &SolverDebugInfo::Clear);
}
