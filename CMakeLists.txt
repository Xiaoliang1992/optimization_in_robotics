cmake_minimum_required(VERSION 3.10)
project(optimization_in_robotics)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_PYTHON_VERSION 3.9)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED PATHS ~/miniconda3/lib/python3.9/site-packages/pybind11/share/cmake/pybind11)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/include)

file(GLOB_RECURSE CPP_FILES src/*.cpp)

add_library (${PROJECT_NAME} SHARED ${CPP_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE)
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)

add_executable(test_problem ${CMAKE_CURRENT_LIST_DIR}/test/test_problem.cpp)
target_link_libraries(test_problem PRIVATE ${PROJECT_NAME})

add_executable(test_solver ${CMAKE_CURRENT_LIST_DIR}/test/test_solver.cpp)
target_link_libraries(test_solver PRIVATE ${PROJECT_NAME} ${PYTHON_LIBRARIES})

add_executable(test_spline_path2d ${CMAKE_CURRENT_LIST_DIR}/test/test_spline_path2d.cpp)
target_link_libraries(test_spline_path2d PRIVATE ${PROJECT_NAME} ${PYTHON_LIBRARIES})

add_executable(test_spline_problem ${CMAKE_CURRENT_LIST_DIR}/test/test_spline_problem.cpp)
target_link_libraries(test_spline_problem PRIVATE ${PROJECT_NAME} ${PYTHON_LIBRARIES})

# pybind
pybind11_add_module(solver_py MODULE jupyter_pybind/pybind.cpp)
target_link_libraries(solver_py PRIVATE ${PROJECT_NAME})

