cmake_minimum_required(VERSION 3.10)
project(optimization_in_robotics)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/include)

file(GLOB_RECURSE CPP_FILES src/*.cpp)

add_library (${PROJECT_NAME} SHARED ${CPP_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE)

add_executable(test_problem ${CMAKE_CURRENT_LIST_DIR}/test/test_problem.cpp)
target_link_libraries(test_problem PRIVATE ${PROJECT_NAME})

add_executable(test_solver ${CMAKE_CURRENT_LIST_DIR}/test/test_solver.cpp)
target_link_libraries(test_solver PRIVATE ${PROJECT_NAME})
