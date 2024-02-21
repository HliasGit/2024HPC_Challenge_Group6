cmake_minimum_required(VERSION 3.26)

project(conjugate_gradients)

add_executable(test example/test.cpp)

# Require MPI for this project:
find_package(MPI REQUIRED)

add_library(conjugate_gradients src/conjugate_gradients.cpp inc/conjugate_gradients.hpp)

target_link_libraries(test PRIVATE conjugate_gradients)
target_link_libraries(test PUBLIC MPI::MPI_CXX)