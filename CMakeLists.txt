cmake_minimum_required(VERSION 3.26)

project(conjugate_gradients)

add_executable(test example/test.cpp)

# Require MPI for this project:
find_package(MPI REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(conjugate_gradients src/conjugate_gradients.cpp inc/conjugate_gradients.hpp)

target_link_libraries(test PRIVATE conjugate_gradients)
target_link_libraries(test PUBLIC MPI::MPI_CXX)