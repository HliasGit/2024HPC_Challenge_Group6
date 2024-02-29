#ifndef CG
#define CG

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <iostream>

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out);
bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols);
void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file);
double dot(const double * x, const double * y, size_t size, int world_rank);
void axpby(double alpha, const double * x, double beta, double * y, size_t size);
void axpbyP(double alpha, const double * x, double beta, double * y, size_t size);
void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols);
void gemvP(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols);
void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error);
double dotP(const double * x, const double * y, size_t size);
#endif