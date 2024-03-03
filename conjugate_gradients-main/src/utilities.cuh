#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}



bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}



void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}
void print_vet(const double * vet, size_t num_rows, FILE * file = stdout)
{
    fprintf(file, "%z\t", num_rows);
    for(size_t r = 0; r < num_rows; r++)
    {
        double val = vet[r];
        printf("%+6.3f\t ", val);
    }
    printf("\n");
}


__global__ void init_cg_solver(double * x, double * p, const double * r, size_t size)
{   
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
    {
		x[index] = 0.0;
        p[index] = r[index];
    }
}
__global__ void init_alpha_beta(double * a, double * b)
{
    *a = 1.0;
    *b = 0.0;
}
__global__ void copy_value(double * a, const double * b)
{
    *a = *b;
}   
__global__ void a_frac_b(const double * x, const double * y, double * z)
{
    *z = (*x) / (*y);
}
__global__ void inv_alpha(const double * alpha, double * alpha_inv)
{
    *alpha_inv = - (*alpha);
}

__global__ void scalar_vet(const double * alpha, double * x, size_t size)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
    {
		x[index] = (*alpha) * x[index];
    }
}

void print_matrix_mcpy(const double * d_A, size_t rows, size_t columns)
{   

    if (columns > 0)
    {
        double *h_A = (double *) malloc(rows * columns * sizeof(double));
        cudaMemcpy(h_A, d_A, rows*columns, cudaMemcpyDeviceToHost);
        print_matrix(h_A, rows, columns);
    }
    else
    {
        double *h_A = (double *) malloc(rows * sizeof(double));
        cudaError_t err = cudaMemcpy(h_A, d_A, rows * sizeof(double), cudaMemcpyDeviceToHost);
        print_vet(h_A, rows);
    }
}
void print_scalar(const double * scalar)
{
        double *h_A = (double *) malloc(sizeof(double));
        cudaError_t err = cudaMemcpy(h_A, scalar, sizeof(double), cudaMemcpyDeviceToHost);
        printf("%+6.5f\n", *h_A);
        
}