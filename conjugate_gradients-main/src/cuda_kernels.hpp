#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

const double cuda_dot(const double * x, const double * y, size_t size, cublasHandle_t * handle)
{
    // Scalar product
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

const void cuda_axpby(double alpha, const double * x, double beta, double * y, size_t size, cublasHandle_t * handle)
{
    // y = alpha * x + beta * y
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



const void cuda_gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols, cublasHandle_t * handle)
{
    // y = alpha * A * x + beta * y;
    // Kernel CUDA
    // cublasDgemv(...)
    
}
