#include <cstdio>
#include <cstdlib>
#include <cmath>
// #include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define BLOCK_DIM_VET 32

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



double dot(const double * x, const double * y, size_t size, cublasHandle_t handle)
{
    // Scalar product
    double result;
    cublasDdot(handle, size, x, size, y, size, &result);
    return result;
}



void axpby(double * alpha, const double * x, double * beta, double * y, size_t size, cublasHandle_t handle)
{
    // y = alpha * x + beta * y
    cublasDaxpy(handle, size, alpha, x, size, y, size);
}



void gemv(double * alpha, const double * A, const double * x, double * beta, double * y, size_t size, cublasHandle_t handle)
{
    // y = alpha * A * x + beta * y;
    cublasDgemv(handle, CUBLAS_OP_N, size, size, alpha, A, size, x, size, beta, y, size);
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
    *z = *x / *y;
}
__global__ void inv_alpha(const double * alpha, double * alpha_inv)
{
    *alpha_inv = *alpha;
}
void conjugate_gradients(const double * d_A, double * d_x, double * d_p, double * d_r,  size_t size, int max_iters, double rel_error)
{
    // Create paramters
    double* d_beta;
    double* d_alpha; 
    double* d_alpha_;
    double* a;
    double* b; 
	double* d_rr_new;
	double* d_rr;
	double* d_bb;
    double* d_Ap;
    

    double * tmp;
    // Relative residual
    double h_bb;
    double h_rr;

    int num_iters;
    cudaMalloc((void **) &d_Ap, size * sizeof(double));
	cudaMalloc((void **) &d_beta, sizeof(double));
    cudaMalloc((void **) &tmp, sizeof(double));
    cudaMalloc((void **) &a, sizeof(double));
    cudaMalloc((void **) &b, sizeof(double));
	cudaMalloc((void **) &d_alpha, sizeof(double));
    cudaMalloc((void **) &d_alpha_, sizeof(double));
	cudaMalloc((void **) &d_rr_new, sizeof(double));
	cudaMalloc((void **) &d_rr, sizeof(double));
	cudaMalloc((void **) &d_bb, sizeof(double));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    dim3 vec_block_dim(BLOCK_DIM_VET);
	dim3 vec_grid_dim((size + BLOCK_DIM_VET - 1) / BLOCK_DIM_VET);
    // Set alpha = 1 and beta = 0
    init_alpha_beta<<<1, 1>>>(a, b);
    // Init solver with x0 = 0 and p = r
    init_cg_solver <<<vec_grid_dim, vec_block_dim>>> (d_x, d_p, d_r, size);
    // Coompute r * r = b*b
    cublasDdot(handle, size, d_r, size, d_r, size, d_rr);
    copy_value <<<1,1>>> (d_bb, d_rr);

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // alpha(k) = rr / p A p
        cublasDgemv(handle, CUBLAS_OP_N, size, size, a, d_A, size, d_p, size, b, d_Ap, size);
        cublasDdot(handle, size, d_p, size, d_Ap, size, tmp);
        a_frac_b<<<1, 1>>> (d_rr, tmp, d_alpha);
        
        inv_alpha<<<1,1>>> (d_alpha, d_alpha_);

        // x_(k+1) = x(k) + alpha * p
        cublasDaxpy(handle, size, d_alpha, d_p, size, d_x, size);
        //r(k+1) = r(k) - alpha * A * p
        cublasDaxpy(handle, size, d_alpha_, d_Ap, size, d_r, size);
        // beta(k) = r(k+1)r(k+1) / r(k)r(k)
        cublasDdot(handle, size, d_r, size, d_r, size, d_rr_new);
        a_frac_b <<<1, 1>>> (d_rr_new, d_rr, d_beta);

        // Update d_rr
        copy_value <<<1, 1>>> (d_rr, d_rr_new);

        // Compute relative residual
        a_frac_b <<<1, 1>>> (d_rr, d_bb, tmp);
        cudaMemcpy(d_rr, &h_rr, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_bb, &h_bb, sizeof(double), cudaMemcpyDeviceToHost);
        // Stopping criteria
        if(std::sqrt(h_rr / h_bb) < rel_error) 
            break; 
        // p(k+1) = r(k+1) + beta * p
        cublasDaxpy(handle, size, d_beta, d_p, size, d_r, size);
    }

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(h_rr / h_bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(h_rr / h_bb));
    }
}

int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");



    double * matrix;
    double * rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Done\n");
        printf("\n");

        if(matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if(rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if(rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        size = matrix_rows;
    }

    printf("Solving the system ...\n");
    double * sol = new double[size];

    // allocate device memory
	double* d_A;
	double* d_b;
	double* d_x;
	double* d_p;
	double* d_r;
	double* d_temp;
	cudaMalloc((void **) &d_A, size * size * sizeof(double));
	cudaMalloc((void **) &d_b, size * sizeof(double));
	cudaMalloc((void **) &d_x, size * sizeof(double));
	cudaMalloc((void **) &d_p, size * sizeof(double));
	cudaMalloc((void **) &d_r, size * sizeof(double));
	cudaMalloc((void **) &d_temp, size * sizeof(double));

    // copy host memory to device
	cudaMemcpy(d_A, matrix, size * size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, sol, size * sizeof(double), cudaMemcpyHostToDevice);
	// assume x0 = 0
	cudaMemcpy(d_p, rhs, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, size * sizeof(double), cudaMemcpyHostToDevice);

    conjugate_gradients(d_A, d_x, d_p, d_r, size, max_iters, rel_error);
    printf("Done\n");
    printf("\n");

    printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
    if(!success_write_sol)
    {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }
    printf("Done\n");
    printf("\n");

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    // Clean device memory
    // cleanup memory device
	cudaFree(d_A);
	cudaFree(d_b);
	cudaFree(d_x);
	cudaFree(d_p);
	cudaFree(d_r);

    printf("Finished successfully\n");

    return 0;
}