#include "cuda_kernels.hpp"

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

bool conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{
    double alpha, beta, bb, rr, rr_new;
    const int tot_size = size * size;
    constexpr int size_byte = sizeof(double);
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];

    int num_iters;

    // Init GPU parameters
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cudaError_t err_memcpy;
    cublasHandle_t handler;
    double *devPtrA, *devPtr_r, *devPtr_p;
    double *devPtr_alpha, *devPtr_beta, *devPtr_bb, *devPtr_rr;

    // Allocate device memory: A, p, r 
    cudaStat = cudaMalloc ((void**)&devPtrA, tot_size * size_byte);
    cudaStat = cudaMalloc ((void*)&devPtr_p, size * size_byte);
    cudaStat = cudaMalloc ((void*)&devPtr_r, size * size_byte)

    cudaStat = cudaMalloc ((void*)&devPtr_alpha, size_byte);
    cudaStat = cudaMalloc ((void*)&devPtr_beta, size_byte);
    cudaStat = cudaMalloc ((void*)&devPtr_bb, size_byte);
    cudaStat = cudaMalloc ((void*)&devPtr_rr, size_byte);
    if (cudaStat != cudaSuccess)
    {
        printf("cudaMalloc doesn't work.");
        return 0;
    }
    // Create handle
    stat = cublasCreate(&handler);
    if (stat != CUBLAS_STATUS_SUCCESS) 
    {
        printf ("CUBLAS initialization failed\n");
        return false;
    }
    // Move data to the device
    stat = cublasSetMatrix (size, size, size_byte, A, size, devPtrA, size);
    stat = cublasSetVector (size, size_byte, p, 0, devPtr_p, 0);
    stat = cublasSetVector (size, size_byte, r, 0, devPtr_r, 0);
    err_memcpy = cudaMemcpy(devPtr_alpha, &alpha, size_byte, cudaMemcpyHostToDevice);
    err_memcpy = cudaMemcpy(devPtr_beta, &beta, size_byte, cudaMemcpyHostToDevice);
    err_memcpy = cudaMemcpy(devPtr_bb, &b, size_byte, cudaMemcpyHostToDevice);
    err_memcpy = cudaMemcpy(devPtr_rr, &rr, size_byte, cudaMemcpyHostToDevice);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Data download failed");
        cudaFree (devPtrA);
        cudaFree (devPtr_p);
        cudaFree (devPtr_r);
        cublasDestroy(handler);
        return false;
    }

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = cuda_dot(b, b, size, &handler);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        cuda_gemv(1.0, A, p, 0.0, Ap, size, size, &handler);
        alpha = rr / cuda_dot(p, Ap, size, &handler);
        cuda_axpby(alpha, p, 1.0, x, size, &handler);
        cuda_axpby(-alpha, Ap, 1.0, r, size, &handler);
        rr_new = cuda_dot(r, r, size, &handler);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        cuda_axpby(1.0, r, beta, p, size, &handler);
    }

    // Free device space
    cudaFree (devPtrA);
    cudaFree (devPtr_p);
    cudaFree (devPtr_r);
    cublasDestroy(handler);
    
    delete[] r;
    delete[] p;
    delete[] Ap;

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
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

    const bool cg_err = conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);
    if (cg_err)
    {
        printf("CG failed.");
        return 0;
    }
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

    printf("Finished successfully\n");

    return 0;
}