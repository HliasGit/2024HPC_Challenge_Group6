#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <mkl.h> // Include Intel MKL header
#include <mpi.h>
#include <vector>


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



void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{
    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    size_t local_size = size / size_mpi;
    size_t residual = size % size_mpi;

    double *local_A = new double[local_size * size];
    double *local_b = new double[local_size];
    double *local_x = new double[local_size];
    double *local_r = new double[local_size];
    double *local_p = new double[local_size];
    double *local_Ap = new double[local_size];

    MPI_Scatter(A, local_size * size, MPI_DOUBLE, local_A, local_size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_size, MPI_DOUBLE, local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double alpha, beta, bb, rr, rr_new;
    double global_rr_new, global_rr;
    int num_iters;

    // Initialize local vectors
    for (size_t i = 0; i < local_size; i++)
    {
        local_x[i] = 0.0;
        local_r[i] = local_b[i];
        local_p[i] = local_b[i];
    }

    bb = cblas_ddot(size, b, 1, b, 1);
    MPI_Allreduce(&bb, &global_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rr = global_rr;
    
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, local_size, size, 1.0, local_A, size, local_p, 1, 0.0, local_Ap, 1);

        // Compute alpha
        double local_pAp = cblas_ddot(local_size, local_p, 1, local_Ap, 1);
        MPI_Allreduce(&local_pAp, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = rr / alpha;

        // Update x and r locally
        cblas_daxpy(local_size, alpha, local_p, 1, local_x, 1);
        cblas_daxpy(local_size, -alpha, local_Ap, 1, local_r, 1);

        // Compute global rr_new
        double local_rr_new = cblas_ddot(local_size, local_r, 1, local_r, 1);
        MPI_Allreduce(&local_rr_new, &global_rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        rr_new = global_rr_new;

        // Check convergence
        if (std::sqrt(rr_new / global_rr) < rel_error)
            break;

        // Compute beta
        beta = rr_new / rr;
        rr = rr_new;

        // Update p locally
        cblas_dscal(local_size, beta, local_p, 1);
        cblas_daxpy(local_size, 1.0, local_r, 1, local_p, 1);
    }

    delete[] local_A;
    delete[] local_b;
    delete[] local_x;
    delete[] local_r;
    delete[] local_p;
    delete[] local_Ap;

    if (num_iters <= max_iters)
    {
        if (rank == 0)
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr_new / global_rr));
    }
    else
    {
        if (rank == 0)
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr_new / global_rr));
    }
}





int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();
    
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
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);
    if (rank == 0) {
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
    }

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("The program is executed in: %f seconds\n", end_time - start_time);
        printf("Finished successfully\n");
    }

    MPI_Finalize();

    return 0;
}
