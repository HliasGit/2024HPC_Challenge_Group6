#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <mkl.h> // Include Intel MKL header
#include <mpi.h>
#include <vector>



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



bool read_vector_from_file(const char * filename, double ** vector_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * vector;
    size_t num_rows;
    size_t num_cols;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    // Read the header containing the number of rows and columns
    MPI_Offset displacement = 0;
    MPI_File_read_at(file, displacement, &num_rows, 1, MPI_UNSIGNED_LONG_LONG, MPI_STATUS_IGNORE);
    displacement += sizeof(size_t);
    MPI_File_read_at(file, displacement, &num_cols, 1, MPI_UNSIGNED_LONG_LONG, MPI_STATUS_IGNORE);
    displacement += sizeof(size_t);

    // Set the vector
    int vector_size = num_rows * num_cols;
    vector = new double[vector_size];

    // Read the portion of the vector assigned to this process
    MPI_File_read_at(file, displacement, vector, vector_size, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&file);

    *vector_out = vector;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    return true;
}



bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    // Read the header containing the number of rows and columns
    MPI_Offset displacement = 0;
    MPI_File_read_at(file, displacement, &num_rows, 1, MPI_UNSIGNED_LONG_LONG, MPI_STATUS_IGNORE);
    displacement += sizeof(size_t);
    MPI_File_read_at(file, displacement, &num_cols, 1, MPI_UNSIGNED_LONG_LONG, MPI_STATUS_IGNORE);
    displacement += sizeof(size_t);

    // Calculate the number of rows each process will read
    MPI_Offset block_row_size = num_rows / size;
    MPI_Offset remainder_row_size = num_rows % size;

    // Adjust block size for processes with rank < remainder
    if (rank < remainder_row_size) {
        block_row_size++;
        displacement += rank * block_row_size * num_cols * sizeof(double);
    } else {
        displacement += (rank * block_row_size + remainder_row_size) * num_cols * sizeof(double);
    }

    // Set the matrix
    int matrix_size = block_row_size * num_cols;
    matrix = new double[matrix_size];

    // Read the portion of the matrix assigned to this process
    MPI_File_read_at(file, displacement, matrix, matrix_size, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&file);

    *matrix_out = matrix;
    *num_rows_out = block_row_size;
    *num_cols_out = num_cols;

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



void conjugate_gradients(const double * A, const double * b, double * x, size_t local_size, size_t size, int max_iters, double rel_error)
{
    // Initialize variables
    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size]; // Residual vector
    double * p = new double[size]; // Search direction vector
    double * Ap = new double[size]; // Temporary vector to store result of A*p
    double * Ap_small = new double[local_size]; // Temporary vector to store portion of Ap computed by each MPI process
    bool continue_execution = true; // Flag to indicate whether to continue execution or not
    int num_iters; // Number of iterations performed
    
    // MPI setup
    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi); 

    // Initialize solution vector x, residual vector r, and search direction vector p
    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    // Compute inner product of b with itself
    bb = cblas_ddot(size, b, 1, b, 1);
    rr = bb; // Initialize residual norm


    // Main iteration loop
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // Compute A*p
        cblas_dgemv(CblasRowMajor, CblasNoTrans, local_size, size, 1.0, A, size, p, 1, 0.0, Ap_small, 1);
        
        // Gather Ap_small computed by each MPI process into Ap
        int* displacements = new int[size_mpi];
        int* recvcounts = new int[size_mpi];
        int elements_per_process = size / size_mpi;
        int remainder = size % size_mpi;
        int displacement = 0; // Initialize global displacement

        // Determine recvcounts and displacements for MPI_Gatherv   
        for (int i = 0; i < size_mpi; i++) {
            recvcounts[i] = elements_per_process;
            if (i < remainder) {
                recvcounts[i]++;
            }
            displacements[i] = displacement; // Displacement for process i
            displacement += recvcounts[i]; // Update global displacement
        }

        // Gather Ap_small from all processes into Ap
        MPI_Gatherv(Ap_small, local_size, MPI_DOUBLE, Ap, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

        // Process 0 updates solution vector x, computes new residual r, and checks convergence
        if(rank == 0){
            alpha = rr / cblas_ddot(size, p, 1, Ap, 1);
            cblas_daxpby(size, alpha, p, 1, 1.0, x, 1);
            cblas_daxpby(size, -alpha, Ap, 1, 1.0, r, 1);
            rr_new = cblas_ddot(size, r, 1, r, 1);
            beta = rr_new / rr;
            rr = rr_new;
            if(std::sqrt(rr / bb) < rel_error) { 
                continue_execution = false; // Signal process 0 to stop execution
            }
            cblas_daxpby(size, 1.0, r, 1, beta, p, 1); // Update search direction vector p
        }
        
        // Broadcast updated p vector to all processes
        MPI_Bcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Check if process 0 signaled to stop execution
        MPI_Bcast(&continue_execution, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (!continue_execution) {
            break; // All processes exit loop
        }
    }

    // Clean up allocated memory
    delete[] Ap_small;
    delete[] r;
    delete[] p;
    delete[] Ap;

    // Print convergence status from process 0
    if(rank == 0){
        if(num_iters <= max_iters)
            {
                printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
            }
            else
            {
                printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
            }
    }
}





int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();
    
    if(rank == 0){
        printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
        printf("All parameters are optional and have default values\n");
        printf("\n");
    }

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

    if(rank == 0){
        printf("Command line arguments:\n");
        printf("  input_file_matrix: %s\n", input_file_matrix);
        printf("  input_file_rhs:    %s\n", input_file_rhs);
        printf("  output_file_sol:   %s\n", output_file_sol);
        printf("  max_iters:         %d\n", max_iters);
        printf("  rel_error:         %e\n", rel_error);
        printf("\n");
    }


    double * matrix;
    double * rhs;
    size_t size, global_size;

    {
        if(rank == 0){
            printf("Reading matrix from file ...\n");
        }

        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        if(rank == 0){
            printf("Done\n\n");
            printf("Reading right hand side from file ...\n");
        }

        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_vector_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }

        if(rank == 0){
            printf("Done\n\n");
        }

        if(matrix_rows > matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if(rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        size = matrix_rows;
        global_size = matrix_cols;
    }

    if(rank == 0){
        printf("Solving the system ...\n");
    }

    double * sol = new double[global_size];
    conjugate_gradients(matrix, rhs, sol, size, global_size, max_iters, rel_error);
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
