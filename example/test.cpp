#include "../inc/conjugate_gradients.hpp"

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    double start_time = MPI_Wtime();

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

    double * matrix;
    double * rhs;
    size_t size;
    size_t matrix_rows;
    size_t matrix_cols;
    size_t rhs_rows;
    size_t rhs_cols;

    if(rank == 0)
    {
        printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
        printf("All parameters are optional and have default values\n");
        printf("\n");

        printf("Command line arguments:\n");
        printf("  input_file_matrix: %s\n", input_file_matrix);
        printf("  input_file_rhs:    %s\n", input_file_rhs);
        printf("  output_file_sol:   %s\n", output_file_sol);
        printf("  max_iters:         %d\n", max_iters);
        printf("  rel_error:         %e\n", rel_error);
        printf("\n");

        printf("Reading matrix from file ...\n");
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
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

    // Broadcast of matrix rows, cols and size
    MPI_Bcast(&matrix_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_cols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        // Allocate the space for the non-master processes
        matrix = new double[size * size]; // Assuming square matrix
        rhs = new double[size];
    }

    // Broadcast the data to all the processes
    MPI_Bcast(matrix, matrix_rows * matrix_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rhs, matrix_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
    double * sol = new double[size];
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);

    if(rank == 0){
        printf("Writing solution to file ...\n");
        bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
        if(!success_write_sol)
        {
            fprintf(stderr, "Failed to save solution\n");
            return 6;
        }
        printf("Done\n");
        printf("\n");

        printf("Finished successfully\n");
    }

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("The program is executed in: %f seconds\n", end_time - start_time);
        printf("Finished successfully\n");
    }

    return 0;
}