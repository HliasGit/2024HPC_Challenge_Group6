#include "../inc/conjugate_gradients.hpp"

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



double dot(const double * x, const double * y, size_t size, int world_rank)
{
    double result = 0.0;
    double partial_result_slave = 0.0;
    double partial_result_master = 0.0;
    if(world_rank == 0){
        for(size_t i = 0; i < size/2; i++)
        {
            partial_result_master += x[i] * y[i];
        }

        MPI_Recv(&partial_result_slave, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result = partial_result_master + partial_result_slave;
        return result;
    } else {
        for(size_t i = size/2; i < size; i++)
        {
            partial_result_slave += x[i] * y[i];
        }
        MPI_Send(&partial_result_slave, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;

    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}



void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error, int world_rank)
{
    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;

    double *first_half_rhs = new double[size/2+1];;
    double *second_half_rhs = new double[size/2+1];;

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
        if(i<(size/2)){
            first_half_rhs[i] = b[i];
        } else{
            second_half_rhs[i-(size/2)] = b[i];
        }
    }

    std::cout << "da Entrambi?  " << first_half_rhs[100] << std::endl;

    /////////
    /*
    int number;
    if (world_rank == 0) {
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        MPI_Send(&second_half_rhs, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

    } else if (world_rank != 0) {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        std::cout << "Process " << world_rank << " received number " << number << " from process 0" << std::endl;
    }
    */

    bb = dot(b, b, size, world_rank);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dot(p, Ap, size, world_rank);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size, world_rank);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby(1.0, r, beta, p, size);
    }

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