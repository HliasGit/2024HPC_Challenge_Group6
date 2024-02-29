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

double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

double dotP(const double * x, const double * y, size_t size) {
    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    // Compute info to split the vector
    size_t local_size = size / size_mpi;
    size_t start = rank * local_size;
    size_t end = (rank == size_mpi - 1) ? size : start + local_size;

    //printf("%d) local_size = %d, start = %d, end = %d\n\n", rank, local_size, start, end);

    double local_sum = 0.0;

    #pragma omp parallel for reduction(+:local_sum)
    for(size_t i = start; i < end; i++) {
        local_sum += x[i] * y[i];
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //printf("%d) reduced  %f\n",rank,  global_sum);

    return global_sum;
}

void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void axpbyP(double alpha, const double * x, double beta, double * y, size_t size) {
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++) {
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

void gemvP(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols) {
    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    size_t local_num_rows = num_rows / size_mpi;
    size_t start_row = rank * local_num_rows;
    size_t end_row = (rank + 1) * local_num_rows;
    if (rank == size_mpi - 1) end_row = num_rows;

    double *local_y = new double[local_num_rows];
    for(size_t r = start_row; r < end_row; r++) {
        double y_val = 0.0;
        
        #pragma omp parallel for reduction(+:y_val)
        for(size_t c = 0; c < num_cols; c++) {
            y_val += A[r * num_cols + c] * x[c];
        }
        local_y[r - start_row] = beta * y[r] + alpha * y_val;
    }

    // Collect partial computations
    MPI_Allgather(local_y, local_num_rows, MPI_DOUBLE, y, local_num_rows, MPI_DOUBLE, MPI_COMM_WORLD);

    delete[] local_y;
}

void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;

    if(rank == 0) {
        for(size_t i = 0; i < size; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
        }
    }

    // Every process needs the local initial data
    MPI_Bcast(x, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(r, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    bb = dotP(b, b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++) {
        gemvP(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dotP(p, Ap, size);
        axpbyP(alpha, p, 1.0, x, size);
        axpbyP(-alpha, Ap, 1.0, r, size);
        rr_new = dotP(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(sqrt(rr / bb) < rel_error) { break; }
        axpbyP(1.0, r, beta, p, size);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if(rank==0){
        std::cout << "n iter " << num_iters << std::endl;
    }
}