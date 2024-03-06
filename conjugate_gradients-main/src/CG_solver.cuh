#include "utilities.cuh"
#define BLOCK_DIM_VET 32

void conjugate_gradients(const double * d_A, double * d_x, double * d_p, double * d_r,  size_t size, int max_iters, double rel_error)
{
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

    // Host relative residual
    double h_bb;
    double h_rr;

    // Allocate device memory 
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
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) 
    {
        printf ("CUBLAS initialization failed\n");
        return;
    }
    // Set only GPU pointers for cuBLAS functions
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    // Choose block and grid dimensions
    dim3 vec_block_dim(BLOCK_DIM_VET);
	dim3 vec_grid_dim((size + BLOCK_DIM_VET - 1) / BLOCK_DIM_VET);
    // Set alpha = 1 and beta = 0
    init_alpha_beta<<<1, 1>>>(a, b);
    // Init solver with x0 = 0 and p = r
    init_cg_solver <<<vec_grid_dim, vec_block_dim>>> (d_x, d_p, d_r, size);
    
    // Coompute r * r = b*b
    cublasDdot(handle, size, d_r, 1, d_r, 1, d_rr);
    copy_value <<<1,1>>> (d_bb, d_rr);
    cudaMemcpy(&h_bb, d_bb, sizeof(double), cudaMemcpyDeviceToHost);

    int num_iters;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // alpha(k) = rr / p A p
        cublasDgemv(handle, CUBLAS_OP_N, size, size, a, d_A, size, d_p, 1, b, d_Ap, 1);
        cublasDdot(handle, size, d_p, 1, d_Ap, 1, tmp);
        a_frac_b<<<1, 1>>> (d_rr, tmp, d_alpha);
        // Compute -alpha
        inv_alpha<<<1, 1>>> (d_alpha, d_alpha_);

        // x(k+1) = x(k) + alpha * p
        cublasDaxpy(handle, size, d_alpha, d_p, 1, d_x, 1);

        //r(k+1) = r(k) - alpha * A * p
        cublasDaxpy(handle, size, d_alpha_, d_Ap, 1, d_r, 1);

        //beta(k) = r(k+1)r(k+1) / r(k)r(k)
        cublasDdot(handle, size, d_r, 1, d_r, 1, d_rr_new);
        a_frac_b <<<1, 1>>> (d_rr_new, d_rr, d_beta);

        // Update d_rr
        copy_value <<<1, 1>>> (d_rr, d_rr_new);

        // Synchronize host's relative residuals
        cudaMemcpy(&h_rr, d_rr, sizeof(double), cudaMemcpyDeviceToHost);
        // Stopping criteria
        if(std::sqrt(h_rr / h_bb) < rel_error) 
            break; 
        // p(k+1) = r(k+1) + beta * p(k)
        scalar_vet <<<vec_grid_dim, vec_block_dim>>> (d_beta, d_p, size);
        cublasDaxpy(handle, size, a, d_r, 1, d_p, 1);
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