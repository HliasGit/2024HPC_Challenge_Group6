#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main(void)
{
    const int nvals = 2;
    const size_t sz = sizeof(double) * (size_t)nvals;
    double x[nvals], y[nvals];
    double A[nvals * nvals];

    double alpha = 1, beta = 1;
    double *alpha_, *beta_;

    double *A_;
    double *x_, *y_;

    for (int i = 0; i < nvals; i++)
    {
        x[i] = y[i] = 1.0;

        for (int j = 0; j < nvals; j++)
        {
            A[i * nvals + j] = 1.0;
        }
    }

    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);

    cudaMalloc((void **)(&x_), sz);
    cudaMalloc((void **)(&y_), sz);
    cudaMalloc((void **)(&A_), sz * nvals);
    cudaMalloc((void **)(&alpha_), sizeof(double));
    cudaMalloc((void **)(&beta_), sizeof(double));

    cudaMemcpy(x_, x, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_, &beta, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_, A, sz * nvals, cudaMemcpyHostToDevice);

    cublasStatus_t stat = cublasDgemv(h, CUBLAS_OP_N, nvals, nvals, alpha_, A_, nvals, x_, 1, beta_, y_, 1);

    if (stat == CUBLAS_STATUS_SUCCESS)
    {
        printf("La chiamata a cublasDgemv è stata eseguita con successo.\n");
    }
    else if (stat == CUBLAS_STATUS_NOT_INITIALIZED)
    {
        printf("Errore: cuBLAS non è stato inizializzato correttamente.\n");
    }
    else if (stat == CUBLAS_STATUS_INVALID_VALUE)
    {
        printf("Errore: uno o più parametri di input sono invalidi.\n");
    }
    else if (stat == CUBLAS_STATUS_ALLOC_FAILED)
    {
        printf("Errore: l'allocazione di memoria sulla GPU ha fallito.\n");
    }

    cudaMemcpy(y, y_, sz, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nvals; i++)
        printf("%f\n", y[i]);

    cublasDestroy(h);

    cudaFree(x_);
    cudaFree(y_);
    cudaFree(A_);
    cudaFree(alpha_);
    cudaFree(beta_);

    return 0;
}