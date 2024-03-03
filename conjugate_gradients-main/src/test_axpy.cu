#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main(void)
{
    const int nvals = 2;
    const size_t sz = sizeof(double) * (size_t)nvals;
    double x[nvals], y[nvals];
    double *x_, *y_, *result_;
    double result=0., resulth=0.;

    // for(int i=0; i<nvals; i++) {
    //     x[i] = y[i] = 1.0;
    //     resulth += x[i] * y[i];
    // }
    x[0] = -0.569;
    x[1] = +0.680;

    y[0] = +0.014;
    y[1] = 0.012;

    double * alpha_, * beta_;
    double alpha = 1, beta = 1;

    cublasHandle_t h;
    cublasStatus_t stat;
    cublasCreate(&h);
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    
    cudaMalloc( (void **)(&x_), sz);
    cudaMalloc( (void **)(&y_), sz);

    cudaMalloc( (void **)(&alpha_), sizeof(double));


    cudaMemcpy(x_, x, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sz, cudaMemcpyHostToDevice);

    cudaMemcpy(alpha_, &alpha, sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(y, y_, sz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nvals; i++)
        printf("%f\n", y[i]);
    printf("\n");

    stat = cublasDaxpy(h, nvals, alpha_, x_, 1, y_, 1);
    {    
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
    }
    cudaMemcpy(y, y_, sz, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nvals; i++)
        printf("%f\n", y[i]);
    printf("\n");

    cudaMemcpy(x, x_, sz, cudaMemcpyDeviceToHost);

    cublasDestroy(h);
    cudaFree(alpha_);

    return 0;
}