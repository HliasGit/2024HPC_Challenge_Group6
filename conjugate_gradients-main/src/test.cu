//Example 2. Application Using C and cuBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 2
#define N 2


int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA, *devPtrb, *devPtrc;
    float* a = 0;
    float* b = 0, *c;
    int size = N;
    a = (float *) malloc (M * N * sizeof (*a));
    b = (float *) malloc (N * sizeof (*a));
    c = (float *) malloc (N * sizeof (*a));
    for (size_t i= 0; i<M; i++)
        for (size_t j=0; j<N; j++)
            a[i*N + j] = 1;
    for (size_t i= 0; i<M; i++)
        b[i] = 1.0;
    for (size_t i= 0; i<M; i++)
        c[i] = 1.0;

    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, N*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrb, N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrc, N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    stat = cublasSetMatrix (M, N, sizeof(float), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        printf("Error in A\n");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaMemcpy(devPtrb, b, size * sizeof(float), cudaMemcpyHostToDevice);
    // stat = cublasSetVector(size, sizeof(float), b, 1, devPtrb, 1);
    // if (stat != CUBLAS_STATUS_SUCCESS) {
    //     printf ("data download failed");
    //     printf("Error in b\n%d", stat);
    //     cudaFree (devPtrA);
    //     cublasDestroy(handle);
    //     return EXIT_FAILURE;
    // }
    cudaMemcpy(devPtrc, c, size * sizeof(float), cudaMemcpyHostToDevice);
    // stat = cublasSetVector (size, sizeof(float), c, 1, devPtrc, 1);
    // if (stat != CUBLAS_STATUS_SUCCESS) {
    //     printf ("data download failed");
    //     printf("Error in c\n");
    //     cudaFree (devPtrA);
    //     cublasDestroy(handle);
    //     return EXIT_FAILURE;
    // }
    float alpha = 1, *d_alpha;
    float beta = 1, *d_beta;
    cudaStat = cudaMalloc ((void**)&d_alpha, sizeof(*a));
    cudaStat = cudaMalloc ((void**)&d_beta, sizeof(*a));

    cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, &beta,  sizeof(float), cudaMemcpyHostToDevice);

    stat = cublasSgemv(handle, CUBLAS_OP_N, size, size, d_alpha, devPtrA, size, devPtrb, 1, d_beta, devPtrc, 1);

    cudaMemcpy(b, devPtrb, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, devPtrc, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Stampiamo b:\n");
    for (j = 0; j < N; j++) {
        printf ("%7.0f", b[i]);
    }
    printf("\n");
    printf("Stampiamo c:\n");
    for (j = 0; j < N; j++) {
        printf ("%7.0f", c[i]);
    }
    printf("\n");
    if (stat == CUBLAS_STATUS_SUCCESS) {
        printf("La chiamata a cublasSgemv è stata eseguita con successo.\n");
    } else if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
        printf("Errore: cuBLAS non è stato inizializzato correttamente.\n");
    } else if (stat == CUBLAS_STATUS_INVALID_VALUE) {
        printf("Errore: uno o più parametri di input sono invalidi.\n");
    } else if (stat == CUBLAS_STATUS_ALLOC_FAILED) {
        printf("Errore: l'allocazione di memoria sulla GPU ha fallito.\n");
    }


    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasGetMatrix (2, 0, sizeof(*a), devPtrb, M, b, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaMemcpy(devPtrb, b, size * sizeof(float), cudaMemcpyDeviceToHost);
    // stat = cublasGetVector (M, sizeof(*a), devPtrb, 1, b, 1);
    // if (stat != CUBLAS_STATUS_SUCCESS) {
    //     printf ("data upload failed");
    //     cudaFree (devPtrA);
    //     cublasDestroy(handle);
    //     return EXIT_FAILURE;
    // }

    stat = cublasGetVector (M, sizeof(*a), devPtrc, 1, c, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cudaFree (devPtrb);
    cudaFree (devPtrc);
    cublasDestroy(handle);
    printf("\n");
    free(a);
    return EXIT_SUCCESS;
}