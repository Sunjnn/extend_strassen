#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 1024
#define K 1024
#define N 1024


void initMatrix(float *A, int m, int n) {
    for (int i = 0; i < m * n; ++i) {
        A[i] = i;
    }
    return;
}

void gemm(float *C, float *A, float *B, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                C[i * m + j] += A[i * m + l] * B[l * k + j];
            }
        }
    }
}

void test(float *A, float *B, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i * m + j] - B[i * m + j] > 0.001 || A[i * m + j] - B[i * m + j] < -0.001) {
                printf("wrong result. i:%d j:%d\n", i, j);
            }
        }
    }
}

int main() {
    float *h_A = (float*)malloc(sizeof(float) * M * K);
    float *h_B = (float*)malloc(sizeof(float) * K * N);
    float *h_C = (float*)malloc(sizeof(float) * M * N);
    float *h_CTest = (float*)malloc(sizeof(float) * M * N);

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    float *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cublasSetMatrix(M, N, sizeof(float), h_A, M, d_A, M);
    cublasSetMatrix(M, N, sizeof(float), h_B, K, d_B, K);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 10.f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);

    gemm(h_CTest, h_A, h_B, M, K, N);

    test(h_C, h_CTest, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CTest);

    return 0;
}